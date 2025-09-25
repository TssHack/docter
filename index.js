import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import slowDown from 'express-slow-down';
import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from '@google/generative-ai';

const app = express();
app.use(cors());
app.use(compression()); // فشرده‌سازی پاسخ‌ها برای افزایش سرعت
app.use(express.json({ limit: '10mb' }));

// محدودیت نرخ درخواست برای جلوگیری از سوءاستفاده
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 دقیقه
  max: 100, // حداکثر 100 درخواست در هر 15 دقیقه
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: 'تعداد درخواست‌های شما بیش از حد مجاز است',
    code: 'RATE_LIMIT_EXCEEDED'
  }
});

// کاهش سرعت برای جلوگیری از حملات
const speedLimiter = slowDown({
  windowMs: 15 * 60 * 1000, // 15 دقیقه
  delayAfter: 50, // بعد از 50 درخواست شروع به کاهش سرعت کن
  delayMs: (hits) => hits * 100, // هر درخواست اضافی 100ms تاخیر داشته باشد
});

app.use('/api/', limiter);
app.use('/api/', speedLimiter);

// سیستم مدیریت لایسنس‌های چرخشی
const API_KEYS = process.env.GEMINI_API_KEYS ? process.env.GEMINI_API_KEYS.split(',') : [];
if (!API_KEYS.length || !API_KEYS[0]) {
  console.error('❌ GEMINI_API_KEYS is missing in .env');
  process.exit(1);
}

let currentApiKeyIndex = 0;

function getNextApiKey() {
  const key = API_KEYS[currentApiKeyIndex];
  currentApiKeyIndex = (currentApiKeyIndex + 1) % API_KEYS.length;
  return key;
}

// سیستم کش ساده برای پاسخ‌های تکراری
const responseCache = new Map();
const CACHE_TTL = 5 * 60 * 1000; // 5 دقیقه

function getCacheKey(message, history) {
  return JSON.stringify({ 
    message, 
    history: history.slice(-5) // فقط 5 پیام آخر را در نظر بگیرید
  });
}

function getFromCache(key) {
  const item = responseCache.get(key);
  if (item && Date.now() - item.timestamp < CACHE_TTL) {
    return item.data;
  }
  return null;
}

function setCache(key, data) {
  responseCache.set(key, {
    data,
    timestamp: Date.now()
  });
}

const MODEL_ID = process.env.MODEL_ID || 'gemini-1.5-pro';

const SYSTEM_INSTRUCTION = `
شما یک دکتر مجاز هستید که به زبان فارسی پاسخ می‌دهید.
اهداف:
- پاسخ‌های واضح، علمی و گام به گام ارائه دهید
- در صورت کمبود اطلاعات، سوالات کوتاه و مفید بپرسید (حداکثر ۳ سوال)
- در مواقع لازم تشخیص‌های افتراقی ارائه دهید
- علائم خطر و زمان مراجعه اورژانسی را توضیح دهید

ایمنی:
- این اطلاعات عمومی است، نه تشخیص یا جایگزین ویزیت پزشک
- تشخیص قطعی ندهید و داروهای کنترل شده تجویز نکنید
- در صورت وجود علائم اورژانسی (درد قفسه سینه، علائم سکته، آنافیلاکسی، افکار خودکشی) فوراً توصیه به تماس با اورژانس کنید
- مراجعه حضوری به پزشک متخصص را توصیه کنید
`;

// middleware برای error handling
app.use((err, req, res, next) => {
  console.error('Express Error:', err);
  res.status(500).json({ 
    error: 'خطای داخلی سرور', 
    message: process.env.NODE_ENV === 'development' ? err.message : undefined 
  });
});

// health check
app.get('/health', (_req, res) => {
  res.json({ 
    ok: true, 
    model: MODEL_ID,
    timestamp: new Date().toISOString(),
    version: '1.0.0',
    apiKeysCount: API_KEYS.length,
    currentApiKeyIndex: currentApiKeyIndex
  });
});

// POST endpoint
app.post('/api/doctor-chat', async (req, res) => {
  const { message, history = [], stream = false } = req.body || {};
  return handleDoctorChat({ message, history, stream }, res);
});

// GET endpoint
app.get('/api/doctor-chat', async (req, res) => {
  const { message, stream } = req.query;
  
  // بررسی وجود پیام
  if (!message) {
    return res.status(400).json({ 
      error: 'پارامتر message الزامی است',
      example: '/api/doctor-chat?message=سه روزه سرفه دارم'
    });
  }

  let history = [];
  try {
    history = req.query.history ? JSON.parse(decodeURIComponent(req.query.history)) : [];
  } catch (e) {
    return res.status(400).json({ 
      error: 'فرمت history نامعتبر است. باید JSON معتبر باشد',
      received: req.query.history
    });
  }

  return handleDoctorChat(
    { 
      message: decodeURIComponent(message), 
      history, 
      stream: stream === 'true' 
    }, 
    res
  );
});

// تابع اصلی پردازش
async function handleDoctorChat({ message, history, stream }, res) {
  try {
    // اعتبارسنجی ورودی
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ 
        error: 'message (string) الزامی است',
        received: typeof message
      });
    }

    if (message.length > 5000) {
      return res.status(400).json({ 
        error: 'پیام نباید بیشتر از ۵۰۰۰ کاراکتر باشد',
        length: message.length
      });
    }

    // اعتبارسنجی تاریخچه
    if (!Array.isArray(history)) {
      return res.status(400).json({ 
        error: 'history باید آرایه باشد',
        received: typeof history
      });
    }

    // اعتبارسنجی هر آیتم در تاریخچه
    for (const item of history) {
      if (!item.role || !item.parts) {
        return res.status(400).json({ 
          error: 'هر آیتم در history باید role و parts داشته باشد',
          invalidItem: item
        });
      }
      
      if (!['user', 'assistant', 'model'].includes(item.role)) {
        return res.status(400).json({ 
          error: 'نقش باید یکی از مقادیر user، assistant یا model باشد',
          invalidRole: item.role
        });
      }
    }

    // محدود کردن تاریخچه
    const limitedHistory = history.slice(-20); // فقط ۲۰ پیام آخر

    // بررسی کش
    const cacheKey = getCacheKey(message, limitedHistory);
    const cachedResponse = getFromCache(cacheKey);

    if (cachedResponse && !stream) {
      return res.json({
        ...cachedResponse,
        fromCache: true,
        timestamp: new Date().toISOString()
      });
    }

    // استفاده از لایسنس چرخشی
    const apiKey = getNextApiKey();
    const genAI = new GoogleGenerativeAI(apiKey);

    const model = genAI.getGenerativeModel({
      model: MODEL_ID,
      systemInstruction: SYSTEM_INSTRUCTION,
    });

    // تبدیل فرمت تاریخچه با مدیریت بهتر
    const formattedHistory = limitedHistory.map(m => {
      let textContent = '';
      if (typeof m.parts === 'string') {
        textContent = m.parts;
      } else if (Array.isArray(m.parts)) {
        textContent = m.parts.map(part => {
          if (typeof part === 'string') return part;
          if (part && part.text) return part.text;
          return JSON.stringify(part);
        }).join('');
      } else if (m.parts && m.parts.text) {
        textContent = m.parts.text;
      } else {
        textContent = JSON.stringify(m.parts);
      }
      
      return {
        role: m.role === 'assistant' ? 'model' : m.role,
        parts: [{ text: textContent }],
      };
    });

    const safetySettings = [
      { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
      { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
      { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
      { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    ];

    const chat = model.startChat({
      safetySettings,
      history: formattedHistory,
      generationConfig: {
        temperature: 0.3,
        topK: 40,
        topP: 0.8,
        maxOutputTokens: 2048,
        responseMimeType: 'text/plain',
      },
    });

    if (stream) {
      // پردازش stream
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      try {
        const result = await chat.sendMessageStream(message);
        let fullResponse = '';
        
        for await (const chunk of result.stream) {
          const text = chunk.text();
          if (text) {
            fullResponse += text;
            res.write(text);
          }
        }
        
        // اضافه کردن metadata در انتهای stream
        res.write('\n\n---METADATA---\n');
        res.write(JSON.stringify({
          nextHistoryItem: { role: 'model', parts: fullResponse },
          timestamp: new Date().toISOString()
        }));
        
        return res.end();
      } catch (streamError) {
        console.error('Stream Error:', streamError);
        if (!res.headersSent) {
          res.status(500).json({ 
            error: 'خطا در پردازش درخواست',
            message: streamError.message 
          });
        } else {
          res.write(`\n\nخطا در پردازش: ${streamError.message}`);
          return res.end();
        }
      }
    } else {
      // پردازش معمولی
      const result = await chat.sendMessage(message);
      const response = await result.response;
      const text = response.text();
      
      if (!text || text.trim() === '') {
        return res.status(500).json({ 
          error: 'پاسخ خالی از مدل دریافت شد',
          candidates: response.candidates?.length || 0
        });
      }

      const responseData = {
        reply: text,
        nextHistoryItem: { role: 'model', parts: text },
        safety: response.candidates?.[0]?.safetyRatings || null,
        timestamp: new Date().toISOString(),
        model: MODEL_ID,
        tokensUsed: response.usageMetadata || null
      };

      // ذخیره در کش
      setCache(cacheKey, responseData);

      return res.json(responseData);
    }
  } catch (err) {
    console.error('خطا در handleDoctorChat:', err);
    
    // بررسی انواع خطاهای مختلف
    if (err.message?.includes('API_KEY') || err.message?.includes('API key')) {
      return res.status(401).json({ 
        error: 'مشکل در کلید API',
        code: 'API_KEY_ERROR'
      });
    }
    
    if (err.message?.includes('quota') || err.message?.includes('QUOTA_EXCEEDED')) {
      return res.status(429).json({ 
        error: 'محدودیت استفاده از API',
        code: 'QUOTA_EXCEEDED'
      });
    }
    
    if (err.message?.includes('safety') || err.message?.includes('SAFETY')) {
      return res.status(400).json({ 
        error: 'پیام شما شامل محتوای نامناسب است',
        code: 'SAFETY_FILTER'
      });
    }

    if (err.message?.includes('400')) {
      return res.status(400).json({ 
        error: 'درخواست نامعتبر',
        code: 'BAD_REQUEST',
        message: err.message
      });
    }

    return res.status(500).json({ 
      error: 'خطای داخلی سرور',
      message: process.env.NODE_ENV === 'development' ? err.message : 'لطفاً بعداً تلاش کنید',
      code: 'INTERNAL_ERROR'
    });
  }
}

// endpoint برای دریافت لیست مدل‌های موجود
app.get('/api/models', async (_req, res) => {
  try {
    const apiKey = getNextApiKey();
    const genAI = new GoogleGenerativeAI(apiKey);
    const models = await genAI.listModels();
    res.json({
      currentModel: MODEL_ID,
      availableModels: models.map(m => ({
        name: m.name,
        displayName: m.displayName,
        description: m.description
      }))
    });
  } catch (err) {
    res.status(500).json({ 
      error: 'خطا در دریافت لیست مدل‌ها',
      message: err.message 
    });
  }
});

// endpoint برای مدیریت لایسنس‌ها
// افزودن لایسنس جدید
app.post('/api/licenses', (req, res) => {
  const { key } = req.body;
  
  if (!key || typeof key !== 'string') {
    return res.status(400).json({ 
      error: 'کلید (string) الزامی است',
      received: typeof key
    });
  }
  
  if (API_KEYS.includes(key)) {
    return res.status(400).json({ 
      error: 'این کلید از قبل وجود دارد'
    });
  }
  
  API_KEYS.push(key);
  
  res.json({
    success: true,
    message: 'کلید با موفقیت اضافه شد',
    totalKeys: API_KEYS.length
  });
});

// حذف لایسنس
app.delete('/api/licenses/:key', (req, res) => {
  const { key } = req.params;
  
  const index = API_KEYS.indexOf(key);
  if (index === -1) {
    return res.status(404).json({ 
      error: 'کلید مورد نظر یافت نشد'
    });
  }
  
  API_KEYS.splice(index, 1);
  
  // اگر کلید حذف شده همان کلید فعلی بود، ایندکس را تنظیم مجدد کن
  if (currentApiKeyIndex >= API_KEYS.length) {
    currentApiKeyIndex = 0;
  }
  
  res.json({
    success: true,
    message: 'کلید با موفقیت حذف شد',
    totalKeys: API_KEYS.length
  });
});

// دریافت لیست لایسنس‌ها
app.get('/api/licenses', (_req, res) => {
  res.json({
    totalKeys: API_KEYS.length,
    currentKeyIndex: currentApiKeyIndex,
    keys: API_KEYS.map((key, index) => ({
      index,
      key: key.substring(0, 8) + '...', // فقط بخشی از کلید را نمایش بده
      isCurrent: index === currentApiKeyIndex
    }))
  });
});

// راه‌اندازی سرور
const port = process.env.PORT || 3000;
const server = app.listen(port, () => {
  console.log(`🚀 Doctor Chat API در حال اجرا: http://localhost:${port}`);
  console.log(`📊 Health check: http://localhost:${port}/health`);
  console.log(`🤖 مدل فعلی: ${MODEL_ID}`);
  console.log(`🔑 تعداد کلیدهای API: ${API_KEYS.length}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('دریافت سیگنال SIGTERM، در حال خاموش کردن سرور...');
  server.close(() => {
    console.log('سرور با موفقیت خاموش شد');
  });
});

process.on('SIGINT', () => {
  console.log('دریافت سیگنال SIGINT، در حال خاموش کردن سرور...');
  server.close(() => {
    console.log('سرور با موفقیت خاموش شد');
  });
});

export default app;
