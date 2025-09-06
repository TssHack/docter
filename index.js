import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from '@google/generative-ai';

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' })); // اضافه کردن محدودیت حجم

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('❌ GEMINI_API_KEY is missing in .env');
  process.exit(1);
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

const genAI = new GoogleGenerativeAI(apiKey);

const safetySettings = [
  { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
  { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
];

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
    version: '1.0.0'
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

    // محدود کردن تاریخچه
    const limitedHistory = history.slice(-20); // فقط ۲۰ پیام آخر

    const model = genAI.getGenerativeModel({
      model: MODEL_ID,
      systemInstruction: SYSTEM_INSTRUCTION,
    });

    // تبدیل فرمت تاریخچه
    const formattedHistory = limitedHistory.map(m => {
      if (!m.role || !m.parts) {
        throw new Error('هر آیتم در history باید role و parts داشته باشد');
      }
      return {
        role: m.role === 'assistant' ? 'model' : m.role, // تبدیل assistant به model
        parts: [{ text: typeof m.parts === 'string' ? m.parts : JSON.stringify(m.parts) }],
      };
    });

    const chat = model.startChat({
      safetySettings,
      history: formattedHistory,
      generationConfig: {
        temperature: 0.3, // کمی کاهش برای پاسخ‌های پزشکی دقیق‌تر
        topK: 40,
        topP: 0.8,
        maxOutputTokens: 2048, // افزایش برای پاسخ‌های کاملتر
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
        res.write(`خطا در پردازش: ${streamError.message}`);
        return res.end();
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

      return res.json({
        reply: text,
        nextHistoryItem: { role: 'model', parts: text },
        safety: response.candidates?.[0]?.safetyRatings || null,
        timestamp: new Date().toISOString(),
        model: MODEL_ID,
        tokensUsed: response.usageMetadata || null
      });
    }
  } catch (err) {
    console.error('خطا در handleDoctorChat:', err);
    
    // بررسی انواع خطاهای مختلف
    if (err.message?.includes('API_KEY')) {
      return res.status(401).json({ 
        error: 'مشکل در کلید API',
        code: 'API_KEY_ERROR'
      });
    }
    
    if (err.message?.includes('quota')) {
      return res.status(429).json({ 
        error: 'محدودیت استفاده از API',
        code: 'QUOTA_EXCEEDED'
      });
    }
    
    if (err.message?.includes('safety')) {
      return res.status(400).json({ 
        error: 'پیام شما شامل محتوای نامناسب است',
        code: 'SAFETY_FILTER'
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

// راه‌اندازی سرور
const port = process.env.PORT || 3000;
const server = app.listen(port, () => {
  console.log(`🚀 Doctor Chat API در حال اجرا: http://localhost:${port}`);
  console.log(`📊 Health check: http://localhost:${port}/health`);
  console.log(`🤖 مدل فعلی: ${MODEL_ID}`);
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
