```js
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import compression from 'compression';
import { GoogleGenerativeAI } from '@google/generative-ai';

const app = express();

// پایه
app.use(cors());
app.use(compression());
app.use(express.json({ limit: '5mb' }));

// API Keys (چرخشی)
const API_KEYS = process.env.GEMINI_API_KEYS?.split(',') || [];
if (!API_KEYS.length) {
  console.error('❌ GEMINI_API_KEYS لازم است');
  process.exit(1);
}
let currentApiKeyIndex = 0;
function getNextApiKey() {
  const key = API_KEYS[currentApiKeyIndex];
  currentApiKeyIndex = (currentApiKeyIndex + 1) % API_KEYS.length;
  return key;
}

// مدل
const MODEL_ID = process.env.MODEL_ID || 'gemini-1.5-flash';

// کش ساده
const cache = new Map();
const CACHE_TTL = 2 * 60 * 1000; // ۲ دقیقه
function getCacheKey(message) {
  return message.trim().toLowerCase();
}
function getFromCache(key) {
  const item = cache.get(key);
  if (item && Date.now() - item.ts < CACHE_TTL) {
    return item.data;
  }
  return null;
}
function setCache(key, data) {
  cache.set(key, { data, ts: Date.now() });
}

// health check
app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    model: MODEL_ID,
    timestamp: new Date().toISOString(),
    cacheSize: cache.size,
    apiKeysCount: API_KEYS.length,
    currentApiKeyIndex
  });
});

// تابع اصلی چت
async function handleChat(req, res) {
  try {
    const { message } = req.body || {};
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'message الزامی است', received: typeof message });
    }

    if (message.length > 5000) {
      return res.status(400).json({ error: 'پیام بیش از حد طولانی است', length: message.length });
    }

    const cacheKey = getCacheKey(message);
    const cached = getFromCache(cacheKey);
    if (cached) {
      return res.json({
        ...cached,
        fromCache: true,
        timestamp: new Date().toISOString()
      });
    }

    const apiKey = getNextApiKey();
    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: MODEL_ID });

    const result = await model.generateContent(message);
    const text = result.response.text() || '';

    if (!text.trim()) {
      return res.status(500).json({ error: 'پاسخ خالی از مدل دریافت شد' });
    }

    const responseData = {
      reply: text,
      nextHistoryItem: { role: 'model', parts: text },
      timestamp: new Date().toISOString(),
      model: MODEL_ID
    };

    setCache(cacheKey, responseData);

    res.json({ ...responseData, fromCache: false });
  } catch (err) {
    console.error('❌ Chat Error:', err);
    res.status(500).json({
      error: 'خطا در پردازش',
      message: process.env.NODE_ENV === 'development' ? err.message : undefined
    });
  }
}

// مسیرها (سازگار با نسخه قبلی)
app.post('/api/chat', handleChat);
app.post('/api/doctor-chat', handleChat);

// start
const port = process.env.PORT || 3000;
const server = app.listen(port, () => {
  console.log(`🚀 Chat API در حال اجرا: http://localhost:${port}`);
  console.log(`📊 Health check: http://localhost:${port}/health`);
  console.log(`🤖 مدل فعلی: ${MODEL_ID}`);
  console.log(`🔑 تعداد کلیدهای API: ${API_KEYS.length}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('📥 SIGTERM دریافت شد، در حال خاموش کردن سرور...');
  server.close(() => console.log('✅ سرور خاموش شد'));
});
process.on('SIGINT', () => {
  console.log('📥 SIGINT دریافت شد، در حال خاموش کردن سرور...');
  server.close(() => console.log('✅ سرور خاموش شد'));
});

export default app;
```
