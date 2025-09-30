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

// API Keys
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

// مدل سریع
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
  res.json({ ok: true, model: MODEL_ID, ts: Date.now(), cacheSize: cache.size });
});

// تابع اصلی چت
async function handleChat(req, res) {
  try {
    const { message } = req.body || {};
    if (!message) return res.status(400).json({ error: 'message الزامی است' });

    const cacheKey = getCacheKey(message);
    const cached = getFromCache(cacheKey);
    if (cached) {
      return res.json({ reply: cached, fromCache: true });
    }

    const apiKey = getNextApiKey();
    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: MODEL_ID });

    const result = await model.generateContent(message);
    const text = result.response.text() || '';

    setCache(cacheKey, text);

    res.json({ reply: text, fromCache: false });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'خطا در پردازش', msg: err.message });
  }
}

// مسیر جدید
app.post('/api/chat', handleChat);

// مسیر قدیمی برای هماهنگی
app.post('/api/doctor-chat', handleChat);

// start
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`🚀 Server at http://localhost:${port}`);
});
