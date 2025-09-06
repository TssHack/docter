import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { GoogleGenerativeAI } from '@google/generative-ai';

const app = express();
app.use(cors());
app.use(express.json());

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('❌ GEMINI_API_KEY is missing in .env');
  process.exit(1);
}

const MODEL_ID = process.env.MODEL_ID || 'gemini-1.5-pro';

const SYSTEM_INSTRUCTION = `
You are a licensed medical doctor answering in Persian (fa-IR).
Goals:
- Give clear, medically sound, step-by-step reasoning and concise recommendations.
- Ask brief clarifying questions if info is missing (max 3).
- Provide differential considerations when appropriate.
- Explain red flags and when to seek urgent care.

Safety:
- This is general information, not a diagnosis or substitute for clinical visit.
- Do NOT provide definitive diagnoses or prescribe controlled drugs.
- If emergency symptoms exist (chest pain, stroke signs, anaphylaxis, suicidal intent) → advise calling emergency services immediately.
- Encourage follow-up with a qualified clinician in person.
`;

const genAI = new GoogleGenerativeAI(apiKey);

// تنظیمات ایمنی با رشته ساده (سازگار با نسخه جدید)
const safetySettings = [
  { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_MEDIUM_AND_ABOVE' },
  { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_MEDIUM_AND_ABOVE' }
];

// Health check
app.get('/health', (_req, res) => {
  res.json({ ok: true, model: MODEL_ID });
});

// POST endpoint
app.post('/api/doctor-chat', async (req, res) => {
  const { message, history = [], stream = false } = req.body || {};
  return handleDoctorChat({ message, history, stream }, res);
});

// GET endpoint
app.get('/api/doctor-chat', async (req, res) => {
  const { message, stream } = req.query;
  let history = [];

  try {
    history = req.query.history ? JSON.parse(req.query.history) : [];
  } catch (e) {
    return res.status(400).json({ error: 'Invalid history JSON' });
  }

  return handleDoctorChat(
    { message, history, stream: stream === 'true' },
    res
  );
});

// Handler مشترک
async function handleDoctorChat({ message, history, stream }, res) {
  try {
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'message (string) is required' });
    }

    const model = genAI.getGenerativeModel({
      model: MODEL_ID,
      systemInstruction: SYSTEM_INSTRUCTION,
    });

    const chat = model.startChat({
      safetySettings,
      history: history.map(m => ({ role: m.role, parts: [{ text: m.parts }] })),
      generationConfig: {
        temperature: 0.4,
        topK: 32,
        topP: 0.9,
        maxOutputTokens: 1024,
        responseMimeType: 'text/plain',
      },
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      const result = await chat.sendMessageStream(message);
      for await (const chunk of result.stream) {
        const t = chunk.text();
        if (t) res.write(t);
      }
      return res.end();
    } else {
      const result = await chat.sendMessage(message);
      const response = await result.response;
      const text = response.text();
      return res.json({
        reply: text,
        nextHistoryItem: { role: 'model', parts: text },
        safety: response.candidates?.[0]?.safetyRatings ?? null,
      });
    }
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: err.message || 'Unknown error' });
  }
}

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`🚀 Doctor Chat API listening on http://localhost:${port}`);
});
