import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from openai import AsyncOpenAI
import httpx

# ──────────────────────────────────────
# إعدادات البيئة
# ──────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
HUMAIN_NODE_API_KEY = os.getenv("HUMAIN_NODE_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

HUMAIN_NODE_BASE_URL = "https://api.node.humain.com/v1"

# ──────────────────────────────────────
# التطبيق
# ──────────────────────────────────────
app = FastAPI(title="chat Sa", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────
# موديلات الطلب/الاستجابة
# ──────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    model: str = "auto"  # auto | gemini | deepseek | humain

class ChatResponse(BaseModel):
    response: str
    model_used: str

# ──────────────────────────────────────
# تصنيف الأسئلة
# ──────────────────────────────────────
def classify_query(message: str) -> str:
    """تصنيف نوع السؤال لتحديد النموذج المناسب"""
    cultural_keywords = [
        "أدب", "شعر", "ثقافة", "فصحى", "تراث", "إسلامي",
        "عربي", "نحو", "صرف", "بلاغة", "قرآن", "حديث",
        "سيرة", "تاريخ", "حضارة", "فقه", "لغة عربية"
    ]
    technical_keywords = [
        "كود", "code", "api", "endpoint", "function",
        "bug", "خطأ", "debug", "database", "server",
        "python", "javascript", "react", "docker"
    ]
    creative_keywords = [
        "اكتب", "compose", "write", "صمم", "design",
        "اقتراح", "suggest", "خاطرة", "قصة", "مقال"
    ]

    msg_lower = message.lower()
    cultural_count = sum(1 for kw in cultural_keywords if kw in msg_lower)
    tech_count = sum(1 for kw in technical_keywords if kw in msg_lower)
    creative_count = sum(1 for kw in creative_keywords if kw in msg_lower)

    if cultural_count > tech_count and cultural_count > creative_count:
        return "humain"       # عربي/ثقافي ← HUMAIN M3
    elif tech_count > creative_count:
        return "deepseek"     # تقني/برمجي ← DeepSeek
    else:
        return "gemini"       # عام/إبداعي ← Gemini

# ──────────────────────────────────────
# استدعاء النماذج
# ──────────────────────────────────────
async def call_gemini(message: str) -> str:
    """استدعاء Google Gemini Flash"""
    prompt = f"أنت مساعد ذكي متخصص في الدعم الفني لنماذج الذكاء الاصطناعي. أجب بالعربية الفصحى بشكل واضح ومختصر.\n\nسؤال المستخدم: {message}"
    resp = gemini_model.generate_content(prompt)
    return resp.text


async def call_deepseek(message: str) -> str:
    """استدعاء DeepSeek"""
    resp = await deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "أنت مساعد تقني متخصص في البرمجة وتطوير البرمجيات. أجب بالعربية الفصحى بشكل واضح ودقيق."},
            {"role": "user", "content": message}
        ],
        temperature=0.7,
        max_tokens=1024
    )
    return resp.choices[0].message.content


async def call_humain_m3(message: str) -> str:
    """استدعاء HUMAIN M3 عبر HUMAIN Node"""
    system_prompt = """أنت مساعد متخصص في اللغة العربية الفصحى والثقافة العربية والأدب.
جميع الإجابات تكون بالفصحى المعاصرة الواضحة. ممنوع استخدام العامية."""

    headers = {
        "Authorization": f"Bearer {HUMAIN_NODE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "humain-m3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        "temperature": 0.5,
        "max_tokens": 1024,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{HUMAIN_NODE_BASE_URL}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

# ──────────────────────────────────────
# Endpoints
# ──────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "service": "chat Sa"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """نقطة المحادثة الرئيسية"""
    try:
        selected_model = request.model if request.model != "auto" else classify_query(request.message)

        if selected_model == "gemini":
            response = await call_gemini(request.message)
        elif selected_model == "deepseek":
            response = await call_deepseek(request.message)
        elif selected_model == "humain":
            response = await call_humain_m3(request.message)
        else:
            raise HTTPException(status_code=400, detail="نموذج غير مدعوم")

        return ChatResponse(response=response, model_used=selected_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    checks = {
        "gemini": "configured" if GEMINI_API_KEY else "missing key",
        "deepseek": "configured" if DEEPSEEK_API_KEY else "missing key",
        "humain": "configured" if HUMAIN_NODE_API_KEY else "missing key",
    }
    return {"status": "ok", "checks": checks}
