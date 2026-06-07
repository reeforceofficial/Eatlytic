from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
import re
import requests as http_requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# API configuration
SITE_NAME      = os.getenv("SITE_NAME", "Eatlytic")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

# Language mapping
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean"
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_base64_and_mime(data_url: str):
    """Parse  data:image/jpeg;base64,<data>  →  (mime_type, image_bytes)"""
    match = re.match(r"data:(image/[^;]+);base64,(.+)", data_url, re.DOTALL)
    if not match:
        raise ValueError("Invalid data URL format")
    mime_type   = match.group(1)
    image_bytes = base64.b64decode(match.group(2).strip())
    return mime_type, image_bytes


def build_prompt(language_name: str) -> str:
    return f"""You are Eatlytic AI, a nutrition expert.

Analyze the food image and provide:

## Estimated Calories
Provide total estimated calories for the entire dish/meal shown.

## Macronutrients
- **Protein** (g)
- **Fat** (g)
- **Carbohydrates** (g)

## Fiber
Dietary fiber content and its health benefits.

## Heart Health Impact
List positive and/or negative factors for cardiovascular health.

## Muscle Impact
Protein quality, amino acid profile, and muscle-building potential.

## Energy Impact
Whether this provides quick (simple carbs) or sustained (complex carbs + protein + fat) energy.

## Smart Eating Tips
Provide 2-3 actionable tips to make this meal healthier or consume it more effectively.

Use markdown formatting with bold headers and bullet points.
Respond entirely in {language_name} (fall back to English only if translation is unavailable)."""


# ── Provider 1: Google Gemini ────────────────────────────────────────────────

def analyze_with_gemini(image_bytes: bytes, mime_type: str, prompt: str) -> str:
    """Call Gemini 2.5 Flash Vision. Returns analysis text or raises Exception."""
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY not configured")

    print("🔗 [Gemini] Sending request to Gemini 2.5 Flash...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    text_part  = types.Part.from_text(text=prompt)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[image_part, text_part])],
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=1200,
        ),
    )

    analysis = response.text
    if not analysis or not analysis.strip():
        raise Exception("Gemini returned an empty response")

    print("✅ [Gemini] Analysis completed successfully")
    return analysis


# ── Provider 2: Groq (llama-4-scout — free vision model) ─────────────────────

def analyze_with_groq(image_data_url: str, prompt: str) -> str:
    """Call Groq llama-4-scout vision. Returns analysis text or raises Exception."""
    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY not configured")

    print("🔗 [Groq] Sending request to llama-4-scout-17b...")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url}
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1200
    }

    response = http_requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code == 401:
        raise Exception("GROQ_API_KEY is invalid or expired")
    if response.status_code == 429:
        raise Exception("Groq rate limit exceeded — try again later")
    if not response.ok:
        raise Exception(f"Groq API error {response.status_code}: {response.text[:300]}")

    result   = response.json()
    analysis = result["choices"][0]["message"]["content"]

    if not analysis or not analysis.strip():
        raise Exception("Groq returned an empty response")

    print("✅ [Groq] Analysis completed successfully")
    return analysis


# ── Main route ────────────────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def analyze_food():

    # 1. At least one API key must exist
    if not GEMINI_API_KEY and not GROQ_API_KEY:
        print("❌ ERROR: No API keys configured")
        return jsonify({"success": False, "error": "No API keys configured on server"}), 400

    # 2. Parse request JSON
    try:
        data = request.get_json(force=True)
    except Exception:
        data = None

    if not data:
        print("❌ ERROR: No JSON data provided")
        return jsonify({"success": False, "error": "No data provided"}), 400

    print("📥 Request received")

    # 3. Extract & validate image
    image_data_url = data.get("imageDataUrl", "").strip()
    if not image_data_url:
        print("❌ ERROR: No image data URL provided")
        return jsonify({"success": False, "error": "Image not provided"}), 400

    if not image_data_url.startswith("data:image/"):
        print(f"❌ ERROR: Invalid image format")
        return jsonify({"success": False, "error": "Invalid image format. Must be a base64 data URL (data:image/...)"}), 400

    try:
        mime_type, image_bytes = extract_base64_and_mime(image_data_url)
    except Exception as e:
        print(f"❌ Image decode error: {e}")
        return jsonify({"success": False, "error": "Could not decode image data"}), 400

    print(f"✅ Image validated — MIME: {mime_type}, size: {len(image_bytes):,} bytes")

    # 4. Resolve language
    language_code = data.get("language", "en").strip()
    if language_code not in LANGUAGE_NAMES:
        language_code = "en"
    language_name = LANGUAGE_NAMES[language_code]
    print(f"🌐 Language: {language_name}")

    # 5. Build prompt
    prompt = build_prompt(language_name)

    # 6. Try Gemini first → fallback to Groq automatically
    gemini_error = None
    groq_error   = None

    # ── Try Gemini ──────────────────────────────────────────────────────────
    if GEMINI_API_KEY:
        try:
            analysis = analyze_with_gemini(image_bytes, mime_type, prompt)
            return jsonify({"success": True, "analysis": analysis, "provider": "gemini"}), 200
        except Exception as e:
            gemini_error = str(e)
            print(f"⚠️  Gemini failed ({gemini_error}) — falling back to Groq...")
    else:
        gemini_error = "key not configured"
        print("⚠️  Gemini key not set — skipping to Groq")

    # ── Fallback: Groq ──────────────────────────────────────────────────────
    if GROQ_API_KEY:
        try:
            analysis = analyze_with_groq(image_data_url, prompt)
            return jsonify({"success": True, "analysis": analysis, "provider": "groq"}), 200
        except Exception as e:
            groq_error = str(e)
            print(f"❌ Groq also failed: {groq_error}")
    else:
        groq_error = "key not configured"
        print("⚠️  Groq key not set — no more fallbacks available")

    # ── Both failed ─────────────────────────────────────────────────────────
    print("❌ All providers failed")
    return jsonify({
        "success": False,
        "error": "All AI providers failed. Please try again later.",
        "details": {
            "gemini": gemini_error,
            "groq":   groq_error
        }
    }), 502


# ── Other routes ──────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": f"{SITE_NAME} AI backend",
        "providers": {
            "gemini": "✅ configured" if GEMINI_API_KEY else "❌ not configured",
            "groq":   "✅ configured" if GROQ_API_KEY   else "❌ not configured",
        }
    })


@app.route("/")
def home():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 48)
    print(f"   {SITE_NAME} — AI Food Analyzer")
    print("=" * 48)
    print(f"   Gemini : {'✅ configured' if GEMINI_API_KEY else '❌ missing (add GEMINI_API_KEY to .env)'}")
    print(f"   Groq   : {'✅ configured' if GROQ_API_KEY   else '❌ missing (add GROQ_API_KEY to .env)'}")
    print("=" * 48)
    print("   🚀  Running on  http://0.0.0.0:5000")
    print("=" * 48)
    app.run(host="0.0.0.0", port=5000, debug=True)