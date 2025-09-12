import google.generativeai as genai
from ..config import settings
from datetime import datetime
import os

# Configure Gemini API
genai.configure(api_key=settings.google_api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def log_debug(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"
    print(log_message.strip())
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(log_message)

def generate_emergency_sentence(words):
    prompt = f"다음 단어들을 사용하여 긴급 상황에서 구급대원에게 전달할 수 있는 간단한 한국어 문장 하나를 생성하세요: {', '.join(words)}. 문장은 간결하고 명확해야 하며 '~해주세요' 형태로 작성하세요."
    try:
        response = gemini_model.generate_content(prompt)
        sentence = response.text.strip()
        log_debug(f"Gemini API 문장 생성 성공 (generate_emergency_sentence): {sentence}")
        return sentence
    except Exception as e:
        log_debug(f"Gemini API 문장 생성 실패 (generate_emergency_sentence): {str(e)}")
        return " ".join(words) + " 도와주세요."