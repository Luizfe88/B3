import os
import requests
from dotenv import load_dotenv

def test_gemini_connectivity():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("[!] ERRO: GEMINI_API_KEY nao encontrada no .env")
        return False

    print(f"Propulsando teste com a chave: {api_key[:10]}...")
    
    # Usando gemini-2.0-flash que apareceu no ListModels
    model = "gemini-2.0-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": "Hello"}]
        }]
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        print("[*] Conectado com sucesso ao modelo 2.0-flash!")
        if "candidates" in data:
             print(f"Resposta: {data['candidates'][0]['content']['parts'][0]['text'].strip()}")
        return True
    except Exception as e:
        print(f"[!] Falha: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Body: {e.response.text}")
        return False

if __name__ == "__main__":
    test_gemini_connectivity()
