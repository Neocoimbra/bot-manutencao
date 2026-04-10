#!/usr/bin/env python3
"""
Bot de Manutenção Automotiva e Agrícola - Multi-API com Fallback + Áudio
Cascata: Gemini x2 -> DeepSeek -> Claude -> OpenAI -> Fallback
Transcrição de áudio: Gemini -> DeepSeek -> OpenAI Whisper -> Fallback
"""

import os
import sys
import json
import logging
import time
import base64
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import urllib.request
import urllib.parse
import urllib.error

# --- Configuração via Variáveis de Ambiente ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

GEMINI_KEYS = []
gk1 = os.environ.get("GEMINI_KEY_1", "")
gk2 = os.environ.get("GEMINI_KEY_2", "")
if gk1: GEMINI_KEYS.append(gk1)
if gk2: GEMINI_KEYS.append(gk2)
GEMINI_MODEL = "gemini-2.0-flash"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano")

CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_MODEL = "claude-3-haiku-20240307"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

KNOWLEDGE_DIR = Path("/tmp/knowledge_base")
KNOWLEDGE_DIR.mkdir(exist_ok=True)

ADMIN_ID = int(os.environ.get("ADMIN_ID", "912095382"))

FALLBACK_MESSAGE = "Dificuldade em processar todas as solicitações, procure o distribuidor!"

# Cache de APIs com falha - pula APIs que falharam nos últimos 5 minutos
_api_fail_cache = {}
FAIL_CACHE_TTL = 300

def mark_api_failed(name):
    _api_fail_cache[name] = time.time()

def is_api_available(name):
    if name not in _api_fail_cache:
        return True
    if time.time() - _api_fail_cache[name] > FAIL_CACHE_TTL:
        del _api_fail_cache[name]
        return True
    return False

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --- System Prompt ---
SYSTEM_PROMPT_BASE = (
    "Você é um mecânico e técnico de manutenção altamente experiente, "
    "especializado em equipamentos automotivos (carros, caminhões, motos) "
    "e de agricultura (tratores, colhedoras, pulverizadores, plantadeiras, etc). "
    "Responda SEMPRE em português brasileiro.\n\n"
    "REGRAS PARA SUAS RESPOSTAS:\n"
    "1. Seja PRÁTICO e DIRETO. Dê passos numerados de diagnóstico e reparo.\n"
    "2. Comece identificando as causas mais prováveis do problema.\n"
    "3. Para cada causa, explique como verificar e como resolver.\n"
    "4. Inclua ferramentas necessárias quando relevante.\n"
    "5. Indique peças que podem precisar ser trocadas com nomes técnicos.\n"
    "6. Alerte sobre riscos de segurança quando houver.\n"
    "7. Se houver informações dos manuais no banco de conhecimento, "
    "PRIORIZE essas informações e cite o manual de referência.\n"
    "8. Use linguagem técnica mas acessível.\n"
    "9. Quando o problema puder ser grave, recomende levar a um profissional."
)

# --- Telegram API ---
def telegram_request(method, data=None, timeout=60):
    url = f"{TELEGRAM_API}/{method}"
    try:
        if data:
            data_encoded = urllib.parse.urlencode(data).encode("utf-8")
        else:
            data_encoded = None
        req = urllib.request.Request(url, data=data_encoded)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        logger.error(f"HTTP Error {e.code} em {method}: {body[:200]}")
        return None
    except Exception as e:
        logger.error(f"Erro em {method}: {e}")
        return None

def send_message(chat_id, text):
    if len(text) > 4000:
        for i in range(0, len(text), 4000):
            telegram_request("sendMessage", {"chat_id": chat_id, "text": text[i:i+4000]})
    else:
        result = telegram_request("sendMessage", {"chat_id": chat_id, "text": text})
        if result and result.get("ok"):
            logger.info(f"Mensagem enviada para {chat_id}")
        else:
            logger.error(f"Falha ao enviar para {chat_id}: {result}")

def send_typing(chat_id):
    telegram_request("sendChatAction", {"chat_id": chat_id, "action": "typing"})

def download_file(file_id):
    result = telegram_request("getFile", {"file_id": file_id})
    if result and result.get("ok"):
        file_path = result["result"]["file_path"]
        url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
        local_path = f"/tmp/{os.path.basename(file_path)}"
        urllib.request.urlretrieve(url, local_path)
        return local_path
    return None

# --- APIs de IA ---
def call_gemini(prompt, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.7},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text = parts[0].get("text", "")
                    if text:
                        return text
            return None
    except urllib.error.HTTPError as e:
        logger.warning(f"Gemini ...{api_key[-6:]} falhou (HTTP {e.code})")
        return None
    except Exception as e:
        logger.warning(f"Gemini ...{api_key[-6:]} erro: {e}")
        return None

def call_openai_compatible(prompt, api_key, base_url, model, name="API"):
    if not api_key:
        return None
    url = f"{base_url}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.7,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data['choices'][0]['message']['content']
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        logger.warning(f"{name} falhou (HTTP {e.code}): {body[:150]}")
        return None
    except Exception as e:
        logger.warning(f"{name} erro: {e}")
        return None

def call_claude(prompt):
    if not CLAUDE_API_KEY:
        return None
    url = "https://api.anthropic.com/v1/messages"
    payload = json.dumps({
        "model": CLAUDE_MODEL,
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result.get("content", [])
            if content:
                return content[0].get("text", "")
            return None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        logger.warning(f"Claude falhou (HTTP {e.code}): {body[:150]}")
        return None
    except Exception as e:
        logger.warning(f"Claude erro: {e}")
        return None

def call_ai_with_fallback(prompt):
    """Cascata com cache: pula APIs que falharam nos últimos 5 min."""
    # 1. Gemini
    for i, key in enumerate(GEMINI_KEYS):
        name = f"gemini_{i+1}"
        if not is_api_available(name):
            continue
        logger.info(f"Tentando Gemini key {i+1}/{len(GEMINI_KEYS)}...")
        res = call_gemini(prompt, key)
        if res:
            logger.info(f"Gemini key {i+1} respondeu ({len(res)} chars)")
            return res
        mark_api_failed(name)

    # 2. DeepSeek
    if is_api_available("deepseek"):
        logger.info("Tentando DeepSeek...")
        res = call_openai_compatible(prompt, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, "DeepSeek")
        if res:
            logger.info(f"DeepSeek respondeu ({len(res)} chars)")
            return res
        mark_api_failed("deepseek")

    # 3. Claude
    if is_api_available("claude"):
        logger.info("Tentando Claude...")
        res = call_claude(prompt)
        if res:
            logger.info(f"Claude respondeu ({len(res)} chars)")
            return res
        mark_api_failed("claude")

    # 4. OpenAI (sempre tenta - backup principal)
    if OPENAI_API_KEY:
        logger.info("Tentando OpenAI...")
        res = call_openai_compatible(prompt, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, "OpenAI")
        if res:
            logger.info(f"OpenAI respondeu ({len(res)} chars)")
            return res

    logger.warning("Todas as APIs falharam!")
    return None

# --- Transcrição de Áudio ---
def transcribe_audio(audio_path):
    """Tenta transcrever áudio usando múltiplos métodos."""
    
    # Converter para WAV
    wav_path = audio_path + ".wav"
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1", "-y", wav_path],
            capture_output=True, timeout=30
        )
        if result.returncode != 0:
            logger.error(f"ffmpeg erro: {result.stderr.decode()[:200]}")
            return None
    except Exception as e:
        logger.error(f"ffmpeg exception: {e}")
        return None

    # Método 1: Gemini com áudio (multimodal)
    for i, key in enumerate(GEMINI_KEYS):
        name = f"gemini_stt_{i+1}"
        if not is_api_available(name):
            continue
        logger.info(f"Transcrição: tentando Gemini key {i+1}...")
        text = transcribe_with_gemini(wav_path, key)
        if text:
            logger.info(f"Gemini key {i+1} transcreveu ({len(text)} chars)")
            return text
        mark_api_failed(name)

    # Método 2: DeepSeek STT
    if is_api_available("deepseek_stt"):
        logger.info("Transcrição: tentando DeepSeek STT...")
        text = transcribe_with_openai_whisper(wav_path, DEEPSEEK_API_KEY, "https://api.deepseek.com/v1")
        if text:
            logger.info(f"DeepSeek STT transcreveu ({len(text)} chars)")
            return text
        mark_api_failed("deepseek_stt")

    # Método 3: OpenAI Whisper (sempre tenta - backup principal)
    if OPENAI_API_KEY:
        logger.info("Transcrição: tentando OpenAI Whisper...")
        text = transcribe_with_openai_whisper(wav_path, OPENAI_API_KEY, OPENAI_BASE_URL)
        if text:
            logger.info(f"OpenAI Whisper transcreveu ({len(text)} chars)")
            return text

    logger.warning("Todas as transcrições falharam!")
    return None

def transcribe_with_gemini(wav_path, api_key):
    """Usa Gemini multimodal para transcrever áudio."""
    try:
        with open(wav_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
        payload = json.dumps({
            "contents": [{
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": audio_data
                        }
                    },
                    {
                        "text": "Transcreva este áudio em português brasileiro. Retorne APENAS o texto transcrito, sem explicações."
                    }
                ]
            }],
            "generationConfig": {"maxOutputTokens": 500, "temperature": 0.1},
        }).encode("utf-8")
        
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text = parts[0].get("text", "").strip()
                    if text:
                        return text
        return None
    except urllib.error.HTTPError as e:
        logger.warning(f"Gemini STT ...{api_key[-6:]} falhou (HTTP {e.code})")
        return None
    except Exception as e:
        logger.warning(f"Gemini STT erro: {e}")
        return None

def transcribe_with_openai_whisper(wav_path, api_key, base_url):
    """Usa OpenAI Whisper API para transcrever."""
    if not api_key:
        return None
    import http.client

    host = base_url.replace("https://", "").replace("http://", "")
    if "/" in host:
        path_prefix = "/" + "/".join(host.split("/")[1:])
        host = host.split("/")[0]
    else:
        path_prefix = ""

    boundary = "----FormBoundary7MA4YWxk"
    with open(wav_path, "rb") as f:
        file_data = f.read()

    body = b""
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
    body += b"Content-Type: audio/wav\r\n\r\n"
    body += file_data
    body += b"\r\n"
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="model"\r\n\r\n'
    body += b"whisper-1\r\n"
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="language"\r\n\r\n'
    body += b"pt\r\n"
    body += f"--{boundary}--\r\n".encode()

    try:
        conn = http.client.HTTPSConnection(host, timeout=30)
        conn.request("POST", f"{path_prefix}/audio/transcriptions", body=body, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        })
        resp = conn.getresponse()
        data = json.loads(resp.read().decode("utf-8"))
        conn.close()
        if resp.status == 200:
            return data.get("text", "")
        logger.warning(f"Whisper falhou (HTTP {resp.status})")
        return None
    except Exception as e:
        logger.warning(f"Whisper erro: {e}")
        return None

# --- Banco de Conhecimento ---
def load_knowledge_base():
    texts = []
    for f in KNOWLEDGE_DIR.iterdir():
        if f.suffix == ".txt":
            try:
                texts.append(f.read_text(encoding="utf-8", errors="ignore"))
            except Exception as e:
                logger.error(f"Erro ao ler {f}: {e}")
    return "\n\n---\n\n".join(texts) if texts else ""

def build_full_prompt(user_text):
    knowledge = load_knowledge_base()
    prompt = SYSTEM_PROMPT_BASE
    if knowledge:
        prompt += (
            "\n\n=== INFORMAÇÕES DOS MANUAIS (PRIORIDADE MÁXIMA) ===\n"
            "Use OBRIGATORIAMENTE estas informações dos manuais como base principal "
            "para suas respostas. Cite o manual quando usar estas informações:\n\n"
            + knowledge[:4000]
        )
    prompt += f"\n\nPergunta do usuário: {user_text}"
    return prompt

# --- Handlers ---
def handle_start(chat_id):
    logger.info(f"Comando /start de {chat_id}")
    send_message(chat_id,
        "🔧 Especialista em Manutenção Online!\n\n"
        "Sou um assistente de IA especializado em manutenção e reparação de "
        "equipamentos automotivos e de agricultura.\n\n"
        "📋 O que posso fazer:\n"
        "• Responder dúvidas sobre manutenção preventiva e corretiva\n"
        "• Ajudar a diagnosticar problemas em equipamentos\n"
        "• Orientar sobre procedimentos de reparo\n\n"
        "📄 Aceito documentos (PDF, Word, TXT) com manuais técnicos\n"
        "🎤 Aceito mensagens de voz e áudio!\n\n"
        "Como posso ajudar no diagnóstico hoje?"
    )

def handle_text(chat_id, text):
    logger.info(f"Mensagem de {chat_id}: {text[:100]}")
    send_typing(chat_id)
    full_prompt = build_full_prompt(text)
    reply = call_ai_with_fallback(full_prompt)
    send_message(chat_id, reply if reply else FALLBACK_MESSAGE)

def handle_voice(chat_id, voice_or_audio):
    """Processa mensagem de voz ou áudio."""
    file_id = voice_or_audio.get("file_id")
    duration = voice_or_audio.get("duration", 0)
    logger.info(f"Áudio de {chat_id}: {duration}s")
    send_typing(chat_id)
    send_message(chat_id, "🎤 Processando seu áudio...")

    try:
        local_path = download_file(file_id)
        if not local_path:
            send_message(chat_id, "⚠️ Não consegui baixar o áudio.")
            return

        transcribed = transcribe_audio(local_path)
        if not transcribed:
            send_message(chat_id, "⚠️ Não consegui transcrever o áudio. Tente enviar como texto.")
            return

        logger.info(f"Transcrição de {chat_id}: {transcribed[:100]}")
        send_message(chat_id, f"🎤 Entendi: \"{transcribed}\"\n\nProcessando resposta...")
        send_typing(chat_id)

        full_prompt = build_full_prompt(transcribed)
        reply = call_ai_with_fallback(full_prompt)
        send_message(chat_id, reply if reply else FALLBACK_MESSAGE)

    except Exception as e:
        logger.error(f"Erro áudio: {e}", exc_info=True)
        send_message(chat_id, "Desculpe, erro ao processar o áudio.")

def handle_document(chat_id, document):
    if chat_id != ADMIN_ID:
        send_message(chat_id, "⚠️ Apenas o administrador pode enviar documentos para o banco de conhecimento.")
        return

    file_name = document.get("file_name", "doc")
    send_message(chat_id, f"📥 Processando manual: {file_name}")

    try:
        local_path = download_file(document.get("file_id"))
        if not local_path:
            send_message(chat_id, "❌ Não consegui baixar o arquivo.")
            return

        text = ""
        if file_name.lower().endswith(".pdf"):
            from PyPDF2 import PdfReader
            reader = PdfReader(local_path)
            text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif file_name.lower().endswith(".docx"):
            from docx import Document
            doc = Document(local_path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        elif file_name.lower().endswith(".txt"):
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            send_message(chat_id, "⚠️ Formato não suportado. Envie PDF, Word (.docx) ou TXT.")
            return

        if text.strip():
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in file_name)
            save_path = KNOWLEDGE_DIR / f"{safe_name}.txt"
            save_path.write_text(text, encoding="utf-8")
            send_message(chat_id,
                f"✅ Manual '{file_name}' indexado com sucesso!\n"
                f"📊 {len(text)} caracteres adicionados ao banco de conhecimento."
            )
        else:
            send_message(chat_id, "❌ Não foi possível extrair texto do arquivo.")
    except Exception as e:
        logger.error(f"Erro doc: {e}", exc_info=True)
        send_message(chat_id, "❌ Erro técnico ao processar documento.")

# --- Main Loop ---
def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN não configurado!")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("BOT MANUTENÇÃO - MULTI-API + ÁUDIO")
    logger.info(f"Cascata IA: Gemini x{len(GEMINI_KEYS)} -> DeepSeek -> Claude -> OpenAI")
    logger.info(f"Admin ID: {ADMIN_ID}")
    logger.info("=" * 50)

    me = telegram_request("getMe")
    if not me or not me.get("ok"):
        logger.error(f"Token inválido: {me}")
        sys.exit(1)
    logger.info(f"Bot: @{me['result']['username']}")

    telegram_request("deleteWebhook", {"drop_pending_updates": "false"})
    logger.info("Webhook removido. Iniciando polling...")

    offset = 0
    consecutive_errors = 0

    while True:
        try:
            result = telegram_request("getUpdates", {
                "offset": offset,
                "timeout": 30,
                "allowed_updates": json.dumps(["message"]),
            }, timeout=60)

            if result and result.get("ok"):
                consecutive_errors = 0
                updates = result.get("result", [])
                if updates:
                    logger.info(f"Recebidos {len(updates)} updates")
                for update in updates:
                    update_id = update["update_id"]
                    offset = update_id + 1
                    try:
                        msg = update.get("message", {})
                        chat_id = msg.get("chat", {}).get("id")
                        if not chat_id:
                            continue
                        text = msg.get("text", "")

                        if text.startswith("/start"):
                            handle_start(chat_id)
                        elif msg.get("voice"):
                            handle_voice(chat_id, msg["voice"])
                        elif msg.get("audio"):
                            handle_voice(chat_id, msg["audio"])
                        elif msg.get("document"):
                            handle_document(chat_id, msg["document"])
                        elif text:
                            handle_text(chat_id, text)
                    except Exception as e:
                        logger.error(f"Erro update {update_id}: {e}", exc_info=True)
            else:
                consecutive_errors += 1
                logger.warning(f"Polling falhou ({consecutive_errors}x)")
                if consecutive_errors > 10:
                    time.sleep(30)
                    consecutive_errors = 0
                else:
                    time.sleep(2)
        except KeyboardInterrupt:
            logger.info("Bot encerrado.")
            break
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Erro no loop: {e}", exc_info=True)
            time.sleep(5)

# --- Health Check HTTP Server para Render ---
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Bot de Manutencao - Online')
    def log_message(self, format, *args):
        pass  # Silencia logs do HTTP server

def start_health_server():
    port = int(os.environ.get('PORT', 10000))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    logger.info(f'Health server na porta {port}')
    server.serve_forever()

if __name__ == "__main__":
    # Inicia o health server em thread separada
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    main()

