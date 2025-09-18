import uvicorn
import redis.asyncio as redis
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime
import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq
from transformers import pipeline
import soundfile as sf

# Setup logging
LOG_PATH = './logs'
LOGFILE_CONTAINER = f'{LOG_PATH}/logfile_container_tts.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(
    filename=LOGFILE_CONTAINER,
    level=logging.INFO,
    format='[%(asctime)s - %(name)s - %(levelname)s - %(message)s]'
)
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] started logging in {LOGFILE_CONTAINER}')

# Global model variables
current_tts_model = None
current_tts_tokenizer = None
current_tts_vocoder = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_tts_model(model_name):
    try:
        global current_tts_model, current_tts_tokenizer, current_tts_vocoder
        
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_tts_model] loading TTS model: {model_name}')
        
        if current_tts_model is None:
            # Load tokenizer and model
            current_tts_tokenizer = AutoTokenizer.from_pretrained(model_name)
            current_tts_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
            
            # Create TTS pipeline
            current_tts_pipeline = pipeline(
                "text-to-speech",
                model=current_tts_model,
                tokenizer=current_tts_tokenizer,
                device=device
            )
            
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_tts_model] TTS model loaded successfully!')
            
        return current_tts_pipeline
    except Exception as e:
        logging.error(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_tts_model] Failed to load TTS model: {e}')
        raise

def generate_speech(text, model_name, output_file="output.wav"):
    try:
        start_time = time.time()
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_speech] generating speech for text: {text[:50]}...')
        
        # Load model if not loaded
        tts_pipeline = load_tts_model(model_name)
        
        # Generate speech
        speech_output = tts_pipeline(text)
        
        # Save to file
        sf.write(output_file, speech_output["audio"], speech_output["sampling_rate"])
        
        processing_time = time.time() - start_time
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_speech] finished generating speech in {processing_time:.2f}s')
        
        return output_file, processing_time
    except Exception as e:
        logging.error(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_speech] Error: {e}')
        return None, 0

# Redis connection (optional)
redis_connection = None

def start_redis(req_redis_port):
    try:
        r = redis.Redis(host="redis", port=req_redis_port, db=0)
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [start_redis] Redis started successfully.')
        return r
    except Exception as e:
        logging.error(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [start_redis] Failed to start Redis: {e}')
        raise

app = FastAPI()

@app.get("/")
async def root():
    return {'message': 'Hello from TTS server!'}

@app.post("/generate")
async def generate_tts(request: Request):
    try:
        req_data = await request.json()
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_tts] request data: {req_data}')
        
        if req_data["method"] == "status":
            return JSONResponse({"status": 200, "message": "Server is running"})
        
        if req_data["method"] == "generate_speech":
            text = req_data["text"]
            model_name = req_data.get("model", "facebook/fastspeech2-en-ljspeech")
            output_file = req_data.get("output_file", "output.wav")
            
            audio_file, processing_time = generate_speech(text, model_name, output_file)
            
            if audio_file:
                return FileResponse(
                    audio_file,
                    media_type="audio/wav",
                    headers={
                        "processing_time": f"{processing_time:.2f}s",
                        "model_used": model_name
                    }
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to generate speech")
    except Exception as e:
        logging.error(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_tts] Error: {e}')
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=f'{os.getenv("TTS_IP")}', port=int(os.getenv("TTS_PORT")))
    