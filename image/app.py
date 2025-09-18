import uvicorn
import redis.asyncio as redis
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime
import os
from diffusers import StableDiffusionPipeline
import torch
import logging
import re
from datetime import datetime


LOG_PATH = './logs'
LOGFILE_CONTAINER = f'{LOG_PATH}/logfile_container_image.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.INFO, 
                   format='[%(asctime)s - %(name)s - %(levelname)s - %(message)s]')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] started logging in {LOGFILE_CONTAINER}')

current_model = None

def load_model(model_id, device, torch_dtype):
    try:
        global current_model
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_model] trying to load model: {model_id}')
        
        if current_model is None:
            current_model = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True
            ).to(device)
            
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_model] [success] Model loaded!')
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_model] [error] Failed to load model: {e}')
        raise

def generate_image(model_id, prompt, device, torch_dtype, output_path):
    try:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] trying to load model: {model_id}')
        load_model(model_id, device, torch_dtype)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] Model loaded!')
        
        start_time = time.time()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] generating image for prompt: {prompt}')
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] generating image for output_path: {output_path}')
        
        image = current_model(prompt).images[0]
        image.save(output_path)
        
        processing_time = time.time() - start_time
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] finished generating image! Saved to {output_path} in {processing_time:.2f}s')
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] changing output_path {output_path}')
        output_path = f'/usr/src/app/image/{output_path}'
        # output_path = f'/image/{output_path}'
        # output_path = f'{output_path}'
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] output_path changed! ->  {output_path}')
        return {
            "output_path": output_path,
            "processing_time": f"{processing_time:.2f}s",
            "status": "success"
        }
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] [error]: {e}')
        return {
            "error": str(e),
            "status": "failed"
        }



def prompt_to_filename(prompt, extension="png", add_timestamp=False):
    filename = prompt.lower()
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '_', filename)
    filename = filename.strip('_')

    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"
    
    return f"{filename}.{extension}"


redis_connection = None

def start_redis(req_redis_port):
    try:
        r = redis.Redis(host="redis", port=req_redis_port, db=0)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [start_redis] Redis started successfully.')
        return r
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [start_redis] Failed to start Redis on port {req_redis_port}: {e}')
        raise

app = FastAPI()

@app.get("/")
async def root():
    return 'Hello from image generation server!'

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = f"./{image_name}"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    if not image_name.lower().endswith('.png'):
        raise HTTPException(status_code=400, detail="Only PNG files are supported")
    return FileResponse(image_path, media_type="image/png")

@app.post("/generate")
async def generate(request: Request):
    try:
        req_data = await request.json()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate] req_data > {req_data}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate] req_data > {req_data}')
        
        if req_data["method"] == "status":
            return JSONResponse({"result_status": 200, "result_data": "ok"})
            
        if req_data["method"] == "generate_image":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] trying to generate image...')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] trying to generate image...')


            req_image_path = prompt_to_filename(req_data["image_prompt"])

            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] req_image_path: {req_image_path}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] req_image_path: {req_image_path}')

            
            result = generate_image(
                req_data["image_model"],
                req_data["image_prompt"],
                req_data["image_device"],
                eval(req_data["image_compute_type"]),
                req_image_path
            )
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] result {result}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] result {result}')
            
            return JSONResponse({"result_status": 200, "result_data": result})
            
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host=f'{os.getenv("IMAGE_IP")}', port=int(os.getenv("IMAGE_PORT")))