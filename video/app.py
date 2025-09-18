import uvicorn
import redis.asyncio as redis
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime
import os
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import torch
import logging
import numpy as np

# Alternative video export implementation
try:
    from diffusers.utils import export_to_video
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False
        print("Warning: Neither OpenCV nor imageio are available for video export")

LOG_PATH = './logs'
LOGFILE_CONTAINER = f'{LOG_PATH}/logfile_container_video.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.INFO, 
                   format='[%(asctime)s - %(name)s - %(levelname)s - %(message)s]')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] started logging in {LOGFILE_CONTAINER}')

current_pipeline = None

def load_pipeline(model_id, device, torch_dtype, variant):
    try:
        global current_pipeline
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_pipeline] trying to load pipeline: {model_id}')
        
        if current_pipeline is None:
            current_pipeline = StableVideoDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                variant=variant
            ).to(device)
            
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_pipeline] [success] Pipeline loaded!')
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_pipeline] [error] Failed to load pipeline: {e}')
        raise

def export_frames_to_video(frames, output_path, fps=10):
    """Export frames to video using available libraries"""
    if HAS_OPENCV:
        # Use the original OpenCV implementation
        from diffusers.utils import export_to_video
        export_to_video(frames, output_path, fps=fps)
    elif HAS_IMAGEIO:
        # Use imageio as fallback
        print(f"Using imageio to export video to {output_path}")
        # Convert frames to uint8 if needed
        if isinstance(frames[0], torch.Tensor):
            frames = [frame.cpu().numpy() for frame in frames]
        if frames[0].dtype != np.uint8:
            frames = [(frame * 255).astype(np.uint8) for frame in frames]
        
        # Write video using imageio
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
    else:
        raise ImportError("Neither OpenCV nor imageio are available. Please install one of them:\n"
                         "pip install opencv-python\nor\npip install imageio")

def generate_video(model_id, input_image_path, device, torch_dtype, variant, 
                  decode_chunk_size=8, motion_bucket_id=180, noise_aug_strength=0.1,
                  output_path="olol.mp4", fps=10):  # Changed path
    try:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] trying to load pipeline: {model_id}')
        load_pipeline(model_id, device, torch_dtype, variant)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] Pipeline loaded!')
        
        # Load input image
        input_image = Image.open(input_image_path)
        
        start_time = time.time()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] generating video from image: {input_image_path}')
        
        # Generate video frames
        frames = current_pipeline(
            input_image,
            decode_chunk_size=decode_chunk_size,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength
        ).frames[0]
        
        # Export to video file
        export_frames_to_video(frames, output_path, fps=fps)
        
        processing_time = time.time() - start_time
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] finished generating video! Saved to {output_path} in {processing_time:.2f}s')
        output_path = f'/usr/src/app/video/{output_path}'
        return {
            "output_path": output_path,
            "processing_time": f"{processing_time:.2f}s",
            "status": "success",
            "frames_generated": len(frames)
        }
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] [error]: {e}')
        return {
            "error": str(e),
            "status": "failed"
        }

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
    return 'Hello from video generation server!'

@app.get("/videos/{video_name}")
async def get_video(video_name: str):
    """
    Serve generated MP4 videos.
    Example: /videos/output_svd.mp4
    """
    # video_path = f"/usr/src/app/videos/{video_name}"  # Updated path
    video_path = f"{video_name}"  # Updated path
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    if not video_name.lower().endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only MP4 files are supported")
    return FileResponse(video_path, media_type="video/mp4")

@app.post("/generate")
async def generate(request: Request):
    try:
        req_data = await request.json()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate] req_data > {req_data}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate] req_data > {req_data}')
        
        if req_data["method"] == "status":
            return JSONResponse({"result_status": 200, "result_data": "ok"})
            
        if req_data["method"] == "generate_video":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] trying to generate video...')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_video] trying to generate video...')
            
            result = generate_video(
                req_data["model_id"],
                req_data["input_image_path"],
                req_data["device"],
                eval(req_data["torch_dtype"]),
                req_data.get("variant", "fp16"),
                req_data.get("decode_chunk_size", 8),
                req_data.get("motion_bucket_id", 180),
                req_data.get("noise_aug_strength", 0.1),
                req_data.get("output_path", "output_svd.mp4"),
                req_data.get("fps", 10)
            )
            
            return JSONResponse({"result_status": 200, "result_data": result})
            
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host=f'{os.getenv("VIDEO_IP")}', port=int(os.getenv("VIDEO_PORT")))