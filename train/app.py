import uvicorn
import redis.asyncio as redis
import time
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime
import os
import logging
import json
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import torch
import asyncio

import torch
import json
import yaml
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset


class LoRATrainer:
    def __init__(self, config_path: str = "./config.yaml"):
        self.config = self.load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with quantization"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Setting up model and tokenizer")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["training"]["model_name"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["training"]["model_name"],
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            lora_alpha=self.config["lora"]["lora_alpha"],
            lora_dropout=self.config["lora"]["lora_dropout"],
            r=self.config["lora"]["r"],
            target_modules=self.config["lora"]["target_modules"],
            task_type=self.config["lora"]["task_type"],
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
    
    def prepare_dataset(self, training_data: List[Dict[str, str]]) -> Dataset:
        """Prepare dataset for training"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Preparing dataset")
        
        # Format training examples
        formatted_data = []
        for example in training_data:
            if "instruction" in example and "input" in example and "output" in example:
                text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
                formatted_data.append({"text": text})
        
        return Dataset.from_list(formatted_data)
    
    def setup_trainer(self, train_dataset: Dataset):
        """Set up the SFT trainer"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Setting up trainer")
        
        training_args = TrainingArguments(
            output_dir=self.config["training"]["output_dir"],
            num_train_epochs=self.config["training"]["num_train_epochs"],
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=self.config["training"]["learning_rate"],
            fp16=self.config["training"]["fp16"],
            save_steps=self.config["training"]["save_steps"],
            logging_steps=self.config["training"]["logging_steps"],
            max_steps=-1,
            optim="paged_adamw_8bit",
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            report_to="none"
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            peft_config=None,  # Already applied
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            args=training_args,
            max_seq_length=self.config["training"]["max_seq_length"],
            packing=False
        )
    
    def train(self, training_data: List[Dict[str, str]]):
        """Execute the training process"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training")
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Prepare dataset
        dataset = self.prepare_dataset(training_data)
        
        # Setup trainer
        self.setup_trainer(dataset)
        
        # Start training
        self.trainer.train()
        
        # Save the final model
        output_dir = f"{self.config['training']['output_dir']}/final_model"
        self.trainer.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training completed. Model saved to {output_dir}")
        
        return output_dir
    
    def generate_plots(self, prompt: str, num_samples: int = 5, temperature: float = 0.8) -> List[str]:
        """Generate multiple plot variations"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not initialized")
        
        generated_plots = []
        
        for _ in range(num_samples):
            inputs = self.tokenizer(
                f"### Instruction:\nWrite a detailed plot summary based on the following description:\n\n### Input:\n{prompt}\n\n### Response:\n",
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids.cuda(),
                    max_new_tokens=self.config["evaluation"]["max_new_tokens"],
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the response part
            response_start = generated_text.find("### Response:") + len("### Response:")
            plot = generated_text[response_start:].strip()
            generated_plots.append(plot)
        
        return generated_plots

# Singleton instance
trainer_instance = None

def get_trainer(config_path: str = "./config.yaml"):
    """Get or create the trainer instance"""
    global trainer_instance
    if trainer_instance is None:
        trainer_instance = LoRATrainer(config_path)
    return trainer_instance


#############################


















# Logging setup
LOG_PATH = './logs'
LOGFILE_CONTAINER = f'{LOG_PATH}/logfile_container_train.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.INFO, 
                   format='[%(asctime)s - %(name)s - %(levelname)s - %(message)s]')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] started logging in {LOGFILE_CONTAINER}')

# Global variables
similarity_model = None
gold_standards = []
gold_embeddings = None
redis_connection = None
training_in_progress = False
training_results = {}
config = {}

def load_config():
    """Load configuration from YAML file"""
    global config
    try:
        with open('./config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [CONFIG] Configuration loaded')
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [ERROR] Failed to load config: {e}')
        config = {}

def load_gold_standards():
    """Load gold standard plots from JSON file"""
    try:
        gold_standards_path = config.get('dataset', {}).get('gold_standards_path', './gold_standards.json')
        if os.path.exists(gold_standards_path):
            with open(gold_standards_path, 'r') as f:
                data = json.load(f)
                return data.get('gold_standards', [])
        return []
    except Exception as e:
        logging.error(f'Error loading gold standards: {e}')
        return []

def initialize_similarity_model():
    """Initialize the sentence transformer model for similarity"""
    global similarity_model, gold_embeddings, gold_standards
    
    try:
        model_name = config.get('evaluation', {}).get('similarity_model', 'all-MiniLM-L6-v2')
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] Loading similarity model: {model_name}')
        similarity_model = SentenceTransformer(model_name)
        
        # Load and embed gold standards
        gold_standards = load_gold_standards()
        if gold_standards:
            gold_texts = [gs['plot'] for gs in gold_standards]
            gold_embeddings = similarity_model.encode(gold_texts, convert_to_tensor=True)
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] Loaded {len(gold_standards)} gold standards')
        else:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [WARNING] No gold standards found')
            
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [ERROR] Failed to load similarity model: {e}')
        raise

def calculate_similarity_score(generated_plot: str) -> float:
    """Calculate semantic similarity score against gold standards"""
    global similarity_model, gold_embeddings
    
    if similarity_model is None or gold_embeddings is None:
        raise ValueError("Similarity model or gold standards not initialized")
    
    try:
        # Encode the generated plot
        gen_embedding = similarity_model.encode(generated_plot, convert_to_tensor=True)
        
        # Calculate cosine similarities
        similarities = util.pytorch_cos_sim(gen_embedding, gold_embeddings)[0]
        
        # Return average similarity
        return float(torch.mean(similarities).item())
    
    except Exception as e:
        logging.error(f'Error calculating similarity: {e}')
        return 0.0

def evaluate_plot_quality(generated_plot: str, source_rating: float = None) -> Dict[str, Any]:
    """Evaluate plot quality using similarity and rating signals"""
    
    similarity_score = calculate_similarity_score(generated_plot)
    min_rating = config.get('evaluation', {}).get('min_rating_threshold', 7.0)
    
    # Apply rating filter if provided
    rating_penalty = 1.0
    if source_rating is not None and source_rating < min_rating:
        rating_penalty = max(0.1, source_rating / min_rating)  # Scale penalty
    
    # Combined score (similarity weighted by rating quality)
    combined_score = similarity_score * rating_penalty
    
    return {
        'similarity_score': similarity_score,
        'source_rating': source_rating,
        'rating_penalty': rating_penalty,
        'combined_score': combined_score,
        'passed_rating_filter': source_rating is None or source_rating >= min_rating
    }

def start_redis(req_redis_port: int):
    """Initialize Redis connection"""
    try:
        r = redis.Redis(host="redis", port=req_redis_port, db=0)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] Redis started successfully.')
        return r
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] Failed to start Redis: {e}')
        raise

def load_training_data() -> List[Dict[str, str]]:
    """Load training data from JSONL file"""
    training_data_path = config.get('dataset', {}).get('train', './training_data.jsonl')
    training_data = []
    
    if os.path.exists(training_data_path):
        try:
            with open(training_data_path, 'r') as f:
                for line in f:
                    training_data.append(json.loads(line))
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Loaded {len(training_data)} training examples')
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [ERROR] Failed to load training data: {e}')
    
    return training_data

async def train_model_async():
    """Async function to train the model"""
    global training_in_progress, training_results
    
    try:
        training_in_progress = True
        training_results = {"status": "training", "progress": 0}
        
        # Load training data
        training_data = load_training_data()
        
        if not training_data:
            training_results = {"status": "error", "message": "No training data found"}
            return
        
        # Initialize and train
        trainer = get_trainer()
        output_dir = trainer.train(training_data)
        
        training_results = {
            "status": "completed", 
            "message": "Training finished successfully",
            "output_dir": output_dir
        }
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [ERROR] {error_msg}')
        training_results = {"status": "error", "message": error_msg}
    
    finally:
        training_in_progress = False

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global redis_connection
    
    try:
        # Load configuration
        load_config()
        
        # Initialize similarity model
        initialize_similarity_model()
        
        # Start Redis
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_connection = start_redis(redis_port)
        
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] Plot trainer service initialized')
        
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] Failed to initialize: {e}')
        raise

@app.get("/")
async def root():
    return {"message": "Hello from Plot Trainer service!"}

@app.post("/evaluate")
async def evaluate_plots(request: Request):
    """Evaluate multiple generated plots and return rankings"""
    try:
        req_data = await request.json()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [evaluate] Received evaluation request')
        
        generated_plots = req_data.get("generated_plots", [])
        source_rating = req_data.get("source_rating")
        min_rating = config.get('evaluation', {}).get('min_rating_threshold', 7.0)
        
        if not generated_plots:
            return JSONResponse({"error": "No generated plots provided"}, status_code=400)
        
        results = []
        for i, plot in enumerate(generated_plots):
            evaluation = evaluate_plot_quality(plot, source_rating)
            results.append({
                "plot_index": i,
                "plot_text": plot[:200] + "..." if len(plot) > 200 else plot,  # Truncate for response
                **evaluation
            })
        
        # Sort by combined score descending
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return JSONResponse({
            "result_status": 200,
            "result_data": {
                "evaluations": results,
                "best_plot_index": results[0]['plot_index'] if results else None,
                "worst_plot_index": results[-1]['plot_index'] if len(results) > 1 else None
            }
        })
        
    except Exception as e:
        error_msg = f'Evaluation error: {str(e)}'
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [ERROR] {error_msg}')
        return JSONResponse({"result_status": 500, "result_data": error_msg})

@app.post("/generate_and_evaluate")
async def generate_and_evaluate(request: Request):
    """Generate multiple plots and evaluate them"""
    try:
        req_data = await request.json()
        prompt = req_data.get("prompt")
        source_rating = req_data.get("source_rating")
        num_samples = req_data.get("num_samples", config.get('evaluation', {}).get('num_generations', 5))
        temperature = req_data.get("temperature", config.get('evaluation', {}).get('temperature', 0.8))
        
        if not prompt:
            return JSONResponse({"error": "Prompt is required"}, status_code=400)
        
        # Generate plots
        trainer = get_trainer()
        generated_plots = trainer.generate_plots(prompt, num_samples, temperature)
        
        # Evaluate plots
        results = []
        for i, plot in enumerate(generated_plots):
            evaluation = evaluate_plot_quality(plot, source_rating)
            results.append({
                "plot_index": i,
                "plot_text": plot,
                **evaluation
            })
        
        # Sort by combined score descending
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Save the best plot to training data if it meets quality threshold
        min_rating = config.get('evaluation', {}).get('min_rating_threshold', 7.0)
        best_plot = results[0]
        
        if best_plot['passed_rating_filter'] and best_plot['similarity_score'] > 0.6:
            training_example = {
                "instruction": "Write a detailed plot summary based on the following description:",
                "input": prompt,
                "output": best_plot['plot_text']
            }
            
            training_data_path = config.get('dataset', {}).get('training_data_path', './training_data.jsonl')
            with open(training_data_path, 'a') as f:
                f.write(json.dumps(training_example) + '\n')
            
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Added new training example to {training_data_path}')
        
        return JSONResponse({
            "result_status": 200,
            "result_data": {
                "evaluations": results,
                "best_plot_index": results[0]['plot_index'] if results else None,
                "best_plot": results[0]['plot_text'] if results else None,
                "training_example_added": best_plot['passed_rating_filter'] and best_plot['similarity_score'] > 0.6
            }
        })
        
    except Exception as e:
        error_msg = f'Generate and evaluate error: {str(e)}'
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [ERROR] {error_msg}')
        return JSONResponse({"result_status": 500, "result_data": error_msg})

@app.post("/train")
async def start_training(background_tasks: BackgroundTasks):
    """Start the training process"""
    global training_in_progress
    
    if training_in_progress:
        return JSONResponse({
            "result_status": 400,
            "result_data": "Training is already in progress"
        })
    
    background_tasks.add_task(train_model_async)
    
    return JSONResponse({
        "result_status": 200,
        "result_data": "Training started in background"
    })

@app.get("/training_status")
async def get_training_status():
    """Get the current training status"""
    return JSONResponse({
        "result_status": 200,
        "result_data": training_results
    })

@app.get("/download_model")
async def download_model():
    """Download the trained model"""
    output_dir = training_results.get("output_dir")
    
    if not output_dir or not os.path.exists(output_dir):
        return JSONResponse({
            "result_status": 404,
            "result_data": "Model not found or training not completed"
        })
    
    # Create a zip file of the model
    import shutil
    zip_path = f"{output_dir}.zip"
    shutil.make_archive(output_dir, 'zip', output_dir)
    
    return FileResponse(
        zip_path, 
        media_type="application/zip", 
        filename="trained_model.zip"
    )

@app.get("/status")
async def status():
    """Service status endpoint"""
    trainer = get_trainer()
    model_loaded = trainer.model is not None if trainer else False
    
    return JSONResponse({
        "result_status": 200,
        "result_data": {
            "status": "ok",
            "similarity_model_loaded": similarity_model is not None,
            "llm_model_loaded": model_loaded,
            "gold_standards_count": len(gold_standards),
            "redis_connected": redis_connection is not None,
            "training_in_progress": training_in_progress
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=f'{os.getenv("TRAIN_IP")}', port=int(os.getenv("TRAIN_PORT")))