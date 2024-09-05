import os
import sys
import torch
import warnings
import requests
from datetime import datetime
from time import time
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, DiffusionPipeline
import random
import gc
import csv
from pathlib import Path

warnings.filterwarnings("ignore", message="Some weights of the model checkpoint were not used when initializing")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")


# ########################################################
# ####  NOTE: LoRA models for SDXL 1.0 work but not for SD 1.5, so for example this will break if added to the model list: digiplay/Photon_v1  ####
# ########################################################

# ###########################
# ####  GLOBAL VARIABLES  ####
# ###########################
MODEL_URL_1 = "https://huggingface.co/martyn/sdxl-turbo-mario-merge-top-rated/blob/main/topRatedTurboxlLCM_v10.safetensors"
MODEL_NAME_1 = "sdxl-turbo-mario-merge-top-rated"
MODEL_ID_2 = "Lykon/dreamshaper-xl-lightning"
MODEL_NAME_2 = "dreamshaper-xl-lightning"

# List of SDXL model paths to test
model_paths = [
    "RunDiffusion/Juggernaut-XL-Lightning",
    "RunDiffusion/juggernaut-xl-v8",
    "RunDiffusion/Juggernaut-X-v10",
    "stablediffusionapi/realism-engine-sdxl-v30",
]

LORA_MODEL_ID = 'ntc-ai/SDXL-LoRA-slider.huge-anime-eyes'
LORA_WEIGHT_NAME = 'huge anime eyes.safetensors'
SEED = 42
RANDOMIZE_SEED = False
PROMPT = "beautiful girl"
NEGATIVE_PROMPT = "sad person"
WIDTH = 512
HEIGHT = 512
NUM_INFERENCE_STEPS = 10
GUIDANCE_SCALE = 2
GENERATED_IMAGES_FOLDER = "generated_images"

CHECKPOINT_PATH = os.path.join("..", "shared", "IP-Adapter", "models", "ip_adapter_faceid_sd15.bin")
CHECKPOINT_URL = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin?download=true"
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)
# ###########################

# Set a fixed or randomized seed
if RANDOMIZE_SEED:
    SEED = random.randint(0, 2**32 - 1)
    print(f"Randomized seed: {SEED}")
else:
    print(f"Using fixed seed: {SEED}")

torch.manual_seed(SEED)

os.makedirs(GENERATED_IMAGES_FOLDER, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

shared_dir = os.path.join("..", "shared")
sys.path.append(os.path.abspath(shared_dir))

from get_ipadapter import ensure_setup
from check_hub_size import check_free_space, free_up_space, CACHE_DIRECTORY, REQUIRED_FREE_SPACE_GB

ensure_setup()

def check_and_manage_disk_space():
    free_space_gb = check_free_space(CACHE_DIRECTORY)
    print(f"Currently available free space: {free_space_gb:.2f} GB")
    if free_space_gb < REQUIRED_FREE_SPACE_GB:
        print("Not enough free space, initiating cleanup...")
        free_up_space(CACHE_DIRECTORY, REQUIRED_FREE_SPACE_GB)
    else:
        print("Enough free space is available, no cleanup needed.")

check_and_manage_disk_space()

if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

def download_file(url, filename):
    if not os.path.isfile(filename):
        print(f"Downloading {filename} from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists.")

print(f"Checking for checkpoint file at {CHECKPOINT_PATH}...")
if os.path.isfile(CHECKPOINT_PATH):
    print(f"Checkpoint file exists: {CHECKPOINT_PATH}")
else:
    print(f"Checkpoint file does not exist. Downloading...")
    download_file(CHECKPOINT_URL, CHECKPOINT_PATH)

# Load the pipeline from a URL
def load_pipeline_from_url(url):
    print(f"Loading pipeline from URL: {url}")
    pipeline = StableDiffusionXLPipeline.from_single_file(url)
    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    print("Pipeline loaded from URL.")
    return pipeline

# Load the pipeline from a pretrained model
def load_pipeline_from_pretrained(model_id):
    print(f"Loading pipeline from pretrained model ID: {model_id}")
    pipeline = DiffusionPipeline.from_pretrained(model_id)
    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    print("Pipeline loaded from pretrained model.")
    return pipeline

def clear_cuda_cache():
    print("Clearing CUDA cache...")
    torch.cuda.empty_cache()
    gc.collect()
    print("CUDA cache cleared.")

def generate_image(pipeline, prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, apply_lora=False, lora_strength=2.0, model_name=""):
    try:
        if apply_lora:
            print("Loading LoRA weights.")
            pipeline.load_lora_weights(LORA_MODEL_ID, weight_name=LORA_WEIGHT_NAME, adapter_name="huge anime eyes")
            pipeline.set_adapters(["huge anime eyes"], adapter_weights=[lora_strength])
        else:
            pipeline.set_adapters([])

        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(SEED)
        images = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "with_lora" if apply_lora else "without_lora"
        output_image_path = os.path.join(GENERATED_IMAGES_FOLDER, f"{timestamp}_{model_name}_{suffix}.png")
        images[0].save(output_image_path)  # Save the first generated image
        print(f"Generated and saved image as: {output_image_path}")
        return output_image_path, True, ""
    except Exception as e:
        print(f"Error during image generation: {e}")
        return "", False, str(e)

log_entries = []

start_time = time()

# Load and generate with the first URL model
check_and_manage_disk_space()
pipeline_url_model = load_pipeline_from_url(MODEL_URL_1)
start_time_with_lora = time()
output_image_with_lora, success_with_lora, error_with_lora = generate_image(
    pipeline=pipeline_url_model,
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    width=WIDTH,
    height=HEIGHT,
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    apply_lora=True,
    lora_strength=3.0,
    model_name=MODEL_NAME_1
)
time_with_lora_url = time() - start_time_with_lora

log_entries.append({
    "model": MODEL_NAME_1,
    "result": "success" if success_with_lora else "failure",
    "error_message": error_with_lora,
    "image_with_lora": output_image_with_lora,
    "image_without_lora": "",
    "time_taken_with": time_with_lora_url
})

start_time_without_lora_model1 = time()
output_image_without_lora, success_without_lora, error_without_lora = generate_image(
    pipeline=pipeline_url_model,
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    width=WIDTH,
    height=HEIGHT,
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    apply_lora=False,
    model_name=MODEL_NAME_1 + "_no_lora_added"
)
time_without_lora_model1 = time() - start_time_without_lora_model1

log_entries[-1].update({
    "image_without_lora": output_image_without_lora,
    "time_taken_without": time_without_lora_model1,
})

# Clear CUDA memory
del pipeline_url_model
clear_cuda_cache()

# Load and generate with the second model (pretrained)
check_and_manage_disk_space()
pipeline2 = load_pipeline_from_pretrained(MODEL_ID_2)
start_time_with_lora_model2 = time()
output_image_with_lora, success_with_lora, error_with_lora = generate_image(
    pipeline=pipeline2,
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    width=WIDTH,
    height=HEIGHT,
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    apply_lora=True,
    lora_strength=3.0,
    model_name=MODEL_NAME_2
)
time_with_lora_model2 = time() - start_time_with_lora_model2

log_entries.append({
    "model": MODEL_NAME_2,
    "result": "success" if success_with_lora else "failure",
    "error_message": error_with_lora,
    "image_with_lora": output_image_with_lora,
    "image_without_lora": "",
    "time_taken_with": time_with_lora_model2,
})

start_time_without_lora_model2 = time()
output_image_without_lora, success_without_lora, error_without_lora = generate_image(
    pipeline=pipeline2,
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    width=WIDTH,
    height=HEIGHT,
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    apply_lora=False,
    model_name=MODEL_NAME_2 + "_no_lora_added"
)
time_without_lora_model2 = time() - start_time_without_lora_model2

log_entries[-1].update({
    "image_without_lora": output_image_without_lora,
    "time_taken_without": time_without_lora_model2,
})

# Clear CUDA memory again
del pipeline2
clear_cuda_cache()

# Load and generate with each pretrained model in the list
for model_id in model_paths:
    try:
        check_and_manage_disk_space()
        print(f"Processing model: {model_id}")
        pipeline = load_pipeline_from_pretrained(model_id)
        model_name = model_id.split("/")[-1]

        start_time_with_lora_model = time()
        output_image_with_lora, success_with_lora, error_with_lora = generate_image(
            pipeline=pipeline,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            apply_lora=True,
            lora_strength=3.0,
            model_name=model_name
        )
        time_with_lora_model = time() - start_time_with_lora_model

        log_entries.append({
            "model": model_name,
            "result": "success" if success_with_lora else "failure",
            "error_message": error_with_lora,
            "image_with_lora": output_image_with_lora,
            "image_without_lora": "",
            "time_taken_with": time_with_lora_model,
        })

        start_time_without_lora_model = time()
        output_image_without_lora, success_without_lora, error_without_lora = generate_image(
            pipeline=pipeline,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            apply_lora=False,
            model_name=model_name + "_no_lora_added"
        )
        time_without_lora_model = time() - start_time_without_lora_model

        log_entries[-1].update({
            "image_without_lora": output_image_without_lora,
            "time_taken_without": time_without_lora_model,
        })

        # Clear CUDA memory
        del pipeline
        clear_cuda_cache()

        print(f"Generated images with/without LoRA for model: {model_name}")
        print(f"Time with LoRA: {time_with_lora_model:.2f} seconds")
        print(f"Time without LoRA: {time_without_lora_model:.2f} seconds")

    except Exception as e:
        log_entries.append({
            "model": model_id.split("/")[-1],
            "result": "failure",
            "error_message": str(e),
            "image_with_lora": "",
            "image_without_lora": "",
            "time_taken_with": 0,
            "time_taken_without": 0,
        })
        print(f"Failed to process model: {model_id}")
        print(f"Error: {e}")
        continue

total_time = time() - start_time

print("\n===== SUMMARY =====")
print(f"Time taken to generate image with LoRA (Model 1, {MODEL_NAME_1}): {time_with_lora_url:.2f} seconds")
print(f"Time taken to generate image without LoRA (Model 1, {MODEL_NAME_1}_no_lora_added): {time_without_lora_model1:.2f} seconds")
print(f"Time taken to generate image with LoRA (Model 2, {MODEL_NAME_2}): {time_with_lora_model2:.2f} seconds")
print(f"Time taken to generate image without LoRA (Model 2, {MODEL_NAME_2}_no_lora_added): {time_without_lora_model2:.2f} seconds")
print(f"Total time taken for all images: {total_time:.2f} seconds")

# Write the results to a CSV file
csv_columns = ["model", "result", "error_message", "image_with_lora", "image_without_lora", "time_taken_with", "time_taken_without"]
csv_file = "testing_results.csv"

try:
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for log in log_entries:
            writer.writerow(log)
    print(f"Results saved to {csv_file}")
except IOError:
    print("I/O error while writing CSV file")