import os
import sys
import torch
import warnings
from datetime import datetime
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import gc

warnings.filterwarnings("ignore", message="Some weights of the model checkpoint were not used when initializing")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")

# ########################################################
# ####  NOTE: Check `test_model_loras.py` for valid SDXL 1.0 models ####
# ########################################################

# ###########################
# ####  GLOBAL VARIABLES  ####
# ###########################
MODEL_ID = "Lykon/dreamshaper-xl-lightning"
MODEL_NAME = "dreamshaper-xl-lightning"
LORA_MODEL_ID = 'ntc-ai/SDXL-LoRA-slider.huge-anime-eyes'
LORA_WEIGHT_NAME = 'huge anime eyes.safetensors'
SEED = 42
PROMPT = "funny cat with huge anime eyes"
NEGATIVE_PROMPT = "blurry, scary"
WIDTH = 512
HEIGHT = 512
NUM_INFERENCE_STEPS = 50  # Increased steps for cleaner, crisper image
GUIDANCE_SCALE = 7.5      # Adjusted for better quality
GENERATED_IMAGES_FOLDER = "generated_images"
CHECKPOINT_PATH = os.path.join("..", "shared", "IP-Adapter", "models", "ip_adapter_faceid_sd15.bin")
CHECKPOINT_URL = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin?download=true"
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)

# ###########################
# ####  MAIN FUNCTIONALITY  ####
# ###########################

def setup_environment():
    print("Setting up the environment...")
    os.makedirs(GENERATED_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")

    shared_dir = os.path.join("..", "shared")
    sys.path.append(os.path.abspath(shared_dir))

    from get_ipadapter import ensure_setup
    from check_hub_size import check_free_space, free_up_space, CACHE_DIRECTORY, REQUIRED_FREE_SPACE_GB

    ensure_setup()

    free_space_gb = check_free_space(CACHE_DIRECTORY)
    print(f"Currently available free space: {free_space_gb:.2f} GB")
    if free_space_gb < REQUIRED_FREE_SPACE_GB:
        print("Not enough free space, initiating cleanup...")
        free_up_space(CACHE_DIRECTORY, REQUIRED_FREE_SPACE_GB)
    else:
        print("Enough free space is available, no cleanup needed.")

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

def load_pipeline():
    print(f"Loading the Diffusion Pipeline for model ID: {MODEL_ID}...")
    pipeline = DiffusionPipeline.from_pretrained(MODEL_ID)
    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    print("Pipeline loaded successfully.")
    return pipeline

def generate_image(pipeline, prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, apply_lora=False):
    try:
        if apply_lora:
            print("Loading LoRA weights...")
            pipeline.load_lora_weights(LORA_MODEL_ID, weight_name=LORA_WEIGHT_NAME, adapter_name="huge anime eyes")
            pipeline.set_adapters(["huge anime eyes"], adapter_weights=[2.0])
        else:
            pipeline.set_adapters([])

        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(SEED)
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]

        print(f"Image generation {'with' if apply_lora else 'without'} LoRA completed.")
        return image

    except Exception as e:
        print(f"Error during image generation: {e}")
        return None

def save_image(image, apply_lora):
    suffix = "with_lora" if apply_lora else "without_lora"
    filename = os.path.join(GENERATED_IMAGES_FOLDER, f"{MODEL_NAME}_{suffix}.png")
    image.save(filename)
    print(f"Generated and saved image as: {filename}")

def main():
    setup_environment()
    pipeline = load_pipeline()

    print("\nStarting image generation with LoRA...")
    image_with_lora = generate_image(pipeline, PROMPT, NEGATIVE_PROMPT, WIDTH, HEIGHT, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, apply_lora=True)
    if image_with_lora:
        save_image(image_with_lora, apply_lora=True)

    print("\nStarting image generation without LoRA...")
    image_without_lora = generate_image(pipeline, PROMPT, NEGATIVE_PROMPT, WIDTH, HEIGHT, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, apply_lora=False)
    if image_without_lora:
        save_image(image_without_lora, apply_lora=False)

    print("\nClearing CUDA cache...")
    torch.cuda.empty_cache()
    gc.collect()
    print("CUDA cache cleared.")

    print("\nImage generation process completed. Check the 'generated_images' folder for output images.")

if __name__ == "__main__":
    main()