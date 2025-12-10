#!/usr/bin/env python3
"""
Test 8-bit quantized inference for Qwen-Image-Edit-2509
Reduces memory from ~47GB to ~24GB
"""

import torch
from diffusers import QwenImageEditPlusPipeline
from transformers import BitsAndBytesConfig
from PIL import Image
import numpy as np
import time
import os

# Use GPU 7 which has most free memory
GPU_ID = 7
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

def create_test_image(size=(512, 512)):
    """Create a simple test image"""
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(img)

def main():
    print("=" * 60)
    print("Qwen-Image-Edit-2509 Quantized Inference Test")
    print("=" * 60)

    device = torch.device('cuda:0')

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Check current GPU memory
    torch.cuda.empty_cache()
    free_mem = torch.cuda.mem_get_info()[0] / 1e9
    print(f"GPU Free Memory: {free_mem:.1f} GB")

    # Configure 8-bit quantization
    quantization_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

    # Configure 4-bit quantization (more aggressive)
    quantization_config_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("\n[1] Loading model with 8-bit quantization...")
    start_time = time.time()

    try:
        # Try loading with quantization
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            quantization_config=quantization_config_8bit,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        load_time = time.time() - start_time
        print(f"    Loaded in {load_time:.1f}s")
        print(f"    GPU Memory Used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    except Exception as e:
        print(f"    8-bit failed: {e}")
        print("\n[1b] Trying 4-bit quantization...")

        torch.cuda.empty_cache()
        try:
            pipeline = QwenImageEditPlusPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2509",
                quantization_config=quantization_config_4bit,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            load_time = time.time() - start_time
            print(f"    Loaded in {load_time:.1f}s")
            print(f"    GPU Memory Used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        except Exception as e2:
            print(f"    4-bit also failed: {e2}")
            print("\n    Falling back to CPU offloading...")

            torch.cuda.empty_cache()
            pipeline = QwenImageEditPlusPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2509",
                torch_dtype=torch.bfloat16,
            )
            pipeline.enable_model_cpu_offload()

    # Create test image
    test_img = create_test_image((512, 512))
    prompt = "Transform this into an oil painting style"

    print("\n[2] Warmup inference...")
    with torch.no_grad():
        _ = pipeline(
            image=[test_img],
            prompt=prompt,
            generator=torch.manual_seed(42),
            true_cfg_scale=4.0,
            negative_prompt=" ",
            num_inference_steps=4,
            guidance_scale=1.0,
        )
    torch.cuda.synchronize()
    print("    Warmup complete")

    # Benchmark
    print("\n[3] Benchmarking inference speed...")
    for steps in [2, 4]:
        times = []
        for i in range(3):
            torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                _ = pipeline(
                    image=[test_img],
                    prompt=prompt,
                    generator=torch.manual_seed(42 + i),
                    true_cfg_scale=4.0,
                    negative_prompt=" ",
                    num_inference_steps=steps,
                    guidance_scale=1.0,
                )

            torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        time_per_step = avg_time / steps

        print(f"    Steps={steps}: {avg_time:.2f}s total, {time_per_step:.3f}s/step, {fps:.2f} fps")

    print("\n[4] Final GPU Memory:")
    print(f"    Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
