#!/usr/bin/env python3
"""
Qwen-Image-Edit-2509 Real-time WebUI
Webcam input with live editing at ~0.4 fps
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
import gradio as gr
from PIL import Image
import numpy as np
import time
from threading import Thread, Event
from queue import Queue
import cv2

# Global state
pipeline = None
is_running = False
stop_event = Event()
current_prompt = "Transform into oil painting style"
processing_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=1)


def load_model():
    """Load the model on startup"""
    global pipeline

    print("Loading Qwen-Image-Edit-2509...")
    from diffusers import QwenImageEditPlusPipeline

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16,
    ).to('cuda')

    # Disable torch.compile - causes slowdown in this context
    # pipeline.transformer = torch.compile(
    #     pipeline.transformer,
    #     mode="default",
    #     fullgraph=False,
    # )

    print(f"Model loaded! GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Warmup with 4 steps (1 step is unstable)
    print("Warming up with 4 steps...")
    dummy_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    with torch.no_grad():
        _ = pipeline(
            image=[dummy_img],
            prompt="test",
            generator=torch.Generator(device='cuda').manual_seed(42),
            true_cfg_scale=4.0,
            negative_prompt=" ",
            num_inference_steps=4,
            guidance_scale=1.0,
        )
    torch.cuda.synchronize()
    print("Ready!")
    return "Model loaded and ready!"


def process_single_frame(image: Image.Image, prompt: str, steps: int = 1) -> Image.Image:
    """Process a single frame"""
    global pipeline

    if pipeline is None:
        return image

    # Resize for speed
    image = image.resize((512, 512), Image.LANCZOS)

    with torch.no_grad():
        result = pipeline(
            image=[image],
            prompt=prompt,
            generator=torch.Generator(device='cuda').manual_seed(42),
            true_cfg_scale=4.0,
            negative_prompt=" ",
            num_inference_steps=steps,
            guidance_scale=1.0,
        )

    return result.images[0]


def process_image(input_image, prompt, steps):
    """Process uploaded image"""
    if input_image is None:
        return None, "No image provided"

    if pipeline is None:
        return None, "Model not loaded! Click 'Load Model' first."

    start = time.time()

    # Convert numpy to PIL if needed
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)

    result = process_single_frame(input_image, prompt, int(steps))

    elapsed = time.time() - start
    fps = 1.0 / elapsed

    return result, f"Processed in {elapsed:.2f}s ({fps:.2f} fps)"


def webcam_stream(webcam_input, prompt, steps):
    """Process webcam frame"""
    if webcam_input is None:
        return None, "Waiting for webcam..."

    if pipeline is None:
        return webcam_input, "Model not loaded!"

    start = time.time()

    # Convert to PIL
    if isinstance(webcam_input, np.ndarray):
        input_image = Image.fromarray(webcam_input)
    else:
        input_image = webcam_input

    result = process_single_frame(input_image, prompt, int(steps))

    elapsed = time.time() - start
    fps = 1.0 / elapsed

    return result, f"{elapsed:.2f}s ({fps:.2f} fps)"


def blend_images(img1, img2, blend_ratio):
    """Blend two images together"""
    if img1 is None or img2 is None:
        return None

    # Convert to PIL
    if isinstance(img1, np.ndarray):
        img1 = Image.fromarray(img1)
    if isinstance(img2, np.ndarray):
        img2 = Image.fromarray(img2)

    # Resize to same size
    size = (512, 512)
    img1 = img1.resize(size, Image.LANCZOS)
    img2 = img2.resize(size, Image.LANCZOS)

    # Blend
    blended = Image.blend(img1, img2, blend_ratio)
    return blended


def process_composite(source_img, ref_img, blend_ratio, prompt, steps):
    """Composite two images and process"""
    if source_img is None:
        return None, None, "Source image required"

    if pipeline is None:
        return None, None, "Model not loaded!"

    start = time.time()

    # If no reference, just process source
    if ref_img is None:
        if isinstance(source_img, np.ndarray):
            source_img = Image.fromarray(source_img)
        blended = source_img.resize((512, 512), Image.LANCZOS)
    else:
        blended = blend_images(source_img, ref_img, blend_ratio)

    if blended is None:
        return None, None, "Blend failed"

    result = process_single_frame(blended, prompt, int(steps))

    elapsed = time.time() - start
    return blended, result, f"Processed in {elapsed:.2f}s"


# Create Gradio interface
def create_ui():
    with gr.Blocks(title="Qwen Image Edit - Real-time") as demo:
        gr.Markdown("# Qwen-Image-Edit-2509 Real-time Demo")
        gr.Markdown("Real-time image editing with webcam input (~0.4 fps with 1 step)")

        with gr.Row():
            load_btn = gr.Button("🚀 Load Model", variant="primary", scale=1)
            status = gr.Textbox(label="Status", value="Click 'Load Model' to start", scale=2)

        with gr.Row():
            prompt = gr.Textbox(
                label="Edit Prompt",
                value="Transform into oil painting style",
                placeholder="Enter your edit prompt..."
            )
            steps = gr.Slider(
                minimum=2, maximum=8, value=4, step=1,
                label="Inference Steps (2=fastest, 4=recommended)"
            )

        with gr.Tabs():
            # Tab 1: Webcam streaming
            with gr.Tab("📹 Webcam"):
                gr.Markdown("Click the webcam to capture, then it will be processed")
                with gr.Row():
                    webcam = gr.Image(
                        sources=["webcam"],
                        label="Webcam Input",
                        streaming=False,  # Capture mode
                    )
                    webcam_output = gr.Image(label="Edited Output")

                webcam_status = gr.Textbox(label="Processing Time")
                process_webcam_btn = gr.Button("🎨 Process Frame", variant="primary")

            # Tab 2: Upload image
            with gr.Tab("📁 Upload Image"):
                with gr.Row():
                    upload_input = gr.Image(
                        label="Upload Image",
                        type="numpy"
                    )
                    upload_output = gr.Image(label="Edited Output")

                upload_status = gr.Textbox(label="Processing Time")
                process_upload_btn = gr.Button("🎨 Process Image", variant="primary")

            # Tab 3: Image Composite
            with gr.Tab("🔀 Composite"):
                gr.Markdown("Blend two images and process with the model")
                with gr.Row():
                    with gr.Column():
                        source_input = gr.Image(
                            label="Source Image (Webcam/Upload)",
                            sources=["webcam", "upload"],
                            type="numpy"
                        )
                        ref_input = gr.Image(
                            label="Reference Image",
                            type="numpy"
                        )
                        blend_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                            label="Blend Ratio (0=Source, 1=Reference)"
                        )
                    with gr.Column():
                        blend_preview = gr.Image(label="Blended Preview")
                        composite_output = gr.Image(label="Edited Output")

                composite_status = gr.Textbox(label="Processing Time")
                process_composite_btn = gr.Button("🎨 Process Composite", variant="primary")

        # Examples
        gr.Markdown("### Example Prompts")
        gr.Examples(
            examples=[
                ["Transform into oil painting style"],
                ["Make it look like a watercolor painting"],
                ["Convert to anime style"],
                ["Add a sunset lighting effect"],
                ["Make it look like a pencil sketch"],
                ["Transform into Van Gogh style"],
            ],
            inputs=[prompt]
        )

        # Event handlers
        load_btn.click(
            fn=load_model,
            outputs=[status]
        )

        process_webcam_btn.click(
            fn=webcam_stream,
            inputs=[webcam, prompt, steps],
            outputs=[webcam_output, webcam_status]
        )

        process_upload_btn.click(
            fn=process_image,
            inputs=[upload_input, prompt, steps],
            outputs=[upload_output, upload_status]
        )

        # Auto-process on webcam capture
        webcam.change(
            fn=webcam_stream,
            inputs=[webcam, prompt, steps],
            outputs=[webcam_output, webcam_status]
        )

        # Composite processing
        process_composite_btn.click(
            fn=process_composite,
            inputs=[source_input, ref_input, blend_slider, prompt, steps],
            outputs=[blend_preview, composite_output, composite_status]
        )

        # Preview blend on slider change
        def preview_blend(src, ref, ratio):
            if src is None:
                return None
            blended = blend_images(src, ref, ratio) if ref is not None else src
            return blended

        blend_slider.change(
            fn=preview_blend,
            inputs=[source_input, ref_input, blend_slider],
            outputs=[blend_preview]
        )

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen-Image-Edit-2509 Real-time WebUI")
    print("=" * 60)

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=False,
        show_error=True,
    )
