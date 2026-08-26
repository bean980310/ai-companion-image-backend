"""
Inpainting generation pipeline using ComfyUI.
"""

import io
import json
import random
import traceback
from typing import List, Tuple, Optional, Dict, Any, Union

import pandas as pd
from PIL import Image, ImageFile
from PIL.ImageShow import show

from comfy_sdk import Comfy
from comfy_sdk.workflows import Workflow
from comfy_sdk.jobs import Job
from comfy_sdk.events import Progress, Preview, OutputReady, StatusChange

from ..comfy_api import ComfyUIClientWrapper
from ..comfyui_workflows import load_inpaint_workflow, load_inpaint_sdxl_workflow, load_inpaint_sdxl_with_refiner_workflow, load_inpaint_workflow_clip_skip, load_inpaint_sdxl_workflow_clip_skip, load_inpaint_sdxl_with_refiner_workflow_clip_skip

from ai_companion_core import logger


class InpaintPipeline:
    """Inpainting generation pipeline using ComfyUI."""

    def __init__(self, model: str, model_type: str = "checkpoint", refiner: Optional[str] = None, loras: Optional[List[str]] = None, vae: Optional[str] = None):
        self.client = Comfy()
        self.model = model
        self.model_type = model_type
        self.refiner = refiner
        self.loras = loras or []
        self.vae = vae

    def _get_seed(self, seed: int, random_seed: bool) -> int:
        if random_seed:
            return random.randint(0, 9007199254740991)
        return seed

    def _apply_loras(self, wf: Workflow, lora_text_weights: List[float], lora_unet_weights: List[float], base_node: str, start_node_id: int) -> Tuple[str, int]:
        current_node_id = start_node_id

        for i, lora in enumerate(self.loras):
            text_weight = lora_text_weights[i] if i < len(lora_text_weights) else 1.0
            unet_weight = lora_unet_weights[i] if i < len(lora_unet_weights) else 1.0
            new_node_id = str(current_node_id)
            wf.json[new_node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora,
                    "strength_model": text_weight,
                    "strength_clip": unet_weight,
                    "model": [base_node, 0],
                    "clip": [base_node, 1],
                },
            }
            base_node = new_node_id
            current_node_id += 1

        return base_node, current_node_id

    def _apply_vae(self, wf: Workflow, current_node_id: int, vae_target_node: str = "8") -> int:
        if self.vae == "Default":
            return current_node_id

        vae_value = self.vae
        new_node_id = str(current_node_id)
        wf.json[new_node_id] = {"class_type": "VAELoader", "inputs": {"vae_name": vae_value}}
        wf.set_input(vae_target_node, "vae", [new_node_id, 0])
        return current_node_id + 1

    def _process_results(self, generated: Job, history_data: Dict[str, Any]) -> Tuple[List[Image.Image], pd.DataFrame]:
        output_images = []
        width = history_data.get("Width", 0)
        height = history_data.get("Height", 0)

        for output in generated.get_outputs("9"):
            try:
                image_file = output.to_file(output.name)
                image = Image.open(image_file)
                output_images.append(image)
                width, height = image.size
            except Exception as e:
                logger.error(f"이미지 로딩 오류: {str(e)}\n\n{traceback.format_exc()}")

        history_data["Width"] = width
        history_data["Height"] = height

        history_df = pd.DataFrame([history_data])
        return output_images, history_df

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        style: str,
        generation_step: int,
        image_input: Union[str, Image.Image, ImageFile.ImageFile],
        denoise_strength: float,
        blur_radius: float,
        blur_expansion_radius: int,
        vae: str,
        clip_skip: int,
        enable_clip_skip: bool,
        clip_g: bool,
        sampler: str,
        scheduler: str,
        batch_count: int,
        cfg_scale: float,
        seed: int,
        random_seed: bool,
        lora_text_weights_json: str,
        lora_unet_weights_json: str,
    ) -> Tuple[List[Image.Image], Optional[pd.DataFrame]]:
        """
        Generate images using inpainting workflow.

        Args:
            positive_prompt: Positive prompt text
            negative_prompt: Negative prompt text
            style: Style preset
            generation_step: Number of generation steps
            image_input: Input image path or PIL Image
            denoise_strength: Denoising strength
            blur_radius: Mask blur radius
            blur_expansion_radius: Mask blur expansion radius
            vae: VAE model name or "Default"
            clip_skip: CLIP skip value
            enable_clip_skip: Whether to enable CLIP skip
            clip_g: Whether to use CLIP-G (SDXL)
            sampler: Sampler name
            scheduler: Scheduler name
            batch_count: Number of batches
            cfg_scale: CFG scale value
            seed: Random seed
            random_seed: Whether to use random seed
            lora_text_weights_json: JSON string of LoRA text weights
            lora_unet_weights_json: JSON string of LoRA UNet weights

        Returns:
            Tuple of (generated images list, history DataFrame)
        """
        try:
            seed = self._get_seed(seed, random_seed)

            lora_text_weights = json.loads(lora_text_weights_json)
            lora_unet_weights = json.loads(lora_unet_weights_json)

            if enable_clip_skip:
                clip_skip = clip_skip * (-1)

            # Load appropriate workflow
            if clip_g:
                if enable_clip_skip:
                    wf = self.client.workflows.from_file("../comfyui_workflows/inpaint_sdxl_clip_skip.json")
                else:
                    wf = self.client.workflows.from_file("../comfyui_workflows/inpaint_sdxl.json")
            else:
                if enable_clip_skip:
                    wf = self.client.workflows.from_file("../comfyui_workflows/inpaint_clip_skip.json")
                else:
                    wf = self.client.workflows.from_file("../comfyui_workflows/inpaint.json")

            # Configure sampler node
            wf.set_input("3", "cfg", cfg_scale)
            wf.set_input("3", "sampler_name", sampler)
            wf.set_input("3", "scheduler", scheduler)
            wf.set_input("3", "seed", seed)
            wf.set_input("3", "steps", generation_step)
            wf.set_input("3", "denoise", denoise_strength)
            wf.set_input("4", "ckpt_name", self.model)

            # Configure prompts
            if clip_g:
                wf.set_input("6", "text_l", positive_prompt)
                wf.set_input("6", "text_g", positive_prompt)
                wf.set_input("7", "text_l", negative_prompt)
                wf.set_input("7", "text_g", negative_prompt)
            else:
                wf.set_input("6", "text", positive_prompt)
                wf.set_input("7", "text", negative_prompt)

            # Set input image
            asset = self.client.assets.from_file(image_input)
            wf.set_input("10", "image", asset)

            # Set blur parameters
            wf.set_input("12", "blur_radius", blur_radius)
            wf.set_input("12", "blur_expansion_radius", blur_expansion_radius)

            if enable_clip_skip:
                wf.set_input("15", "stop_at_clip_layer", clip_skip)

            # Apply LoRAs
            base_node = "4"
            current_node_id = 16 if enable_clip_skip else 15

            base_node, current_node_id = self._apply_loras(wf, lora_text_weights, lora_unet_weights, base_node, current_node_id)

            wf.set_input("3", "model", [base_node, 0])
            if enable_clip_skip:
                wf.set_input("15", "clip", [base_node, 1])
            else:
                wf.set_input("6", "clip", [base_node, 1])
                wf.set_input("7", "clip", [base_node, 1])

            # Apply VAE
            current_node_id = self._apply_vae(wf, current_node_id)

            # Generate
            job = self.client.submit(wf)
            for event in job.events():
                match event:
                    case Progress() as p:
                        print(f"Progress: {p.value * 100:.2f}% - {p.message}")
                    case Preview() as pv:
                        show(pv.to_pil())
                    case OutputReady() as o:
                        o.output.to_file(f"partial/{o.output.name}")
                    case StatusChange(status="succeeded"):
                        break
            generated = job.result()

            history_data = {"Positive Prompt": positive_prompt, "Negative Prompt": negative_prompt, "Generation Steps": generation_step, "Model": self.model, "Sampler": sampler, "Scheduler": scheduler, "CFG Scale": cfg_scale, "Seed": seed, "Width": 0, "Height": 0}

            return self._process_results(generated, history_data)

        except Exception as e:
            logger.error(f"이미지 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
            return [], None

    def generate_with_refiner(
        self,
        positive_prompt: str,
        negative_prompt: str,
        style: str,
        generation_step: int,
        diffusion_img2img_start: int,
        diffusion_refiner_start: int,
        image_input: str,
        denoise_strength: float,
        blur_radius: float,
        blur_expansion_radius: int,
        vae: str,
        clip_skip: int,
        enable_clip_skip: bool,
        sampler: str,
        scheduler: str,
        batch_count: int,
        cfg_scale: float,
        seed: int,
        random_seed: bool,
        lora_text_weights_json: str,
        lora_unet_weights_json: str,
    ) -> Tuple[List[Image.Image], Optional[pd.DataFrame]]:
        """
        Generate images using inpainting workflow with refiner.
        """
        try:
            seed = self._get_seed(seed, random_seed)

            lora_text_weights = json.loads(lora_text_weights_json)
            lora_unet_weights = json.loads(lora_unet_weights_json)

            if enable_clip_skip:
                clip_skip = clip_skip * (-1)

            if enable_clip_skip:
                wf = self.client.workflows.from_file("../comfyui_workflows/inpaint_sdxl_with_refiner_clip_skip.json")
            else:
                wf = self.client.workflows.from_file("../comfyui_workflows/inpaint_sdxl_with_refiner.json")

            # Configure base sampler
            wf.set_input("3", "cfg", cfg_scale)
            wf.set_input("3", "sampler_name", sampler)
            wf.set_input("3", "scheduler", scheduler)
            wf.set_input("3", "seed", seed)
            wf.set_input("3", "steps", generation_step)
            wf.set_input("3", "start_at_step", diffusion_img2img_start)
            wf.set_input("3", "end_at_step", diffusion_refiner_start)
            wf.set_input("4", "ckpt_name", self.model)

            # Configure base prompts
            wf.set_input("6", "text_l", positive_prompt)
            wf.set_input("6", "text_g", positive_prompt)
            wf.set_input("7", "text_l", negative_prompt)
            wf.set_input("7", "text_g", negative_prompt)

            # Configure refiner sampler
            wf.set_input("10", "cfg", cfg_scale)
            wf.set_input("10", "sampler_name", sampler)
            wf.set_input("10", "scheduler", scheduler)
            wf.set_input("10", "noise_seed", seed)
            wf.set_input("10", "steps", generation_step)
            wf.set_input("10", "start_at_step", diffusion_refiner_start)

            # Configure refiner prompts
            wf.set_input("11", "text_l", positive_prompt)
            wf.set_input("11", "text_g", positive_prompt)
            wf.set_input("12", "text_l", negative_prompt)
            wf.set_input("12", "text_g", negative_prompt)
            wf.set_input("13", "ckpt_name", self.refiner)

            # Set input image
            asset = self.client.assets.from_file(image_input)
            wf.set_input("14", "image", asset)

            # Set blur parameters
            wf.set_input("16", "blur_radius", blur_radius)
            wf.set_input("16", "blur_expansion_radius", blur_expansion_radius)

            if enable_clip_skip:
                wf.set_input("19", "stop_at_clip_layer", clip_skip)

            # Apply LoRAs
            base_node = "4"
            current_node_id = 20 if enable_clip_skip else 19

            base_node, current_node_id = self._apply_loras(wf, lora_text_weights, lora_unet_weights, base_node, current_node_id)

            wf.set_input("3", "model", [base_node, 0])
            if enable_clip_skip:
                wf.set_input("19", "clip", [base_node, 1])
            else:
                wf.set_input("6", "clip", [base_node, 1])
                wf.set_input("7", "clip", [base_node, 1])

            # Apply VAE
            current_node_id = self._apply_vae(wf, current_node_id)

            # Generate
            job = self.client.submit(wf)
            for event in job.events():
                match event:
                    case Progress() as p:
                        print(f"Progress: {p.value * 100:.2f}% - {p.message}")
                    case Preview() as pv:
                        show(pv.to_pil())
                    case OutputReady() as o:
                        o.output.to_file(f"partial/{o.output.name}")
                    case StatusChange(status="succeeded"):
                        break
            generated = job.result()

            history_data = {"Positive Prompt": positive_prompt, "Negative Prompt": negative_prompt, "Generation Steps": generation_step, "Model": self.model, "Sampler": sampler, "Scheduler": scheduler, "CFG Scale": cfg_scale, "Seed": seed, "Width": 0, "Height": 0}

            return self._process_results(generated, history_data)

        except Exception as e:
            logger.error(f"이미지 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
            return [], None


# Backward compatibility functions
def generate_images_inpaint(
    positive_prompt: str,
    negative_prompt: str,
    style: str,
    generation_step: int,
    diffusion_model: str,
    diffusion_model_type: str,
    loras: List[str],
    vae: str,
    clip_skip: int,
    enable_clip_skip: bool,
    clip_g: bool,
    sampler: str,
    scheduler: str,
    batch_count: int,
    cfg_scale: float,
    seed: int,
    random_seed: bool,
    image_input: Union[str, Image.Image, ImageFile.ImageFile],
    denoise_strength: float,
    blur_radius: float,
    blur_expansion_radius: int,
    lora_text_weights_json: str,
    lora_unet_weights_json: str,
) -> Tuple[List[Image.Image], Optional[pd.DataFrame]]:
    """Backward compatible function for inpainting generation."""
    pipeline = InpaintPipeline(model=diffusion_model, model_type=diffusion_model_type, loras=loras, vae=vae)
    return pipeline.generate(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        style=style,
        generation_step=generation_step,
        image_input=image_input,
        denoise_strength=denoise_strength,
        blur_radius=blur_radius,
        blur_expansion_radius=blur_expansion_radius,
        vae=vae,
        clip_skip=clip_skip,
        enable_clip_skip=enable_clip_skip,
        clip_g=clip_g,
        sampler=sampler,
        scheduler=scheduler,
        batch_count=batch_count,
        cfg_scale=cfg_scale,
        seed=seed,
        random_seed=random_seed,
        lora_text_weights_json=lora_text_weights_json,
        lora_unet_weights_json=lora_unet_weights_json,
    )


def generate_images_inpaint_with_refiner(
    positive_prompt: str,
    negative_prompt: str,
    style: str,
    generation_step: int,
    diffusion_img2img_start: int,
    diffusion_refiner_start: int,
    diffusion_model: str,
    diffusion_refiner_model: str,
    diffusion_model_type: str,
    loras: List[str],
    vae: str,
    clip_skip: int,
    enable_clip_skip: bool,
    clip_g: bool,
    sampler: str,
    scheduler: str,
    batch_count: int,
    cfg_scale: float,
    seed: int,
    random_seed: bool,
    image_input: str,
    denoise_strength: float,
    blur_radius: float,
    blur_expansion_radius: int,
    lora_text_weights_json: str,
    lora_unet_weights_json: str,
) -> Tuple[List[Image.Image], Optional[pd.DataFrame]]:
    """Backward compatible function for inpainting generation with refiner."""
    pipeline = InpaintPipeline(model=diffusion_model, model_type=diffusion_model_type, refiner=diffusion_refiner_model, loras=loras, vae=vae)
    return pipeline.generate_with_refiner(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        style=style,
        generation_step=generation_step,
        diffusion_img2img_start=diffusion_img2img_start,
        diffusion_refiner_start=diffusion_refiner_start,
        image_input=image_input,
        denoise_strength=denoise_strength,
        blur_radius=blur_radius,
        blur_expansion_radius=blur_expansion_radius,
        vae=vae,
        clip_skip=clip_skip,
        enable_clip_skip=enable_clip_skip,
        sampler=sampler,
        scheduler=scheduler,
        batch_count=batch_count,
        cfg_scale=cfg_scale,
        seed=seed,
        random_seed=random_seed,
        lora_text_weights_json=lora_text_weights_json,
        lora_unet_weights_json=lora_unet_weights_json,
    )
