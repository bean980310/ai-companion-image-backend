"""
Text-to-Image generation pipeline using ComfyUI.
"""

from PIL.ImageShow import show

import io
import json
import random
import traceback
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
from PIL import Image

from comfy_sdk import Comfy
from comfy_sdk.workflows import Workflow
from comfy_sdk.jobs import Job
from comfy_sdk.events import Progress, Preview, OutputReady, StatusChange

from ..comfy_api import ComfyUIClientWrapper
from ..comfyui_workflows import load_txt2img_workflow, load_txt2img_sdxl_workflow, load_txt2img_sdxl_with_refiner_workflow, load_txt2img_workflow_clip_skip, load_txt2img_sdxl_workflow_clip_skip, load_txt2img_sdxl_with_refiner_workflow_clip_skip

from ai_companion_core import logger

from tqdm import tqdm


class Txt2ImgPipeline:
    """Text-to-Image generation pipeline using ComfyUI."""

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

            wf.json[new_node_id] = {"class_type": "LoraLoader", "inputs": {"lora_name": lora, "strength_model": text_weight, "strength_clip": unet_weight, "model": [base_node, 0], "clip": [base_node, 1]}}

            # prompt[new_node_id] = {"class_type": "LoraLoader", "inputs": {"lora_name": lora, "strength_model": text_weight, "strength_clip": unet_weight, "model": [base_node, 0], "clip": [base_node, 1]}}
            base_node = new_node_id
            current_node_id += 1

        return base_node, current_node_id

    def _apply_vae(self, wf: Workflow, base_node: str, current_node_id: int, vae_target_node: str = "8") -> Tuple[str, int]:
        if self.vae == "Default":
            return base_node, current_node_id

        vae_value = self.vae
        new_node_id = str(current_node_id)
        wf.json[new_node_id] = {"class_type": "VAELoader", "inputs": {"vae_name": vae_value}}
        # prompt[new_node_id] = {"class_type": "VAELoader", "inputs": {"vae_name": vae_value}}
        base_node = new_node_id
        current_node_id += 1
        wf.set_input(vae_target_node, "vae", [base_node, 0])
        # prompt[vae_target_node]["inputs"]["vae"] = [base_node, 0]

        return base_node, current_node_id

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

        # Update width/height in history if changed by image loading
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
        width: int,
        height: int,
        clip_skip: int,
        enable_clip_skip: bool,
        clip_g: bool,
        sampler: str,
        scheduler: str,
        batch_size: int,
        batch_count: int,
        cfg_scale: float,
        seed: int,
        random_seed: bool,
        lora_text_weights_json: str,
        lora_unet_weights_json: str,
    ) -> Tuple[List[Image.Image], Optional[pd.DataFrame]]:
        try:
            seed = self._get_seed(seed, random_seed)

            lora_text_weights = json.loads(lora_text_weights_json)
            lora_unet_weights = json.loads(lora_unet_weights_json)

            if enable_clip_skip:
                clip_skip = clip_skip * (-1)

            if clip_g:
                if enable_clip_skip:
                    wf = self.client.workflows.from_file(path="../comfyui_workflows/txt2img_sdxl_clip_skip.json")
                    # prompt = load_txt2img_sdxl_workflow_clip_skip()
                else:
                    wf = self.client.workflows.from_file(path="../comfyui_workflows/txt2img_sdxl.json")
                    # prompt = load_txt2img_sdxl_workflow()
            else:
                if enable_clip_skip:
                    wf = self.client.workflows.from_file(path="../comfyui_workflows/txt2img_clip_skip.json")
                    # prompt = load_txt2img_workflow_clip_skip()
                else:
                    wf = self.client.workflows.from_file(path="../comfyui_workflows/txt2img.json")
                    # prompt = load_txt2img_workflow()

            wf.set_input("3", "cfg", cfg_scale)
            wf.set_input("3", "sampler_name", sampler)
            wf.set_input("3", "scheduler", scheduler)
            wf.set_input("3", "seed", seed)
            wf.set_input("3", "steps", generation_step)
            wf.set_input("4", "ckpt_name", self.model)
            wf.set_input("5", "batch_size", batch_size)
            wf.set_input("5", "width", width)
            wf.set_input("5", "height", height)

            # prompt["3"]["inputs"]["cfg"] = cfg_scale
            # prompt["3"]["inputs"]["sampler_name"] = sampler
            # prompt["3"]["inputs"]["scheduler"] = scheduler
            # prompt["3"]["inputs"]["seed"] = seed
            # prompt["3"]["inputs"]["steps"] = generation_step
            # prompt["4"]["inputs"]["ckpt_name"] = self.model
            # prompt["5"]["inputs"]["batch_size"] = batch_size
            # prompt["5"]["inputs"]["width"] = width
            # prompt["5"]["inputs"]["height"] = height

            if clip_g:
                wf.set_input("6", "text_l", positive_prompt)
                wf.set_input("6", "text_g", positive_prompt)
                wf.set_input("7", "text_l", negative_prompt)
                wf.set_input("7", "text_g", negative_prompt)
                # prompt["6"]["inputs"]["text_l"] = positive_prompt
                # prompt["6"]["inputs"]["text_g"] = positive_prompt
                # prompt["7"]["inputs"]["text_l"] = negative_prompt
                # prompt["7"]["inputs"]["text_g"] = negative_prompt
            else:
                wf.set_input("6", "text", positive_prompt)
                wf.set_input("7", "text", negative_prompt)
                # prompt["6"]["inputs"]["text"] = positive_prompt
                # prompt["7"]["inputs"]["text"] = negative_prompt

            if enable_clip_skip:
                wf.set_input("10", "stop_at_clip_layer", clip_skip)
                # prompt["10"]["inputs"]["stop_at_clip_layer"] = clip_skip

            base_node = "4"
            if enable_clip_skip:
                current_node_id = 11
            else:
                current_node_id = 10

            base_node, current_node_id = self._apply_loras(wf, lora_text_weights, lora_unet_weights, base_node, current_node_id)

            # prompt["3"]["inputs"]["model"] = [base_node, 0]
            wf.set_input("3", "model", [base_node, 0])  # Ensure the workflow also has the model input set
            if enable_clip_skip:
                wf.set_input("10", "clip", [base_node, 1])
                # prompt["10"]["inputs"]["clip"] = [base_node, 1]
            else:
                wf.set_input("6", "clip", [base_node, 1])
                wf.set_input("7", "clip", [base_node, 1])
                # prompt["6"]["inputs"]["clip"] = [base_node, 1]
                # prompt["7"]["inputs"]["clip"] = [base_node, 1]

            # VAE handling
            # Note: The original code logic for VAE seems to use a new node ID sequence continuing from Lora
            # But wait, in original code:
            # base_node = new_node_id (from lora loop)
            # prompt["3"]["inputs"]["model"] = [base_node, 0]
            # ... clip connections ...
            # THEN vae check.
            # If vae != Default:
            # new_node_id = str(current_node_id) ...
            # base_node = new_node_id
            # prompt["8"]["inputs"]["vae"] = [base_node, 0]

            # So the base_node for VAE is the last Lora node (or the checkpoint if no loras).
            # But wait, VAE Loader usually doesn't take model/clip as input, it just loads VAE.
            # Ah, looking at the original code:
            # prompt[new_node_id] = { "class_type": "VAELoader", "inputs": { "vae_name": vae_value } }
            # base_node = new_node_id
            # prompt["8"]["inputs"]["vae"] = [base_node, 0]
            # It seems it just sets base_node to the VAE loader ID, and then connects node 8 (VAE Decode?) to it.
            # Yes, standard VAE Loader.

            _, current_node_id = self._apply_vae(wf, base_node, current_node_id)

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
            # for output in result.get_outputs("9"):
            # generated = self.client.text2image_generate(prompt)

            history_data = {"Positive Prompt": positive_prompt, "Negative Prompt": negative_prompt, "Generation Steps": generation_step, "Model": self.model, "Sampler": sampler, "Scheduler": scheduler, "CFG Scale": cfg_scale, "Seed": seed, "Width": width, "Height": height}

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
        diffusion_refiner_start: int,
        width: int,
        height: int,
        clip_skip: int,
        enable_clip_skip: bool,
        clip_g: bool,
        sampler: str,
        scheduler: str,
        batch_size: int,
        batch_count: int,
        cfg_scale: float,
        seed: int,
        random_seed: bool,
        lora_text_weights_json: str,
        lora_unet_weights_json: str,
    ) -> Tuple[List[Image.Image], Optional[pd.DataFrame]]:
        try:
            seed = self._get_seed(seed, random_seed)

            lora_text_weights = json.loads(lora_text_weights_json)
            lora_unet_weights = json.loads(lora_unet_weights_json)

            if enable_clip_skip:
                clip_skip = clip_skip * (-1)

            if enable_clip_skip:
                wf = self.client.workflows.from_file(path="../comfyui_workflows/txt2img_sdxl_with_refiner_clip_skip.json")
                # prompt = load_txt2img_sdxl_with_refiner_workflow_clip_skip()
            else:
                wf = self.client.workflows.from_file(path="../comfyui_workflows/txt2img_sdxl_with_refiner.json")
                # prompt = load_txt2img_sdxl_with_refiner_workflow()

            wf.set_input("3", "cfg", cfg_scale)
            wf.set_input("3", "sampler_name", sampler)
            wf.set_input("3", "scheduler", scheduler)
            wf.set_input("3", "seed", seed)
            wf.set_input("3", "steps", generation_step)
            wf.set_input("3", "end_at_step", diffusion_refiner_start)
            wf.set_input("4", "ckpt_name", self.model)
            wf.set_input("5", "batch_size", batch_size)
            wf.set_input("5", "width", width)
            wf.set_input("5", "height", height)
            wf.set_input("6", "text_l", positive_prompt)
            wf.set_input("6", "text_g", positive_prompt)
            wf.set_input("7", "text_l", negative_prompt)
            wf.set_input("7", "text_g", negative_prompt)
            wf.set_input("10", "cfg", cfg_scale)
            wf.set_input("10", "sampler_name", sampler)
            wf.set_input("10", "scheduler", scheduler)
            wf.set_input("10", "seed", seed)
            wf.set_input("10", "steps", generation_step)
            wf.set_input("10", "start_at_step", diffusion_refiner_start)
            wf.set_input("11", "text_l", positive_prompt)
            wf.set_input("11", "text_g", positive_prompt)
            wf.set_input("12", "text_l", negative_prompt)
            wf.set_input("12", "text_g", negative_prompt)
            wf.set_input("13", "ckpt_name", self.refiner)

            if enable_clip_skip:
                wf.set_input("14", "stop_at_clip_layer", clip_skip)
                # prompt["14"]["inputs"]["stop_at_clip_layer"] = clip_skip

            base_node = "4"
            if enable_clip_skip:
                current_node_id = 15
            else:
                current_node_id = 14

            base_node, current_node_id = self._apply_loras(wf, lora_text_weights, lora_unet_weights, base_node, current_node_id)

            wf.set_input("3", "model", [base_node, 0])
            if enable_clip_skip:
                wf.set_input("14", "clip", [base_node, 1])
            else:
                wf.set_input("6", "clip", [base_node, 1])
                wf.set_input("7", "clip", [base_node, 1])

            # Note: The original code had commented out refiner lora logic. I will keep it that way (omitted).

            _, current_node_id = self._apply_vae(wf, base_node, current_node_id)

            # generated = self.client.text2image_generate(prompt)
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

            history_data = {"Positive Prompt": positive_prompt, "Negative Prompt": negative_prompt, "Generation Steps": generation_step, "Model": self.model, "Sampler": sampler, "Scheduler": scheduler, "CFG Scale": cfg_scale, "Seed": seed, "Width": width, "Height": height}

            return self._process_results(generated, history_data)

        except Exception as e:
            logger.error(f"이미지 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}")
            return [], None


# Instantiate the pipeline for backward compatibility if needed, or just leave the class.
# The user asked to "refactor into a class".
# I will expose the class.
