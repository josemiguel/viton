from io import BytesIO
from pathlib import Path

import io
from pathlib import Path

from modal import (
    App,
    Image,
    Mount,
    asgi_app,
    build,
    enter,
    gpu,
    method,
    web_endpoint,
)

app = App("idm-viton")

viton_image = (
    Image
    .debian_slim()
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "transformers==4.36.2",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "torchaudio==2.0.2",
        "numpy==1.24.4",
        "scipy==1.10.1",
        "scikit-image==0.21.0",
        "opencv-python==4.7.0.72",
        "pillow==9.4.0",
        "diffusers==0.25.0",
        "transformers==4.36.2",
        "accelerate==0.26.1",
        "matplotlib==3.7.4",
        "tqdm==4.64.1",
        "config==0.5.1",
        "einops==0.7.0",
        "onnxruntime==1.16.2",
        "basicsr",
        "av",
        "fvcore",
        "cloudpickle",
        "omegaconf",
        "pycocotools",
        "boto3"
    )
    .copy_local_dir('.', '/root/')
  )


with viton_image.imports():
    from fastapi import Response

    import io
    import torch
    import requests
    import apply_net

    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.unet_hacked_tryon import UNet2DConditionModel

    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

    from PIL import Image
    from transformers import AutoTokenizer
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
        CLIPTextModel,
        CLIPTextModelWithProjection,
    )
    from diffusers import DDPMScheduler, AutoencoderKL
    from typing import List
    from utils_mask import get_mask_location
    from torchvision import transforms
    from torchvision.transforms.functional import to_pil_image

@app.cls(gpu="A10G", container_idle_timeout=240, image=viton_image)
class Model:

    @build()
    @enter()
    def enter(self):
        base_path = 'yisol/IDM-VTON'

        unet = UNet2DConditionModel.from_pretrained(
            base_path,
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        unet.requires_grad_(False)
        tokenizer_one = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )
        noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

        text_encoder_one = CLIPTextModel.from_pretrained(
            base_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            base_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_path,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
            )
        vae = AutoencoderKL.from_pretrained(base_path,
                                            subfolder="vae",
                                            torch_dtype=torch.float16,
        )

        # "stabilityai/stable-diffusion-xl-base-1.0",
        UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            base_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
        )

        self.parsing_model = Parsing(0)
        self.openpose_model = OpenPose(0)

        UNet_Encoder.requires_grad_(False)
        image_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        self.tensor_transfrom = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
            )

        self.pipe = TryonPipeline.from_pretrained(
                base_path,
                unet=unet,
                vae=vae,
                feature_extractor= CLIPImageProcessor(),
                text_encoder = text_encoder_one,
                text_encoder_2 = text_encoder_two,
                tokenizer = tokenizer_one,
                tokenizer_2 = tokenizer_two,
                scheduler = noise_scheduler,
                image_encoder=image_encoder,
                torch_dtype=torch.float16,
        )
        self.pipe.unet_encoder = UNet_Encoder

    def _inference(self, user_id, human_img_url, garm_img_url, garment_des, denoise_steps=24, seed=42):
        device = "cuda"

        def url_to_pil(url):
            req = requests.get(url)
            return Image.open(io.BytesIO(req.content))
        
        self.openpose_model.preprocessor.body_estimation.model.to(device)
        self.pipe.to(device)
        self.pipe.unet_encoder.to(device)

        garm_img = url_to_pil(garm_img_url)
        human_img = url_to_pil(human_img_url)

        garm_img = garm_img.convert("RGB").resize((768, 1024))
        human_img = human_img.convert("RGB").resize((768, 1024))
        
        keypoints = self.openpose_model(human_img.resize((384,512)))
        model_parse, _ = self.parsing_model(human_img.resize((384,512)))

        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768,1024))

        mask_gray = (1 - transforms.ToTensor()(mask)) * self.tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

        human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        
        params = ('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', 
                           './ckpt/densepose/model_final_162be9.pkl', 
                  'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda')

        args = apply_net.create_argument_parser().parse_args(params)

        pose_img = args.func(args, human_img_arg)    
        pose_img = pose_img[:,:,::-1]    
        pose_img = Image.fromarray(pose_img).resize((768,1024))
        
        with torch.no_grad():
            # Extract the images
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    prompt = "model is wearing " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    with torch.inference_mode():
                        (
                            prompt_embeds,
                            negative_prompt_embeds,
                            pooled_prompt_embeds,
                            negative_pooled_prompt_embeds,
                        ) = self.pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=True,
                            negative_prompt=negative_prompt,
                        )
                                        
                        prompt = "a photo of " + garment_des
                        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                        if not isinstance(prompt, List):
                            prompt = [prompt] * 1
                        if not isinstance(negative_prompt, List):
                            negative_prompt = [negative_prompt] * 1
                        with torch.inference_mode():
                            (
                                prompt_embeds_c,
                                _,
                                _,
                                _,
                            ) = self.pipe.encode_prompt(
                                prompt,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False,
                                negative_prompt=negative_prompt,
                            )

                        pose_img =  self.tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                        garm_tensor =  self.tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                        generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                        images = self.pipe(
                            prompt_embeds=prompt_embeds.to(device,torch.float16),
                            negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                            num_inference_steps=denoise_steps,
                            generator=generator,
                            strength = 1.0,
                            pose_img = pose_img.to(device,torch.float16),
                            text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                            cloth = garm_tensor.to(device,torch.float16),
                            mask_image=mask,
                            image=human_img, 
                            height=1024,
                            width=768,
                            ip_adapter_image = garm_img.resize((768,1024)),
                            guidance_scale=2.0,
                        )[0]

        img = images[0]

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='jpeg')

        return img_byte_arr

    @method()
    def inference(self, user_id, human_img_url, garm_img_url, prompt, denoise_steps=20, seed=42):
        return self._inference(
                user_id, human_img_url, garm_img_url, prompt, denoise_steps, seed
        ).getvalue()

    @web_endpoint(docs=True)
    def web_inference(
            self, user_id: str, human_img_url: str, garm_img_url: str, prompt: str, denoise_steps: int = 20, seed: int = 42):
        return Response(
            content=self._inference(
                user_id, human_img_url, garm_img_url, prompt, denoise_steps, seed
            ).getvalue(),
            media_type="image/jpeg",
        )

@app.local_entrypoint()
def main(prompt: str = ""):
    image_bytes = Model().inference.remote("123", 
                "https://onholy.com/model.jpg", 
                "https://static.zara.net/assets/public/4717/1ee8/f1f34efca038/8ca426e91f0d/08372290043-e1/08372290043-e1.jpg?ts=1710318244848&w=850", prompt)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)
