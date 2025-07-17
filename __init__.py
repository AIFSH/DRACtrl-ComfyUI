from .utils import offload
from .utils.offload import profile_type,fast_load_transformers_model

import cv2
import copy
import torch
import os.path as osp
import folder_paths
import comfy.model_management as mm
from huggingface_hub import snapshot_download

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()
weight_dtype = torch.bfloat16
aifsh_dir = osp.join(folder_paths.models_dir,"AIFSH")

vram_optimization_opts = [
    'No_Optimization',
    'HighRAM_HighVRAM',
    'HighRAM_LowVRAM',
    'LowRAM_HighVRAM',
    'LowRAM_LowVRAM',
    'VerylowRAM_LowVRAM'
]
task_opts = ["canny","coloring","deblurring",
            "depth","depth_pred","fill",
            "sr","style_transfer","subject_driven"]
import peft
import diffusers,transformers
from peft import LoraConfig
from safetensors.torch import safe_open
from .models import HunyuanVideoTransformer3DModel
from .pipelines import HunyuanVideoImageToVideoPipeline

class DRACtrlLoraLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "lora_name":(task_opts,),
            }
        }
    
    RETURN_TYPES = ("DRACtrlLora",)
    RETURN_NAMES = ("Lora",)

    FUNCTION = "load_lora"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/DRACtrl"

    def load_lora(self,lora_name):
        lora_dir = osp.join(aifsh_dir,"DRACtrl")
        lora_path = osp.join(lora_dir,f"{lora_name}.safetensors")
        if not osp.exists(lora_path):
            snapshot_download(repo_id="Kunbyte/DRA-Ctrl",local_dir=lora_dir)
        res = dict(lora_name=lora_name,lora_path=lora_path)
        return (res,)

from transformers.models.llava.configuration_llava import LlavaConfig
from .models.llava.modeling_llava import LlavaForConditionalGeneration

class DRACtrlPipeLoader:
    def __init__(self):
        local_path = osp.join(aifsh_dir,"HunyuanVideo-I2V")
        if not osp.exists(osp.join(local_path,"vae/diffusion_pytorch_model.safetensors")):
            snapshot_download(repo_id="hunyuanvideo-community/HunyuanVideo-I2V",local_dir=local_path,ignore_patterns="transformer*")
        self.hyi2v_dir = local_path

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "hy_mmgp_transfomer":(folder_paths.get_filename_list("diffusion_models"),),
                "hy_mmgp_text_encoder":(folder_paths.get_filename_list("text_encoders"),),
                "vram_optimization":(vram_optimization_opts,{
                    "default": 'HighRAM_HighVRAM',
                })
            }
        }
    
    RETURN_TYPES = ("DRACtrlPipe",)
    RETURN_NAMES = ("pipe",)

    FUNCTION = "load_pipe"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/DRACtrl"

    def load_pipe(self,hy_mmgp_transfomer,hy_mmgp_text_encoder,vram_optimization):
        i2v_model_root = self.hyi2v_dir
        '''
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(f'{i2v_model_root}/transformer', 
                                                                 inference_subject_driven=False, 
                                                                 low_cpu_mem_usage=True, 
                                                                 torch_dtype=weight_dtype).requires_grad_(False)
        # offload.save_model(transformer,file_path=osp.join("/root/ComfyUI/models/diffusion_models","hyi2v_transformer_mmgp.safetensors"),do_quantize=True)

        '''
        transformer = fast_load_transformers_model(
            model_path=folder_paths.get_full_path_or_raise("diffusion_models",hy_mmgp_transfomer),
            do_quantize=True,modelClass=HunyuanVideoTransformer3DModel
        )
        
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler()
        vae = diffusers.AutoencoderKLHunyuanVideo.from_pretrained(f'{i2v_model_root}/vae', 
                                                                low_cpu_mem_usage=True, 
                                                                torch_dtype=weight_dtype).requires_grad_(False)
        '''
        # transformers.LlavaForConditionalGeneration
        config = LlavaConfig.from_json_file("/root/ComfyUI/models/AIFSH/HunyuanVideo-I2V/text_encoder/config.json")
        text_encoder = transformers.LlavaForConditionalGeneration._from_config(config,torch_dtype=weight_dtype)
        print("loading text_encoder")
        offload.load_model_data(text_encoder,
                                file_path=folder_paths.get_full_path_or_raise("text_encoders",hy_mmgp_text_encoder),
                                do_quantize=True,verboseLevel=1)
        '''
        text_encoder = LlavaForConditionalGeneration.from_pretrained(f'{i2v_model_root}/text_encoder', 
                                                                                low_cpu_mem_usage=True, 
                                                                                torch_dtype=weight_dtype).requires_grad_(False)
        text_encoder.to(dtype=weight_dtype)
        # offload.save_model(text_encoder,file_path=osp.join("/root/ComfyUI/models/text_encoders","hyi2v_text_encoder_mmgp.safetensors"),do_quantize=True)
        
        text_encoder_2 = transformers.CLIPTextModel.from_pretrained(f'{i2v_model_root}/text_encoder_2', 
                                                                    low_cpu_mem_usage=True, 
                                                                    torch_dtype=weight_dtype).requires_grad_(False)
        tokenizer = transformers.AutoTokenizer.from_pretrained(f'{i2v_model_root}/tokenizer')
        tokenizer_2 = transformers.CLIPTokenizer.from_pretrained(f'{i2v_model_root}/tokenizer_2')
        image_processor = transformers.CLIPImageProcessor.from_pretrained(f'{i2v_model_root}/image_processor')

        vae.enable_tiling()
        vae.enable_slicing()
        # insert LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            init_lora_weights="gaussian",
            target_modules=[
                'attn.to_k', 'attn.to_q', 'attn.to_v', 'attn.to_out.0',
                'attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj', 'attn.to_add_out',
                'ff.net.0.proj', 'ff.net.2',
                'ff_context.net.0.proj', 'ff_context.net.2',
                'norm1_context.linear', 'norm1.linear',
                'norm.linear', 'proj_mlp', 'proj_out',
            ]
        )
        transformer.add_adapter(lora_config)

        # hack LoRA forward
        def create_hacked_forward(module):
            if not hasattr(module, 'original_forward'):
                module.original_forward = module.forward
            img_sequence_length = int((512 / 8 / 2) ** 2)
            encoder_sequence_length = 144 + 252 # encoder sequence: 144 img 252 txt
            num_imgs = 4
            num_generated_imgs = 3

            def hacked_lora_forward(self, x, *args, **kwargs):
                lora_forward = self.original_forward

                if x.shape[1] == img_sequence_length * num_imgs and len(x.shape) > 2:
                    return torch.cat((
                        lora_forward(x[:, :-img_sequence_length*num_generated_imgs], *args, **kwargs),
                        self.base_layer(x[:, -img_sequence_length*num_generated_imgs:], *args, **kwargs)
                    ), dim=1)
                elif x.shape[1] == encoder_sequence_length * 2 or x.shape[1] == encoder_sequence_length:
                    return lora_forward(x, *args, **kwargs)
                elif x.shape[1] == img_sequence_length * num_imgs + encoder_sequence_length:
                    return torch.cat((
                        lora_forward(x[:, :(num_imgs - num_generated_imgs)*img_sequence_length], *args, **kwargs),
                        self.base_layer(x[:, (num_imgs - num_generated_imgs)*img_sequence_length:-encoder_sequence_length], *args, **kwargs),
                        lora_forward(x[:, -encoder_sequence_length:], *args, **kwargs)
                    ), dim=1)
                elif x.shape[1] == img_sequence_length * num_imgs + encoder_sequence_length * 2:
                    return torch.cat((
                        lora_forward(x[:, :(num_imgs - num_generated_imgs)*img_sequence_length], *args, **kwargs),
                        self.base_layer(x[:, (num_imgs - num_generated_imgs)*img_sequence_length:-2*encoder_sequence_length], *args, **kwargs),
                        lora_forward(x[:, -2*encoder_sequence_length:], *args, **kwargs)
                    ), dim=1)
                elif x.shape[1] == 3072:
                    return self.base_layer(x, *args, **kwargs)
                else:
                    raise ValueError(
                        f"hacked_lora_forward receives unexpected sequence length: {x.shape[1]}, input shape: {x.shape}!"
                    )

            return hacked_lora_forward.__get__(module, type(module))

        for n, m in transformer.named_modules():
            if isinstance(m, peft.tuners.lora.layer.Linear):
                m.forward = create_hacked_forward(m)

        pipe = HunyuanVideoImageToVideoPipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            vae=vae,
            scheduler=copy.deepcopy(scheduler),
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            image_processor=image_processor,
        )
        if vram_optimization == 'No_Optimization':
            pipe.to(device)
        else:
            [
            'No_Optimization',
            'HighRAM_HighVRAM',
            'HighRAM_LowVRAM',
            'LowRAM_HighVRAM',
            'LowRAM_LowVRAM',
            'VerylowRAM_LowVRAM'
        ]
            if vram_optimization == 'HighRAM_HighVRAM':
                optimization_type = profile_type.HighRAM_HighVRAM
            elif vram_optimization == 'HighRAM_HighVRAM':
                optimization_type = profile_type.HighRAM_HighVRAM
            elif vram_optimization == 'HighRAM_LowVRAM':
                optimization_type = profile_type.HighRAM_LowVRAM
            elif vram_optimization == 'LowRAM_HighVRAM':
                optimization_type = profile_type.LowRAM_HighVRAM
            elif vram_optimization == 'LowRAM_LowVRAM':
                optimization_type = profile_type.LowRAM_LowVRAM
            elif vram_optimization == 'VerylowRAM_LowVRAM':
                optimization_type = profile_type.VerylowRAM_LowVRAM
            offload.profile(pipe, optimization_type)
        return (pipe,)

import numpy as np
from PIL import Image,ImageFilter,ImageDraw
def comfy2pil(image):
        i = 255. * image.cpu().numpy()[0]
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img
    
def pil2comfy(pil):
    image = pil.convert("RGB")
    image = np.array(pil).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image

class DRACtrlGetCImage:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "pipe":("DRACtrlPipe",),
                "lora":("DRACtrlLora",),
                "ref_image":("IMAGE",),
                "width":("INT",{
                    "default":512
                }),
                "height":("INT",{
                    "default":512
                }),
            },
            "optional":{
                "inpainting": ("BOOLEAN",),
                "fill_x1": ("INT",),
                "fill_x2": ("INT",),
                "fill_y1": ("INT",),
                "fill_y2": ("INT",),
            }
        }
    
    RETURN_TYPES = ("DRACtrlPipe","IMAGE",)
    RETURN_NAMES = ("pipe","image",)

    FUNCTION = "get_c_img"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/DRACtrl"

    def get_c_img(self,pipe,lora,ref_image,width,height,
                  inpainting,fill_x1,fill_x2,fill_y1,fill_y2):

        ## load lora
        try:
            with safe_open(lora['lora_path'], framework="pt") as f:
                lora_weights = {}
                for k in f.keys():
                    param = f.get_tensor(k) 
                    if k.endswith(".weight"):
                        k = k.replace('.weight', '.default.weight')
                    lora_weights[k] = param
                pipe.transformer.load_state_dict(lora_weights, strict=False)
        except Exception as e:
            raise ValueError(f'{e}')

        pipe.transformer.requires_grad_(False)

        c_img = comfy2pil(ref_image)
        c_img = c_img.resize((width, height))
        task = lora['lora_name']
        pipe.task = task
        if task not in ['subject_driven', 'style_transfer']:
            if task == "canny":
                def get_canny_edge(img):
                    img_np = np.array(img)
                    img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(img_gray, 100, 200)
                    edges_tmp = Image.fromarray(edges).convert("RGB")
                    edges[edges == 0] = 128
                    return Image.fromarray(edges).convert("RGB")
                c_img = get_canny_edge(c_img)
            elif task == "coloring":
                c_img = (
                    c_img.resize((width, height))
                    .convert("L")
                    .convert("RGB")
                )
            elif task == "deblurring":
                '''
                blur_radius = 10
                c_img = (
                    c_img.convert("RGB")
                    .filter(ImageFilter.GaussianBlur(blur_radius))
                    .resize((width, height))
                    .convert("RGB")
                )
                '''
                c_img = c_img.resize((width, height))
            elif task == "depth":
                def get_depth_map(img):
                    from transformers import pipeline

                    depth_pipe = pipeline(
                        task="depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Small-hf",
                    )
                    return depth_pipe(img)["depth"].convert("RGB").resize((width, height))
                c_img = get_depth_map(c_img)
                k = (255 - 128) / 255
                b = 128
                c_img = c_img.point(lambda x: k * x + b)
            elif task == "depth_pred":
                c_img = c_img
            elif task == "fill":
                c_img = c_img.resize((width, height)).convert("RGB")
                x1, x2 = fill_x1, fill_x2
                y1, y2 = fill_y1, fill_y2
                mask = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(mask)
                draw.rectangle((x1, y1, x2, y2), fill=255)
                if inpainting:
                    mask = Image.eval(mask, lambda a: 255 - a)
                c_img = Image.composite(
                    c_img,
                    Image.new("RGB", (width, height), (255, 255, 255)),
                    mask
                )
                c_img = Image.composite(
                    c_img,
                    Image.new("RGB", (width, height), (128, 128, 128)),
                    mask
                )
            elif task == "sr":
                # c_img = c_img.resize((int(width / 4), int(height / 4))).convert("RGB")
                c_img = c_img.resize((width, height))
        return (pipe, pil2comfy(c_img),)

class DRACtrlSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "pipe":("DRACtrlPipe",),
                "condition_image":("IMAGE",),
                "target_prompt":("STRING",),
                "num_steps":("INT",{
                    "default": 50,
                }),
                "seed":("INT",{
                    "default": 42,
                }),
            },
            "optional":{
                "condition_image_prompt":("STRING",)
            }
        }
    
    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("image","images",)

    FUNCTION = "sample"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/DRACtrl"

    def sample(self,pipe,condition_image,target_prompt,
               num_steps,seed,condition_image_prompt=None):
        c_txt = condition_image_prompt
        t_txt = target_prompt
        c_img = comfy2pil(condition_image)
        width, height = c_img.size
        task = pipe.task

        gen_img = pipe(
            image=c_img,
            prompt=[t_txt.strip()],
            prompt_condition=[c_txt.strip()] if c_txt is not None else None,
            prompt_2=[t_txt],
            height=width,
            width=height,
            num_frames=5,
            num_inference_steps=num_steps,
            guidance_scale=6.0,
            num_videos_per_prompt=1,
            generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed),
            output_type='pt',
            image_embed_interleave=4,
            frame_gap=48,
            mixup=True,
            mixup_num_imgs=2,
            enhance_tp=task in ['subject_driven'],
        ).frames
        output_images = []
        for i in range(10):
            out = gen_img[:, i:i+1, :, :, :]
            out = out.squeeze(0).squeeze(0).cpu().to(torch.float32).numpy()
            out = np.transpose(out, (1, 2, 0))
            out = (out * 255).astype(np.uint8)
            out = Image.fromarray(out)
            output_images.append(out)
        image = pil2comfy(output_images[0])
        images = []
        for i_image in output_images[1:]+[output_images[0]]:
            images.append(pil2comfy(i_image))
        images = torch.concat(images)
        return (image,images,)

NODE_CLASS_MAPPINGS = {
    "DRACtrlPipeLoader": DRACtrlPipeLoader,
    "DRACtrlLoraLoader":DRACtrlLoraLoader,
    "DRACtrlSampler":DRACtrlSampler,
    "DRACtrlGetCImage":DRACtrlGetCImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DRACtrlPipeLoader": "PipeLoader@关注超级面爸微信公众号",
    "DRACtrlSampler":"Sampler@关注超级面爸微信公众号",
    "DRACtrlLoraLoader":"LoraLoader@关注超级面爸微信公众号",
    "DRACtrlGetCImage":"GetCImage@关注超级面爸微信公众号",
}