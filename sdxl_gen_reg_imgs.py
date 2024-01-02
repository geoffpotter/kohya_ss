# This script will create the caption text files in the specified folder using the specified file pattern and caption text.
#
# eg: python caption.py D:\some\folder\location "*.png, *.jpg, *.webp" "some caption text"

import argparse
import random
# import glob
from pathlib import Path
from typing import Any
from PIL import Image
from sdxl_gen_img import main as GenImages, setup_parser as setup_gen_parser


import datetime
import math
import os
from einops import repeat
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer
from diffusers import EulerDiscreteScheduler
from PIL import Image
import open_clip
from safetensors.torch import load_file

from library import model_util, sdxl_model_util
import networks.lora as lora

# scheduler: このあたりの設定はSD1/2と同じでいいらしい
# scheduler: The settings around here seem to be the same as SD1/2
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

# Time EmbeddingはDiffusersからのコピー
# Time Embedding is copied from Diffusers


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=timesteps.device
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


def get_timestep_embedding(x, outdim):
    assert len(x.shape) == 2
    b, dims = x.shape[0], x.shape[1]
    # x = rearrange(x, "b d -> (b d)")
    x = torch.flatten(x)
    emb = timestep_embedding(x, outdim)
    # emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=outdim)
    emb = torch.reshape(emb, (b, dims * outdim))
    return emb


def getMakeImage(ckpt_path, steps = 50, guidance_scale = 7, crop_top = 0, crop_left = 0):
    # 画像生成条件を変更する場合はここを変更 / change here to change image generation conditions

    # SDXLの追加のvector embeddingへ渡す値 / Values to pass to additional vector embedding of SDXL
    DEVICE = "cuda"
    DTYPE = torch.float16  # bfloat16 may work




    # HuggingFaceのmodel id
    text_encoder_1_name = "openai/clip-vit-large-patch14"
    text_encoder_2_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

    # checkpointを読み込む。モデル変換についてはそちらの関数を参照
    # Load checkpoint. For model conversion, see this function

    # 本体RAMが少ない場合はGPUにロードするといいかも
    # If the main RAM is small, it may be better to load it on the GPU
    text_model1, text_model2, vae, unet, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
        sdxl_model_util.MODEL_VERSION_SDXL_BASE_V0_9, ckpt_path, "cpu"
    )

    # Text Encoder 1はSDXL本体でもHuggingFaceのものを使っている
    # In SDXL, Text Encoder 1 is also using HuggingFace's

    # Text Encoder 2はSDXL本体ではopen_clipを使っている
    # それを使ってもいいが、SD2のDiffusers版に合わせる形で、HuggingFaceのものを使う
    # 重みの変換コードはSD2とほぼ同じ
    # In SDXL, Text Encoder 2 is using open_clip
    # It's okay to use it, but to match the Diffusers version of SD2, use HuggingFace's
    # The weight conversion code is almost the same as SD2

    # VAEの構造はSDXLもSD1/2と同じだが、重みは異なるようだ。何より謎のscale値が違う
    # fp16でNaNが出やすいようだ
    # The structure of VAE is the same as SD1/2, but the weights seem to be different. Above all, the mysterious scale value is different.
    # NaN seems to be more likely to occur in fp16

    unet.to(DEVICE, dtype=DTYPE)
    unet.eval()

    vae_dtype = DTYPE
    if DTYPE == torch.float16:
        print("use float32 for vae")
        vae_dtype = torch.float32
    vae.to(DEVICE, dtype=vae_dtype)
    vae.eval()

    text_model1.to(DEVICE, dtype=DTYPE)
    text_model1.eval()
    text_model2.to(DEVICE, dtype=DTYPE)
    text_model2.eval()

    unet.set_use_memory_efficient_attention(True, False)
    if torch.__version__ >= "2.0.0": # PyTorch 2.0.0 以上対応のxformersなら以下が使える
        vae.set_use_memory_efficient_attention_xformers(True)

    # Tokenizers
    tokenizer1 = CLIPTokenizer.from_pretrained(text_encoder_1_name)
    tokenizer2 = lambda x: open_clip.tokenize(x, context_length=77)

    # # LoRA
    # for weights_file in args.lora_weights:
    #     if ";" in weights_file:
    #         weights_file, multiplier = weights_file.split(";")
    #         multiplier = float(multiplier)
    #     else:
    #         multiplier = 1.0

    #     lora_model, weights_sd = lora.create_network_from_weights(
    #         multiplier, weights_file, vae, [text_model1, text_model2], unet, None, True
    #     )
    #     lora_model.merge_to([text_model1, text_model2], unet, weights_sd, DTYPE, DEVICE)

    # scheduler
    scheduler = EulerDiscreteScheduler(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
    )

    def generate_image(output_dir, image_name, prompt, prompt2, negative_prompt, target_width, target_height, original_width=None, original_height=None, seed=None):

        if original_height is None:
            original_height = target_height
        if original_width is None:
            original_width = target_width

        if prompt2 is None:
            prompt2 = prompt
        
        if negative_prompt is None:
            negative_prompt = ""
        # 将来的にサイズ情報も変えられるようにする / Make it possible to change the size information in the future
        # prepare embedding
        with torch.no_grad():
            # vector
            emb1 = get_timestep_embedding(torch.FloatTensor([original_height, original_width]).unsqueeze(0), 256)
            emb2 = get_timestep_embedding(torch.FloatTensor([crop_top, crop_left]).unsqueeze(0), 256)
            emb3 = get_timestep_embedding(torch.FloatTensor([target_height, target_width]).unsqueeze(0), 256)
            # print("emb1", emb1.shape)
            c_vector = torch.cat([emb1, emb2, emb3], dim=1).to(DEVICE, dtype=DTYPE)
            uc_vector = c_vector.clone().to(DEVICE, dtype=DTYPE)  # ちょっとここ正しいかどうかわからない I'm not sure if this is right

            # crossattn

        # Text Encoderを二つ呼ぶ関数  Function to call two Text Encoders
        def call_text_encoder(text, text2):
            # text encoder 1
            batch_encoding = tokenizer1(
                text,
                truncation=True,
                return_length=True,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            tokens = batch_encoding["input_ids"].to(DEVICE)

            with torch.no_grad():
                enc_out = text_model1(tokens, output_hidden_states=True, return_dict=True)
                text_embedding1 = enc_out["hidden_states"][11]
                # text_embedding = pipe.text_encoder.text_model.final_layer_norm(text_embedding)    # layer normは通さないらしい

            # text encoder 2
            with torch.no_grad():
                tokens = tokenizer2(text2).to(DEVICE)

                enc_out = text_model2(tokens, output_hidden_states=True, return_dict=True)
                text_embedding2_penu = enc_out["hidden_states"][-2]
                # print("hidden_states2", text_embedding2_penu.shape)
                text_embedding2_pool = enc_out["text_embeds"]

            # 連結して終了 concat and finish
            text_embedding = torch.cat([text_embedding1, text_embedding2_penu], dim=2)
            return text_embedding, text_embedding2_pool

        # cond
        c_ctx, c_ctx_pool = call_text_encoder(prompt, prompt2)
        # print(c_ctx.shape, c_ctx_p.shape, c_vector.shape)
        c_vector = torch.cat([c_ctx_pool, c_vector], dim=1)

        # uncond
        uc_ctx, uc_ctx_pool = call_text_encoder(negative_prompt, negative_prompt)
        uc_vector = torch.cat([uc_ctx_pool, uc_vector], dim=1)

        text_embeddings = torch.cat([uc_ctx, c_ctx])
        vector_embeddings = torch.cat([uc_vector, c_vector])

        # メモリ使用量を減らすにはここでText Encoderを削除するかCPUへ移動する

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # # random generator for initial noise
            # generator = torch.Generator(device="cuda").manual_seed(seed)
            generator = None
        else:
            generator = None

        # get the initial random noise unless the user supplied it
        # SDXLはCPUでlatentsを作成しているので一応合わせておく、Diffusersはtarget deviceでlatentsを作成している
        # SDXL creates latents in CPU, Diffusers creates latents in target device
        latents_shape = (1, 4, target_height // 8, target_width // 8)
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(DEVICE, dtype=DTYPE)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma

        # set timesteps
        scheduler.set_timesteps(steps, DEVICE)

        # このへんはDiffusersからのコピペ
        # Copy from Diffusers
        timesteps = scheduler.timesteps.to(DEVICE)  # .to(DTYPE)
        num_latent_input = 2
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                noise_pred = unet(latent_model_input, t, text_embeddings, vector_embeddings)

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # latents = 1 / 0.18215 * latents
            latents = 1 / sdxl_model_util.VAE_SCALE_FACTOR * latents
            latents = latents.to(vae_dtype)
            image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # image = self.numpy_to_pil(image)
        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]

        # 保存して終了 save and finish
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        saved_paths = []
        for i, img in enumerate(image):
            if image_name is None:
                image_name = f"image_{timestamp}_{i:03d}.png"
            image_path = os.path.join(output_dir, image_name)
            img.save(image_path)
            saved_paths.append(image_path)

        return saved_paths



    return generate_image


def create_caption_files(ckpt, base_folder: str, file_pattern: str, caption_file_ext: str, tag_file_ext: str, overwrite: bool):
    # Split the file patterns string and strip whitespace from each pattern
    patterns = [pattern.strip() for pattern in file_pattern.split(",")]

    # Create a Path object for the image folder
    folder = Path(base_folder)

    makeImages = getMakeImage("E:\stuff\stable-diffusion-webui\models\Stable-diffusion\sd_xl_base_1.0.safetensors", 50)
    # Iterate over the file patterns
    for pattern in patterns:
        # Use the glob method to match the file patterns
        files = folder.glob(pattern)

        # Iterate over the matched files
        for idx, file in enumerate(files):
            if file.parent.name.endswith("_reg"):
                continue
            # Check if a text file with the same name as the current file exists in the folder
            # Open the image file
            cap_file = file.with_suffix(caption_file_ext)
            tag_file = file.with_suffix(tag_file_ext)
            img = Image.open(file)
            caps = []
            if cap_file.exists():
                caps = cap_file.read_text(encoding='utf-8').split('\n') 
            if len(caps) == 1: # if there's only one line, assume the captions are split by commas
                caps = caps[0].split(",")
            
            tags = []
            if tag_file.exists():
                tags = tag_file.read_text(encoding='utf-8').split(",")
            reg_dir = Path(file.parent.as_posix() + "_reg")

            if not reg_dir.is_dir():
                print("making reg directory:", reg_dir)
                reg_dir.mkdir()
            

            width = img.width
            height = img.height
            # width = 1208
            # height = 888
            # print(width, height)
            aspect = float(width) / float(height)
            # print(aspect)
            if aspect > 1: #wide
                width = 1024
                height = int(1024 * (1 / aspect))
            else: #tall
                width = int(1024 * aspect)
                height = 1024

            if height % 64 != 0:
                height -= height % 64
            if width % 64 != 0:
                width -= width % 64
            
            base_file_name = file.with_suffix("").name
            # print(width, height)
            # return
            for itr in range(5):
                fileName = base_file_name + "_reg_" + str(itr) + ".png"
                if Path(reg_dir, fileName).exists():
                    print("reg file exists, skipping", Path(reg_dir, fileName))
                    continue
                ourTags = random.choices(tags, k=5)
                ourCap = random.choice(caps)
                prompt = ourCap + ", " + ",".join(ourTags)
                created_paths = makeImages(reg_dir, fileName, prompt, prompt, "illustration, drawing, painting", width, height)
                for created_path in created_paths:
                    this_cap_file = Path(created_path).with_suffix(caption_file_ext)
                    this_cap_file.write_text(prompt)

            
            

            # if not txt_file.exists() or overwrite:
            #     # Create a text file with the caption text in the folder, if it does not already exist
            #     # or if the overwrite argument is True
            #     with open(txt_file, "w") as f:
            #         f.write(caption_text)

def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str, help="the folder where the image files are located")
    parser.add_argument("--file_pattern", type=str, default="**/*.png, **/*.jpg, **/*.jpeg, **/*.webp", help="the pattern to match the image file names")
    parser.add_argument("--caption_file_ext", type=str, default=".captions", help="the caption file extension.")
    parser.add_argument("--tag_file_ext", type=str, default=".tags", help="the caption file extension.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="whether to overwrite existing caption files")

    # Create a mutually exclusive group for the caption_text and caption_file arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--caption_text", type=str, help="the text to include in the caption files")
    group.add_argument("--caption_file", type=argparse.FileType("r"), help="the file containing the text to include in the caption files")
    
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint of model / モデルのcheckpointファイルまたはディレクトリ")
    

    # Parse the command-line arguments
    args = parser.parse_args()
    base_folder = args.base_folder
    file_pattern = args.file_pattern
    caption_file_ext = args.caption_file_ext
    tag_file_ext = args.tag_file_ext
    overwrite = args.overwrite
    ckpt = args.ckpt

    # Get the caption text from either the caption_text or caption_file argument
    if args.caption_text:
        caption_text = args.caption_text
    elif args.caption_file:
        caption_text = args.caption_file.read()
    else:
        caption_text = ""

    # Create a Path object for the image folder
    folder = Path(base_folder)

    # Check if the image folder exists and is a directory
    if not folder.is_dir():
        raise ValueError(f"{base_folder} is not a valid directory.")
        
    # Create the caption files
    create_caption_files(ckpt, base_folder, file_pattern, caption_file_ext, tag_file_ext, overwrite)

# python sdxl_gen_reg_imgs.py --ckpt "E:\stuff\stable-diffusion-webui\models\Stable-diffusion\sd_xl_base_1.0.safetensors" --base_folder "E:\stuff\New folder\datasets\geoffp"


# "python sdxl_minimal_inference.py --prompt "a man" --prompt2 "suit" --negative_prompt "drawing" --ckpt_path E:\stuff\stable-diffusion-webui\models\Stable-diffusion\sd_xl_base_1.0.safetensors"

if __name__ == "__main__":
    main()



