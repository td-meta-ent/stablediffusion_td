import os
import cv2
import torch
import numpy as np
import PIL
import streamlit as st
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

torch.set_grad_enabled(False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

PROJECT_PATH = "/home/td/Project/stablediffusion"
CONFIG_PATH = os.path.join(PROJECT_PATH,"configs/stable-diffusion/v2-inference-v.yaml")
CHECKPOINT_PATH = os.path.join(PROJECT_PATH,"weight/768-v-ema.ckpt")

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    # model.eval()
    return model

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

@st.cache(allow_output_mutation=True)
def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)
    return model, sampler

def inference(model, sampler, prompt, image, steps, ddim_eta, seed, n_samples, scale, t_enc, callback=None, do_full_sample=False):
    
    seed_everything(seed)
    n_iter = 3 
    ddim_eta= 0.0
    batch_size = n_samples
    n_rows = batch_size
    data = [batch_size * [prompt]]
        
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    
    # assert os.path.isfile(image_path)
    init_image = image.to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False)

    precision_scope = autocast 
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                output = []
                # for n in trange(n_samples, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)

                    # encode (scaled latent)
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
                    # decode it
                    samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, callback = callback)

                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img = put_watermark(img, wm_encoder)
                        img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1
                    all_samples.append(x_samples)

                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid = put_watermark(grid, wm_encoder)
                output.append(grid)
                #grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                grid_count += 1

    return output


st.title("Stable Diffusion Image2Image")
st.markdown("""
    ## If size of input image is bigger than (1024,1024), the image will resize to (1024,1024)
""")
video_file = open('/home/td/Project/stablediffusion/scripts/samples/image2image.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

model, sampler = initialize_model(CONFIG_PATH, CHECKPOINT_PATH)

outpath = "outputs/img2img-samples"
os.makedirs(outpath, exist_ok=True)

wm = "SDV2"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

image_path=None
image_path = st.file_uploader("Image", ["jpg", "png"])
if image_path:
    image = Image.open(image_path)
    def image_convert(image):
        image = image.convert("RGB")
        w, h = image.size
        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        w = min(w,1024)
        h = min(h,1024)
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2. * image - 1.
    img_for_running = image_convert(image)
    w, h = image.size
    st.text(f"loaded input image of size ({w}, {h})")
    width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    width = min(width,1024)
    height = min(height,1024)
    image = image.resize((width, height))
    st.text(f"resized input image to size ({width}, {height} (w, h))")
    st.image(image)


    prompt = st.text_input("Prompt")

    seed = st.number_input("Seed", min_value=0, max_value=1000000, value=0)
    # n_samples = st.number_input("Number of Samples", min_value=1, max_value=64, value=1)
    scale = st.slider("Scale", min_value=0.1, max_value=30.0, value=9.0, step=0.1)
    # steps = st.slider("DDIM Steps", min_value=0, max_value=50, value=50, step=1)
    # ddim_eta = st.slider("DDIM eta", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    # strength = st.slider("Strength", min_value=0., max_value=1., value=0.9)

    n_samples = 2
    # scale = 9.0
    steps = 50
    ddim_eta = 0.0
    strength = 0.9

    t_progress = st.progress(0)
    def t_callback(t):
        t_progress.progress(min((t + 1) / t_enc, 1.))

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    do_full_sample = strength == 1.
    t_enc = int(strength * steps)

    if st.button("Sample"):
        if prompt != None :
            result = inference(
                model = model,
                sampler=sampler,
                prompt=prompt,
                image = img_for_running,
                steps=steps,
                ddim_eta = ddim_eta,
                seed=seed,
                scale=scale,
                t_enc=t_enc,
                n_samples=n_samples,
                callback=t_callback,
                do_full_sample=do_full_sample,
            )
            st.write("Result")
            for image in result:
                st.image(image, output_format='PNG')


    