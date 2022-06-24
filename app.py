import os
from urllib.parse import urlparse

from PIL import Image
import requests
import torch
from timm.models.hub import download_cached_file
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import gradio as gr
from mm_commerce import BLIP_Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    return checkpoint


image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

model = BLIP_Decoder(med_config='configs/med_large_config.json', vit='large_v2', prompt='[DEC]')

ckpt = 'https://huggingface.co/zhezh/mm_commerce_zhcn/resolve/main/model.pth'
sd = load_checkpoint(ckpt)
model.load_state_dict(sd, strict=True)

model.eval()
model = model.to('cuda')


def inference(raw_image, strategy):
    image = transform(raw_image).unsqueeze(0).to(device)
    with torch.no_grad():
        if strategy == "Beam search":
          caption = model.generate(image, sample=False, num_beams=10, max_length=100, min_length=10)
        else:
          caption = model.generate(image, sample=True, top_p=0.9, max_length=100, min_length=10)
        return '商品描述: ' + '"' + ''.join(caption[0][6:-5].split()) + '"'

    
inputs = [
    gr.inputs.Image(type='pil'),
    gr.inputs.Radio(choices=['Beam search','Nucleus sampling'], type="value", default="Beam search", label="文本生成策略")
]
outputs = gr.outputs.Textbox(label="生成的标题(Output)")

title = "MM Commerce ZhCN (中文商品描述生成)"

description = "中文商品描述生成 -- By Zhe Zhang"

demo = gr.Interface(
    inference, inputs, outputs, title=title, description=description,
    # article=article,
    examples=[
        ['starrynight.jpeg', "Nucleus sampling"],
        ['resources/examples/zhuobu.jpg', "Beam search"],
        ['resources/examples/jiandao.jpg', "Beam search"],
        ['resources/examples/lego-yellow.jpg', "Beam search"],
        ['resources/examples/charger.jpg', "Beam search"],
        ['resources/examples/charger-ugreen.jpg', "Beam search"],
        ['resources/examples/charger-hw.jpg', "Beam search"],
    ],
)
# demo.launch(enable_queue=True, share=True, server_name='0.0.0.0', server_port=8080,)
demo.launch(enable_queue=True)
