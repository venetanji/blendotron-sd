import torch
import yaml
import string
from pathlib import Path
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm
import openai
from dotenv import load_dotenv
import urllib.request
import os
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


batch_config =  yaml.safe_load(open('batch_config.yaml','r'))

for seed in tqdm(batch_config['seeds']):
    torch.manual_seed(seed)
    print(f'Using seed {seed}')
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16, revision="fp16")
    pipe = pipe.to("cuda")

    for template in batch_config['templates']:
        tokens = [v[1] for v in string.Formatter().parse(template)]
        default = {}
        for idx, token in enumerate(tokens):
            default[token] = f'T{idx}'

        outpath = Path(batch_config['outpath'])/template.format(**default)
        outpath.mkdir(exist_ok=True, parents=True)
        for model in batch_config['models']: Path(outpath/model).mkdir(exist_ok=True) 
        
        for idx, itoken in enumerate(tokens):
            if idx + 1 < len(tokens):
                jtoken = tokens[idx+1]
                for tki in tqdm(batch_config[itoken]):
                    for tkj in tqdm(batch_config[jtoken]):
                        pformat = {}
                        pformat[itoken] = tki
                        pformat[jtoken] = tkj
                        prompt = template.format(**pformat)
                        if 'sd' in batch_config['models']:
                            image = pipe(prompt).images[0]  
                            image.save(outpath/'sd'/f'sd-{prompt}-{str(seed)}.png')
                        if 'dalle' in batch_config['models']:
                            image_resp = openai.Image.create(prompt=prompt, n=1, size="512x512")
                            url = image_resp['data'][0]['url']
                            urllib.request.urlretrieve(url, outpath / 'dalle' /f'dalle-{prompt}-{image_resp["created"]}.png') 
    del pipe
