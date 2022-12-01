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

def create_outfolder(template,models):
    tokens = [v[1] for v in string.Formatter().parse(template) if v[1]]
    default = {}
    for idx, token in enumerate(tokens):
        default[token] = f'T{idx}'
    fn = template.format(**default)
    outpath = Path(batch_config['outpath'])/"".join(i for i in fn if i not in "\/:*?<>|")
    outpath.mkdir(exist_ok=True, parents=True)
    for model in batch_config[models]: Path(outpath/model).mkdir(exist_ok=True)
    return outpath, tokens


for seed in tqdm(batch_config['seeds']):
    torch.manual_seed(seed)
    print(f'Using seed {seed}')
    if 'sd' in batch_config['img-models']:
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16, revision="fp16")
        pipe = pipe.to("cuda")

    for template in batch_config['img-templates']:
        outpath, tokens = create_outfolder(template, 'img-models')
        print(tokens)       
        for idx, itoken in enumerate(tokens):
            if idx + 1 < len(tokens):
                jtoken = tokens[idx+1]
                for tki in tqdm(batch_config[itoken]):
                    for tkj in tqdm(batch_config[jtoken]):
                        pformat = {}
                        pformat[itoken] = tki
                        pformat[jtoken] = tkj
                        prompt = template.format(**pformat)
                        if 'sd' in batch_config['img-models']:
                            image = pipe(prompt).images[0]  
                            image.save(outpath/'sd'/f'sd-{prompt}-{str(seed)}.png')
                        if 'dalle' in batch_config['img-models']:
                            image_resp = openai.Image.create(prompt=prompt, n=1, size="512x512")
                            url = image_resp['data'][0]['url']
                            urllib.request.urlretrieve(url, outpath / 'dalle' /f'dalle-{prompt}-{image_resp["created"]}.png') 

    for template in batch_config['text-templates']:
        outpath, tokens = create_outfolder(template, 'text-models')
        print(tokens)
        for idx, itoken in enumerate(tokens):
            if idx + 1 < len(tokens):
                jtoken = tokens[idx+1]
                for tki in tqdm(batch_config[itoken]):
                    for tkj in tqdm(batch_config[jtoken]):
                        pformat = {}
                        pformat[itoken] = tki
                        pformat[jtoken] = tkj
                        prompt = template.format(**pformat)
                        completion = openai.Completion.create(
                            model="text-davinci-003",
                            prompt=prompt,
                            max_tokens=2000,
                            temperature=0.5,
                            presence_penalty=0.5,
                            frequency_penalty=0.5
                        )
                        with open(outpath / 'gpt3' / "".join(i for i in prompt if i not in "\/:*?<>|"), 'w') as f:
                            f.write(completion.choices[0].text)
                            print(completion)
