import torch
import yaml
import string
from pathlib import Path
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm


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
        outpath = Path(batch_config['outpath'])/template.format(**default) / str(seed)
        print(outpath)
        outpath.mkdir(exist_ok=True, parents=True)
        
        for idx, itoken in enumerate(tokens):
            if idx + 1 < len(tokens):
                jtoken = tokens[idx+1]
                for tki in tqdm(batch_config[itoken]):
                    for tkj in tqdm(batch_config[jtoken]):
                        pformat = {}
                        pformat[itoken] = tki
                        pformat[jtoken] = tkj
                        prompt = template.format(**pformat)
                        image = pipe(prompt).images[0]  
                        image.save(outpath/f'{prompt}.png')
    del pipe
