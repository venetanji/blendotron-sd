# blendotron-sd

Add `OPENAI_API_KEY='sk-***'` with your api key to a `.env` file in the root of the project or set it as environmental variable.

## Edit batch_config.yaml

Specify output directory and desired seeds.
```
outpath: emotionsets
seeds: [42,892,234,2389]
```

Specify which models to run the batch with:
```
img-models:
  - sd 
  - dalle

text-models:
  - gpt3
```

Specify as many templates as you like. For dalle and sd:
```
img-templates:
  - a {subjects} expressing the emotion {emotions}
  - a {contexts} expressing {emotions}
```
For gpt3:
```
text-templates:
  - 'Write a story about a {contexts} expressing the emotion {emotions}:'
```

Specify two (TODO: generalize to n > 2) lists for each token:
```
contexts:
  - robot
  - person

emotions: 
  - amusement
  - ...
```
