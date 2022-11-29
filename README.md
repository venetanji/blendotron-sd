# blendotron-sd

## Edit batch_config.yaml

Specify output directory and desired seeds.
```
outpath: emotionsets
seeds: [42,892,234,2389]
```

Specify as many templates as you like
```
templates:
  - a {subjects} expressing the emotion {emotions}
  - a {contexts} expressing {emotions}
```

Specify two (TODO: generalize to n > 2) lists for each token:
```
contexts:
  - robot
  - person

emotions: 
  - Amusement
  - ...
```
