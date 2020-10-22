# cornel_birdcall_solution

<p>This is the repository to show my solution of <a href="https://www.kaggle.com/c/birdsong-recognition">cornel birdcall identification</a> at kaggle.</p>
<p>I was 117th/1390, and got bronze medal(🥉).</p>

<h3>How to run this code?</h3>
<p>I didn't upload data to this repo, for training was too heavy...</p>


1. clone this repo
```bash

git clone git@github.com:yohei-freelance/cornel_birdcall_solution.git
cd cornel_birdcall_solution
```
2. fetch data from kaggle
```bash

kaggle competitions download -c birdsong-recognition
kaggle datasets download -d ttahara/birdsong-resampled-train-audio-00
kaggle datasets download -d ttahara/birdsong-resampled-train-audio-01
kaggle datasets download -d ttahara/birdsong-resampled-train-audio-02
kaggle datasets download -d ttahara/birdsong-resampled-train-audio-03
kaggle datasets download -d ttahara/birdsong-resampled-train-audio-04
```

<p>and place folders to cornel_birdcall_solution/data .</p>
<p>then, don't forget to set output dir in cornel_birdcall_solution/configs/config.yaml</p>

3. run training!
```bash

bash scripts/train.sh
```

<p>I'm verry sorry, but I will upload the inference code and figure of solution later.</p>
