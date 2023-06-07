

## Zero-Forge: Feedforward Text-to-Shape Without 3D Supervision

![CLIP](/images/main.png)

Current state-of-the-art methods for text-to-shape generation either require supervised training using a labeled dataset of pre-defined 3D shapes, or perform expensive inference-time optimization of implicit neural representations. In this work, we present ZeroForge, an approach for zero-shot text-to-shape generation that avoids both pitfalls. To achieve open-vocabulary shape generation, we require careful architectural adaptation of existing feed-forward approaches, as well as a combination of data-free CLIP-loss and contrastive losses to avoid mode collapse. Using these techniques, we are able to considerably expand the generative ability of existing feed-forward text-to-shape models such as CLIP-Forge. We support our method via extensive qualitative and quantitative evaluations.

Paper Link: [[Paper]](Arxiv link soon)

If you find our code or paper useful, you can cite at:

[Insert citation here]


## Installation

First create an anaconda environment called `clip_forge` using
```
conda env create -f environment.yaml
conda activate clip_forge
```

Then, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision. Please change the CUDA version based on your requirements. 

```bash
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install git+https://github.com/openai/CLIP.git
pip install sklearn
```

Choose a folder to download the data, classifier and model: 
```
wget https://clip-forge-pretrained.s3.us-west-2.amazonaws.com/exps.zip
unzip exps.zip
```

## Neural Voxel Renderer

The Neural Voxel Renderer+ ([insert link]) submodule will be installed with ZeroForge. To get the weights for the NVR+ model install them with [insert installation instructions]



## Training

```
python continued_training.py 
```

## Inference







