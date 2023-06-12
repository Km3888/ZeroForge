

## Zero-Forge: Feedforward Text-to-Shape Without 3D Supervision
by **Kelly Marshall**, **Minh Pham**, **Ameya Joshi**, **Anushrut Jignasu**, **Adarsh Krishnamurthy** and **Chinmay Hegde**

![CLIP](/images/main.png)

Current state-of-the-art methods for text-to-shape generation either require supervised training using a labeled dataset of pre-defined 3D shapes, or perform expensive inference-time optimization of implicit neural representations. In this work, we present ZeroForge, an approach for zero-shot text-to-shape generation that avoids both pitfalls. To achieve open-vocabulary shape generation, we require careful architectural adaptation of existing feed-forward approaches, as well as a combination of data-free CLIP-loss and contrastive losses to avoid mode collapse. Using these techniques, we are able to considerably expand the generative ability of existing feed-forward text-to-shape models such as CLIP-Forge. We support our method via extensive qualitative and quantitative evaluations.

Paper Link: [[Paper]](Arxiv link soon)

If you find our code or paper useful, you can cite at:

[Insert citation here]

## Installation

Our code is an extension of the [CLIP-Forge repo](https://github.com/AutodeskAILab/Clip-Forge) as our method uses their trained model as an initialization. After cloning the repo, you can set up your environment as follows:


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

You can download the CLIP-Forge initialization weights published by Sanghi et al. by running:

```
wget https://clip-forge-pretrained.s3.us-west-2.amazonaws.com/exps.zip
unzip exps.zip
```
This downloads a folder of their experimental results, the only part of which we're interested in is the models subfolder. Alternatively, you can train your own CLIP-Forge weights using their [instructions](https://github.com/AutodeskAILab/Clip-Forge). 

## Neural Voxel Renderer

The Neural Voxel Renderer+ ([insert link]) submodule will be installed with ZeroForge. To get the weights for the NVR+ model install them with [insert installation instructions]

## Running ZeroForge
The main file for training is `zf_training.py` which performs training on an array of text queries. Results are logged using tensorboard in the specified log directory.

```
python zf_training.py --query_array [QUERY] --log_dir [LOGDIR] 
```
The query arrays we used for training ZeroForge are stored in `query_arrays.json` by name, but any set of text queries can be trained on after being added to the .json file.

## Inference







