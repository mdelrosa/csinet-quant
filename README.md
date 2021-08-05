<!-- >📋  A template README.md for code accompanying a Machine Learning paper -->

# Trainable Codewords and Compression Bounds for Deep Learning-based Multi-Antenna CSI Feedback

This repository is the official implementation of "Trainable Codewords and Compression Bounds for Deep Learning-based Multi-Antenna CSI Feedback (preprint forthcoming). 

<!-- >📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

## Requirements

To install requirements (based on pytorch=1.9.0):

```setup
conda env create -f environment-torch19.yml
```

Includes git repositories ([mdelrosa/brat](https://github.com/mdelrosa/brat), [mdelrosa/mcr](https://github.com/mdelrosa/mcr2)) as dependencies.

### Recommended File Hierarchy

The 

```hierarchy
home
|_ data
|_ models (downloaded from (here)[drive.google.com/softquant-models]) # TODO: write this
|_ git
  |_ brat
  |_ mcr2
  |_ csinet-quant (this repository)
```

<!-- >📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc... -->

## Training

Training of CsiNet-SoftQuant involves three steps.

1. Autoencoder pretraining
2. Center pretraining
3. End-to-end training (autoencoder + soft quantization)

To run all three steps, first change to the `csi_net` directory, then run the following `python` command:

```train
python csi_net.py -d 0 -e outdoor -l norm_sphH4 -dt norm_sphH4 -L 1024 -p1 true -p2 true -tr true 
```

The rate-distortion curves for different models are drawn by running Step 3 multiple times with different values of beta. These training runs use the same results from Steps 1 and 2. To train models with different values of beta, run the following: 

```finetune
python csi_net.py -d 0 -e outdoor -l norm_sphH4 -dt norm_sphH4 -L 1024 -p1 false -p2 false -tr true --tail_dir <beta_tail_dir> --beta <float>
```

<!-- >📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters. -->

## Evaluation (**TODO**)

To evaluate a given model on the COST2100 dataset, run:

```eval
cd csi_net
python csi_net_eval.py -d 0 -e outdoor -l norm_sphH4 -dt norm_sphH4 -L 1024 
```

<!-- >📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below). -->

## Pre-trained Models

You can download pretrained models here:

- [CsiNet-SoftQuant](https://drive.google.com/mymodel.pth). Models are organized as follows:

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on the COST2100 dataset:

### Comparison Between CsiNet-Pro and DeepCMC

![Comparison of CsiNet-Pro and DeepCMC networks under spherical normalization](images/softquant_csinet_vs_cmcnet.jpg "CsiNet Pro vs. DeepCMC")

<!-- <p align="center">
    <img src="images/softquant_csinet_vs_cmcnet.jpg" width="750"\><br>
</p>
<p align="center">
 -->

Reproduce this figure by accessing `results/rd_curves.ipynb`.

<!-- >📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.  -->

## Contributing (**TODO**)

Creative Commons License.

<!-- >📋  Pick a licence and describe how to contribute to your code repository.  -->