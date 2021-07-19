# **SimNet**: Enabling Robust Unknown Object Manipulation from Pure Synthetic Data via Stereo
[Thomas Kollar](mailto:thomas.kollar@tri.global), [Michael Laskey](mailto:michael.laskey@tri.global), [Kevin Stone](mailto:kevin.stone@tri.global), [Brijen Thananjeyan](mailto:bthananjeyan@berkeley.edu), [Mark Tjersland](mailto:mark.tjersland@tri.global)
<a href="https://www.tri.global/" target="_blank">
 <img align="right" src="/media/tri-logo.png" width="20%"/>
</a>

[**paper**](https://arxiv.org/abs/2106.16118) / [**project site**](https://sites.google.com/view/simnet-corl-2021) / [**blog**](https://medium.com/toyotaresearch/enabling-real-world-object-manipulation-with-sim-to-real-transfer-20d4962e029)

<img width="90%" src="/media/model.png"/>

This repo contains the code to train the SimNet architecture on procedurally generated simulation
data from scratch (no transfer learning required). We also provide a small set of in-house
manually labelled validation data containing 3d oriented bounding box labels.


## Training the model

### Requirements

You will need a Nvidia GPU with at least 12GB of RAM. All code was tested and developed on Ubuntu
20.04.

All commands are assumed to be run from the root of the `simnet` repo directory (represented by
`$SIMNET_REPO` in commands below).

### Setup

#### Python
Create a python 3.8 virtual environment and install requirements:

```bash
cd $SIMNET_REPO
conda create -y --prefix ./env python=3.8
./env/bin/python -m pip install --upgrade pip
./env/bin/python -m pip install -r frozen_requirements.txt
```

#### Docker
Make sure docker is installed and working without requiring `sudo`. If it is not installed, follow
the [official instructions](https://docs.docker.com/engine/install/) for setting it up.
```bash
docker ps
```

#### Wandb

Launch `wandb` local server for logging training results (*you do not need to do this if you already have a wandb
account setup*). This will launch a local webserver [http://localhost:8080](http://localhost:8080) using docker that you
can use to visualize training progress and validation images. You will have to visit the
[http://localhost:8080/authorize](http://localhost:8080/authorize) page to get the local API access token (this can
take a few minutes the first time). Once you get the key you can paste it into the terminal to continue.

```bash
cd $SIMNET_REPO
./env/bin/wandb local
```


#### Datasets

Download and untar train+val datasets
[simnet2021a.tar](https://tri-robotics-public.s3.amazonaws.com/github/simnet/datasets/simnet2021a.tar)
(18GB, md5 checksum:`b8e1d3cb7200b44b1de223e87141f14b`). This file contains all the training and
validation you need to replicate our small objects results.

```bash
cd $SIMNET_REPO
wget https://tri-robotics-public.s3.amazonaws.com/github/simnet/datasets/simnet2021a.tar -P datasets
tar xf datasets/simnet2021a.tar -C datasets
```

### Train and Validate

Overfit test:
```bash
./runner.sh net_train.py @config/net_config_overfit.txt
```

Full training run (requires 12GB GPU memory)
```bash
./runner.sh net_train.py @config/net_config.txt
```

#### Results

Check wandb ([http://localhost:8080](http://localhost:8080)) to see training progress. On a Titan V, it takes about 48
hours for training to converge, but decent validation results can be seen around 24 hours.

Example validation image visualization:
<img width="100%" src="/media/wandb_example_val_images.png"/>

Example 3D oriented bounding box mAP on validation dataset:
<img width="50%" src="/media/wandb_example_3dmap.png"/>


## Licenses

The source code is released under the MIT license.

The datasets are released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
