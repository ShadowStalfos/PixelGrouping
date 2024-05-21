# pixelGrouping

This is the code for **Two photos to find them all: PixelGrouping** by Tim van Gelder, Leon Hulshof,	Thomas Komen, Jerrijn Mandjes and Max de Redelijkheid.
In this method, pixelSplat ([code](https://github.com/dcharatan/pixelsplat)) and Gaussian Grouping ([code](https://github.com/lkeab/gaussian-grouping)) are combined to produce an algorithm capable of 3D object segmentation as proposed in Gaussian Grouping, while retaining the improvements upon the 3D scene reconstruction as proposed in pixelSplat.

## Codebase

This version of the codebase matches the original code used for producing the results presented in the paper. 

## Installation

Since the architecture of PixelGrouping is based on the pixelSplat method, the installation and troubleshooting follows the same steps as presented for their method, restated here for convenience:

To get started, create a virtual environment using Python 3.10+:

```bash
python3.10 -m venv venv
source venv/bin/activate
# Install these first! Also, make sure you have python3.11-dev installed if using Ubuntu.
pip install wheel torch torchvision torchaudio
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
```

If your system does not use CUDA 12.1 by default, see the troubleshooting tips below.

<details>
<summary>Troubleshooting</summary>
<br>

The Gaussian splatting CUDA code (`diff-gaussian-rasterization`) must be compiled using the same version of CUDA that PyTorch was compiled with. As of December 2023, the version of PyTorch you get when doing `pip install torch` was built using CUDA 12.1. If your system does not use CUDA 12.1 by default, you can try the following:

- Install a version of PyTorch that was built using your CUDA version. For example, to get PyTorch with CUDA 11.8, use the following command (more details [here](https://pytorch.org/get-started/locally/)):

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- Install CUDA Toolkit 12.1 on your system. One approach (_try this at your own risk!_) is to install a second CUDA Toolkit version using the `runfile (local)` option [here](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local). When you run the installer, disable the options that install GPU drivers and update the default CUDA symlinks. If you do this, you can point your system to CUDA 12.1 during installation as follows:

```bash
LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64 pip install -r requirements.txt
# If everything else was installed but you're missing diff-gaussian-rasterization, do:
LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64 pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
```

</details>

## Acquiring Pre-trained Checkpoints

The checkpoints for this model will be available soon

## Running the Code

### WIP
