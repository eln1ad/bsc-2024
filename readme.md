# Semsey Dániel Szakdolgozat - OE-NIK 2024 

The goal of the project is to implement a boxing video snippet classifier with temporal boundary regression, the unified model can be viewed as a temporal action detector, thus the whole project belongs to the field of temporal action detection (a subset of computer vision).

The code is not yet ready, updates are to come ...

## Video Classification

At the start of the project a basic video classifier is implemented. The model architecture is largely influenced by the C3D research model, but it is a much smaller version of that. During my experiments I train 2 separate models, one on RGB images and the other on optical flow frames. These models are trained with an NVIDIA GTX 1080TI GPU utilising relatively small number of epochs.

## How to create environment?

```bash
conda create -n bsc python=3.11.*
conda activate bsc
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0 tensorflow==2.13.0
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
conda install matplotlib numpy seaborn pandas tqdm pysimplegui scikit-learn natsort
pip install opencv-python
```

## Note to self
Locations of the videos
- /home/elniad/datasets/boxing/videos (this contains UP-TO-DATE videos)
- /media/elniad/4tb_hdd/datasets/boxing/videos (this contains an OLDER version of videos)

## Bugfixes
Az `natsorted` package `os_sorted` függvényével óvatosan kell bánni, amikor generátort ír az ember  
ha sok file van egy mappában akkor a rendezés sok időbe telik (0,5s) ami miatt **nagyon** belassul  
a tanítás. A tensorflow profiler segítségével meg lehet nézni, hogy mi lassítja a tanítást (I/O művelet, stb.)

A tensorboard így tudom meghívni úgy, hogy a profile szekció is elérhető legyen:  
```bash
conda activate bsc
tensorboard --logdir=/home/elniad/bsc-2024/logs/<CÉLPONT TBOARD PREFIXŰ MAPPA NEVE>
```
