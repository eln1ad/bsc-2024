## Video Classifier

How to train the video classifier?

Small snippets can be sampled from every video, with a window size of 8 or 16. Every snippet should have a label attached to it, based on it's IoU value with the closest ground truth instance. If the IoU is higher than 0.5 then the window will be marked as positive, else negative.

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

Videos are located at multiple places:

- /home/elniad/datasets/boxing/videos
- /media/elniad/4tb_hdd/datasets/boxing/videos

Az `natsorted` package `os_sorted` függvényével óvatosan kell bánni, amikor generátort ír az ember  
ha sok file van egy mappában akkor a rendezés sok időbe telik (0,5s) ami miatt **nagyon** belassul  
a tanítás. A tensorflow profiler segítségével meg lehet nézni, hogy mi lassítja a tanítást (I/O művelet, stb.)

A tensorboard így tudom meghívni úgy, hogy a profile szekció is elérhető legyen:  
```bash
conda activate bsc
tensorboard --logdir=/home/elniad/bsc-2024/logs/<CÉLPONT TBOARD PREFIXŰ MAPPA NEVE>
```
