# Shape from Semantic

**Unofficial reproduction of "Shape from Semantics: 3D Shape Generation from Multi-View Semantics".** 

## Installation

```bash
git clone https://github.com/wonderer-kale/Shape_from_Semantics.git --recursive
conda create -n tetsplatting python=3.9
conda activate shape_from_semantic

# install pytorch (e.g. cuda 11.8)
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# install other denpendencies
pip install -r requirements.txt --no-build-isolation

pip install xatlas imageio[ffmpeg] modelscope
```

Download pretrained weights:

```bash
python tools/download_nd_models.py
# copy 256_tets file for dmtet.
cp ./pretrained_models/Damo_XR_Lab/Normal-Depth-Diffusion-Model/256_tets.npz ./load/tets/
# link your huggingface models to ./pretrained_models/huggingface
cd pretrained_models && ln -s ~/.cache/huggingface ./
```

Due to the shutdown of stabilityai/stable-diffusion-2-1-base, we use the alternative model Manojb/stable-diffusion-2-1-base. 

```bash
python -m huggingface_hub.snapshot_download \
    --repo-id Manojb/stable-diffusion-2-1-base \
    --cache-dir ./pretrained_models/huggingface/hub
```

## Generation

Specify the condition in the --views string

```bash
python ./run_tetsplatting.py \
    --views '[{"prompt":"Pumpkin Carriage","elevation":0,"azimuth":0},
          {"prompt":"A lion","elevation":0,"azimuth":180}]' \
    -o outputs/<your output folder> --gpus <GPU ID>
```

## Acknowledgement

This work is built on many amazing research works:

- [threestudio](https://github.com/threestudio-project/threestudio)
- [RichDreamer](https://github.com/modelscope/RichDreamer)
- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
- [nvdiffrec](https://github.com/NVlabs/nvdiffrec)
- [StopThePop](https://github.com/r4dl/StopThePop-Rasterization)
- [TeT-Splatting](https://github.com/fudan-zvg/tet-splatting)
