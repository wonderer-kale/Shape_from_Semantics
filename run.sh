python ./run_tetsplatting.py \
    --views '[{"prompt":"a running dog","elevation":15,"azimuth":0},
          {"prompt":" an apple","elevation":15,"azimuth":90},
          {"prompt":"a book","elevation":15,"azimuth":180}]' \
    -o outputs/multiview --gpus 6

python ./run_tetsplatting.py \
    --views '[{"prompt":"an apple","elevation":0,"azimuth":0},
          {"prompt":"Isaac Newton","elevation":0,"azimuth":120},
          {"prompt":"Open Book","elevation":0,"azimuth":-120}]' \
    -o outputs/multiview70 --gpus 2

python ./run_tetsplatting.py \
    --views '[{"prompt":"an apple","elevation":0,"azimuth":0},
          {"prompt":"Isaac Newton","elevation":0,"azimuth":120},
          {"prompt":"Open Book","elevation":0,"azimuth":-120}]' \
    --config_file sfs-view \
    -o outputs/sfs70 --gpus 6

python ./run_tetsplatting.py \
    --views '[{"prompt":"an apple","elevation":0,"azimuth":0},
          {"prompt":"Isaac Newton","elevation":0,"azimuth":120},
          {"prompt":"Open Book","elevation":0,"azimuth":-120}]' \
    --config_file sfs-view \
    -o outputs/sfs70_batch12 --gpus 6