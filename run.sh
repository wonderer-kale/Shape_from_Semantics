python ./run_tetsplatting.py \
    --views '[{"prompt":"a running dog","elevation":15,"azimuth":0},
          {"prompt":" an apple","elevation":15,"azimuth":90},
          {"prompt":"a book","elevation":15,"azimuth":180}]' \
    -o outputs/multiview --gpus 6