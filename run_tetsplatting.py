import argparse
import os
import sys
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", type=str, default='[{"prompt":"a running dog","elevation":0,"azimuth":0},\
          {"prompt":"a bird","elevation":0,"azimuth":90}]')
    
    parser.add_argument("--run_type", "-r", type=str, default="run_single_new")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--iters", "-i", type=int, default=None)
    args, extras = parser.parse_known_args()

    return args, extras


def main(args, extras):
    start_time = time.time()
    
    views = args.views
    if args.iters is not None:
        cmd = f'bash scripts/tetsplatting/{args.run_type}.sh {args.gpus} \'{views}\' {args.output_dir} trainer.max_steps={args.iters}'
    else:
        cmd = f'bash scripts/tetsplatting/{args.run_type}.sh {args.gpus} \'{views}\' {args.output_dir} '
    print(f"cmd:{cmd}")
    os.system(cmd)
    print(f"time cost:{time.time() - start_time}")


if __name__ == "__main__":
    args, extras = get_args()
    main(args, extras)