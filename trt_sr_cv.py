import os
import argparse

import cv2
import pycuda.autoinit
from utils.camera import Camera
from utils.sr import TrtSR
# from utils.sr import Trtsr # in-progress

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', type=str, required=True,
    help='model name'
)
parser.add_argument(
    '-v', '--video', type=str, required=True,
    help='input video file name'
)
parser.add_argument(
    '-o', '--output', type=str, required=True,
    help='output video file name'
)
parser.add_argument(
    '-w', '--weight', type=str, required=True,
    help='Tensor-RT engine path'
)
parser.add_argument(
    '--upcale_factor', type=int, default=4
)
parser.add_argument(
    '--frame', type=int, default=24,
    help='output video file frame'
)


def loop_and_detect(cap,
                    trt_sr:TrtSR,
                    writer)->None:
    
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        
        sr = trt_sr(frame)   # by __call__
        writer.write(sr)
        print('.', end='', flush=True)
    
    print('\n Done.')

def main(parser):
    args = parser.parse_args()
    if not os.path.isfile(args.weight):
        raise SystemExit('# [ERROR] weight not found !')
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("# [ERROR] Cannot open the video file")
    frame_width, frame_height = map(int, \
        (cap.get(3) * args.upscale_factor, \
            cap.get(4) * args.upscale_factor))
    writer = cv2.VideoWriter(
        args.output,    # output name
        cv2.VideoWriter_fourcc(*'mp4v'),
        args.frame,
        (frame_width, frame_height)
    )
    
    trt_sr = TrtSR(
        #args.model
        #args.upscale_factor
    )
    
    loop_and_detect(cap, trt_sr, writer)
    writer.release()
    cap.release()

if __name__ == "__main__":
    main(parser)