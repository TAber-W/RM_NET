import numpy as np
import argparse
import cv2
import os
from rm_dataset.process_c import get_yaml_data

def run(cfg='data/4_point.yaml'):
    cls_num,cls_name,train_path,val_path = get_yaml_data(cfg)
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', nargs='+', type=str, default='data/4_point.yaml', help='train data info')
    #parser.add_argument('--frame_pth', nargs='+', type=str, default='', help='Frame path for save')
    #parser.add_argument('--frame_len', nargs='+', type=int, default=1, help='Stride for frame')

    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)