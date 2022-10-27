import numpy as np
import argparse
import cv2
import os
from rm_dataset.process_c import get_yaml_data
from rm_dataset.dataloader import Load_img
from torch.utils.data import Dataset, DataLoader


def run(cfg='data/4_point.yaml'):
    cls_num,cls_name,train_path,val_path = get_yaml_data(cfg)
    #Load_img.load_image()
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', nargs='+', type=str, default='data/4_point.yaml', help='train data info')

    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)