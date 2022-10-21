#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2022 Enflame. All Rights Reserved.
#

import os
import pickle
import argparse
import cv2

import mxnet as mx
from tqdm import tqdm


def get_args_parser(add_help=True):
    """
    Params parser
    Args:
        add_help: whether show the help information.
    """
    parser = argparse.ArgumentParser(description='ArcFace MS1M Data Convert',  add_help=add_help)
    parser.add_argument('--input_data_dir',
                        type=str,
                        help='origin data dir')
    parser.add_argument('--output_data_dir',
                        type=str,
                        help='converted data save dir')
    return parser


def convert_bin(src_path, dst_path):
    """
    Convert binary image data encode and decode style.
    Args:
        src_path: orgin data path (str)
        dst_path: converted data save path (str)
    """
    print(f'Read dataset name: {src_path}')
    bins, issame_list = pickle.load(open(src_path, 'rb'), encoding='bytes')
    dst_list = []
    for i in tqdm(range(0, len(bins))):
        img_bin = bins[i]
        img = mx.image.imdecode(img_bin)
        img = img.asnumpy()
        # use '.png' style for lossless compression
        enc_img = cv2.imencode('.png', img)[1]
        dst_list.append(enc_img.tobytes())

    with open(dst_path, 'wb') as writer:
        pickle.dump([dst_list, issame_list], writer, -1)
    print(f'Save dataset name: {dst_path}')

def main(params):
    """
    Convert dataset
    """
    bin_names = []
    names = os.listdir(params.input_data_dir)
    for name in names:
        if name.split('.')[-1] != 'bin':
            continue
        bin_names.append(name)
    print(f'Find dataset: {bin_names}')
    if not os.path.exists(f'{params.output_data_dir}/converted_ms1m_face'):
        os.makedirs(f'{params.output_data_dir}/converted_ms1m_face')
    for bin_name in bin_names:
        src_name = f'{params.input_data_dir}/{bin_name}'
        dst_name = f'{params.output_data_dir}/converted_ms1m_face/{bin_name}'
        convert_bin(src_path=src_name, dst_path=dst_name)


if __name__ == '__main__':
    args = get_args_parser()
    main(args.parse_args())
