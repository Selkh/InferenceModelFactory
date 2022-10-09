"""
/* Copyright 2018 The Enflame Tech Company. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
"""
# !/usr/bin/python
# -*- coding: utf-8 -*-

from onnx_models.base import OnnxModelFactory, OnnxModel
from common.dataset import read_csv, Item
from common.model import Model
import numpy as np
import cv2
import os


class UNET2DFactory(OnnxModelFactory):
    model = "2dunet"

    def new_model():
        return UNET2D()


class UNET2DItem(Item):
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask


class UNET2D(OnnxModel):
    def __init__(self):
        super(UNET2D, self).__init__()
        self.options = self.get_options()
        self.options.add_argument("--model_path",
                                  default="./model/2dunet-tf-op13-fp32-N.onnx",
                                  help="Onnx path")
        self.options.add_argument("--data_path",
                                  default="./dagm2007",
                                  type=str,
                                  help="Dataset path.")
        self.options.add_argument("--batch_size",
                                  default=1,
                                  type=int,
                                  help="Batch size.")
        self.options.add_argument("--input_height",
                                  default=512,
                                  type=int,
                                  help="Model input image height.")
        self.options.add_argument("--input_width",
                                  default=512,
                                  type=int,
                                  help="Model input image width.")
        self.options.add_argument("--iou_thres",
                                  nargs="+",
                                  type=float,
                                  default=[0.5, 0.75, 0.85, 0.95, 0.99],
                                  help="IoU threshold for eval.")

    def create_dataset(self):
        csv_path = os.path.join(
            self.options.get_data_path(), "private/Class1/test_list.csv")
        return read_csv(csv_path)

    def load_data(self, path):
        image_dir = os.path.join(
            self.options.get_data_path(), "private/Class1/Test")
        mask_image_dir = os.path.join(
            self.options.get_data_path(), "private/Class1/Test/Label")

        input_image_name, image_mask_name = path[0], path[1]
        image_filepath = os.path.join(image_dir, input_image_name)
        image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)

        if image_mask_name == '':
            mask = None
        else:
            mask_filepath = os.path.join(mask_image_dir, image_mask_name)
            mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        return UNET2DItem(image, mask)

    def preprocess(self, item):
        def process(image, normalize_data_method):
            if (image is None) and (normalize_data_method == "zero_one"):
                image = np.zeros((self.options.get_input_height(
                ), self.options.get_input_width(), 1), dtype=np.float32)
            else:
                image = cv2.resize(image, (self.options.get_input_height(
                ), self.options.get_input_width())).astype(np.float32)
                if normalize_data_method == "zero_centered":
                    image = image / 127.5 - 1
                elif normalize_data_method == "zero_one":
                    image = image / 255.0
                image = np.expand_dims(image, axis=2)
            return image

        item.data = process(item.data, normalize_data_method="zero_centered")
        item.mask = process(item.mask, normalize_data_method="zero_one")

        return item

    def run_internal(self, sess, items):
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        datas = Model.make_batch([item.data for item in items])
        res = sess.run([output_name], {input_name: datas})[0]
        for z in zip(items, res):
            Model.assignment(*z, 'res')
        return items

    def postprocess(self, item):
        def iou_score(pred, label, threshold, eps=1e-5):
            label = (label > threshold).astype(float)
            pred = (pred > threshold).astype(float)
            intersection = label * pred
            intersection = np.sum(intersection, axis=(0, 1))
            numerator = 2.0 * intersection + eps
            divisor = np.sum(label, axis=(0, 1)) + \
                np.sum(pred, axis=(0, 1)) + eps
            return np.mean(numerator / divisor)

        iou = []
        for thres in self.options.get_iou_thres():
            iou.append(iou_score(item.res, item.mask, threshold=thres))
        return iou

    def eval(self, collections):
        print("2dunet eval")
        iou_results = list(np.mean(np.array(collections), axis=0))
        eval_results = {}
        for idx, thres in enumerate(self.options.get_iou_thres()):
            eval_results["IOU-{}".format(thres)
                         ] = format(iou_results[idx], ".3%")
        print("eval_results:", eval_results)
        return eval_results