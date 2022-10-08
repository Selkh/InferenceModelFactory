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
from common.dataset import read_text, Item
from common.model import Model
from PIL import Image
from common.data_process.img_preprocess import img_resize, img_center_crop
import numpy as np
import os


class MobileNetV1Factory(OnnxModelFactory):
    model = "mobilenet_v1"

    def new_model():
        return MobileNetV1()


class MobileNetV1Item(Item):
    def __init__(self, data, name, label):
        super().__init__()
        self.data = data
        self.name = name
        self.label = label


class MobileNetV1(OnnxModel):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.options = self.get_options()
        self.options.add_argument('--input_height',
                                  default=224,
                                  type=int,
                                  help='model input image height')
        self.options.add_argument('--input_width',
                                  default=224,
                                  type=int,
                                  help='model input image width')
        self.options.add_argument('--model_path',
                                  default='mobilenet_v1-tf-op13-fp32-N.onnx',
                                  help='onnx path')
        self.options.add_argument("--data_path",
                                  help="dataset path")
        self.options.add_argument('--batch_size',
                                  default=1,
                                  type=int,
                                  help='batch size')

    def create_dataset(self):
        data_path = os.path.join(self.options.get_data_path(), 'val_map.txt')
        return read_text(data_path)

    def load_data(self, path):
        img_file, label = path.split(" ")
        img_file = os.path.join(self.options.get_data_path(), img_file)
        data = Image.open(img_file).convert("RGB")
        name = img_file.split("/")[-1]
        label = int(label)
        return MobileNetV1Item(data, name, label)

    def preprocess(self, item):
        width = int(self.options.get_input_width())
        height = int(self.options.get_input_height())
        input_size = (width, height)
        max_size = max(width, height)

        image = img_resize(item.data, 256 if max_size <= 256 else 342)
        image = img_center_crop(image, input_size)
        image_data = np.array(image, dtype='float32')
        norm_image_data = (image_data / 255 - 0.5) * 2
        norm_image_data = norm_image_data.reshape(
            height, width, 3).astype('float32')
        norm_image_data = np.array(norm_image_data).transpose(2, 0, 1)

        item.data = norm_image_data
        return item

    def run_internal(self, sess, items):
        datas = Model.make_batch([item.data for item in items])
        input_name = sess.get_inputs()[0].name
        res = sess.run([], {input_name: datas})[0]
        for z in zip(items, res):
            Model.assignment(*z, 'res')
        return items

    @staticmethod
    def arg_topk(array, k=5, axis=-1):
        topk_ind_unsort = np.argpartition(
            array, -k, axis=axis).take(indices=range(-k, 0), axis=axis)
        return topk_ind_unsort

    def postprocess(self, item):
        # TODO: item.res shape is (1001,). it has no batch dimension, neither does item.label
        item.res = np.expand_dims(item.res, axis=0)
        item.label = np.expand_dims(item.label, axis=0)

        pred = np.argmax(item.res, axis=-1)
        acc1 = 1 if pred == item.label else 0

        indices = self.arg_topk(item.res)
        acc5 = (item.label[..., None] == indices).any(axis=-1)
        return acc1, acc5

    def eval(self, collections):
        collections = np.array(collections)
        final_result = {"acc1": np.sum(collections[:, 0])/len(collections[:, 0]),
                        "acc5": np.sum(collections[:, 1])/len(collections[:, 1])}
        print("final_result:", final_result)
        return final_result
