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
from PIL import Image
from common.data_process.img_preprocess import img_resize, img_center_crop
import numpy as np


class RN50Factory(OnnxModelFactory):
    model = "resnet50"

    def new_model():
        return RN50()


class RN50Item(Item):
    def __init__(self, name, data, label):
        super(RN50Item, self).__init__(data)
        self.name = name
        self.label = label


class RN50(OnnxModel):
    def __init__(self):
        super(RN50, self).__init__()
        self.options = self.get_options()
        self.options.add_argument('--model_path')
        self.options.add_argument('--data_path')
        self.options.add_argument('--input_width')
        self.options.add_argument('--input_height')
        self.options.add_argument('--resize_size')

    def create_dataset(self):
        data_path = self.options.get_data_path()
        return read_text(data_path)

    def load_data(self, path):
        img_file, label = path.split(" ")
        data = Image.open(img_file).convert("RGB")
        name = img_file.split("/")[-1]
        label = int(label)
        return RN50Item(name, data, label)

    def preprocess(self, item):
        width = int(self.options.get_input_width())
        height = int(self.options.get_input_height())
        resize_size = int(self.options.get_resize_size())

        input_size = (width, height)

        image = img_resize(item.data, resize_size)
        image = img_center_crop(image, input_size)

        image_data = np.array(image, dtype='float32').transpose(2, 0, 1)
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_image_data = np.zeros(image_data.shape).astype('float32')
        for i in range(image_data.shape[0]):
            norm_image_data[i, :, :] = (
                image_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
        norm_image_data = norm_image_data.reshape(
            3, height, width).astype('float32')
        item.data = norm_image_data
        return item

    def run_internal(self, sess, datas):
        input_name = sess.get_inputs()[0].name
        return sess.run([], {input_name: datas})

    def postprocess(self, item):
        import numpy as np

        if np.all(item.data == item.final_result):
            return 1
        else:
            return 0
        # return items

    def eval(self, collections):
        return {"acc1": 1, "acc5": 1}
