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

from onnx_models.common_classification import ClassificationModel
from onnx_models.base import OnnxModelFactory
from common.data_process.img_preprocess import img_resize, img_center_crop
import numpy as np


class MobileNetV1Factory(OnnxModelFactory):
    model = "mobilenetv1"

    def new_model():
        return MobileNetV1()


class MobileNetV1(ClassificationModel):
    def __init__(self):
        super(MobileNetV1, self).__init__()

    def preprocess(self, item):
        width = int(self.options.get_input_width())
        height = int(self.options.get_input_height())
        input_size = (width, height)

        image = img_resize(item.data, self.options.get_resize_size())
        image = img_center_crop(image, input_size)
        image_data = np.array(image, dtype='float32')

        norm_image_data = (image_data / 255 - 0.5) * 2
        norm_image_data = norm_image_data.reshape(
            height, width, 3).astype('float32')
        norm_image_data = np.array(norm_image_data).transpose(2, 0, 1)
        item.data = norm_image_data
        return item
