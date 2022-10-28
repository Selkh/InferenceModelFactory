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
import numpy as np
from scipy.special import softmax


class ShuffleNetTVFactory(OnnxModelFactory):
    model = "shufflenet-tv"

    def new_model():
        return ShuffleNetTV()


class ShuffleNetTV(ClassificationModel):
    def __init__(self):
        super(ShuffleNetTV, self).__init__()

    def postprocess(self, item):
        item.res = np.expand_dims(item.res, axis=0)
        item.res = softmax(item.res, axis=1)
        item.label = np.expand_dims(item.label, axis=0)

        pred = np.argmax(item.res, axis=-1)
        acc1 = 1 if pred == item.label else 0

        indices = self.arg_topk(item.res)
        acc5 = (item.label[..., None] == indices).any(axis=-1).sum()

        return acc1, acc5
