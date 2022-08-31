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
#!/usr/bin/python
# -*- coding: utf-8 -*-

from onnx_models.base import OnnxModelFactory, OnnxModel


class RN50Factory(OnnxModelFactory):
    model = "resnet50"

    def new_model():
        return RN50()


class RN50(OnnxModel):
    def __init__(self):
        super(RN50, self).__init__()
        self.options = self.get_options()
        self.options.add_argument('--model_path')

    def get_model(self):
        return self.options.get_model_path()

    def preprocess(self, *args, **kwargs):
        # model_path = self.options.get_model_path()
        model_path = self.get_model()
        print("model path: ", model_path)
        print("rn50 engine preprocessing")

    def run_internal(self, sess, *args, **kwargs):
        pass

    def postprocess(self, *args, **kwargs):
        print("rn50 engine postprocessing")

