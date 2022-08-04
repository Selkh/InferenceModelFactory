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

import sys
sys.path.append("..")
from .base_onnx import OnnxModelFactory
from engine import CommonEngine


class RN50(OnnxModelFactory):
    model = "resnet50"

    def create_engine() ->CommonEngine:
        return RN50Engine()


class RN50Engine(CommonEngine):
    def preprocessing(self, *args, **kwargs):
        print("rn50 engine preprocessing")

    def run(self, *args, **kwargs):
        print("rn50 engine run")

    def postprocessing(self, *args, **kwargs):
        print("rn50 engine postprocessing")

