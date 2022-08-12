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

from .base import OnnxModelFactory, ONNXSession
from engine import BaseEngine


class yolo(OnnxModelFactory):
    model = "yolo"

    def new_model() -> BaseEngine:
        return YoloEngine()


class YoloEngine(BaseEngine):
    def preprocess(self, *args, **kwargs):
        print("yolo engine preprocessing")

    def run(self, *args, **kwargs):
        sess = ONNXSession()
        print("session name: ", sess.name())
        print("yolo engine run")

    def postprocess(self, *args, **kwargs):
        print("yolo engine postprocessing")
