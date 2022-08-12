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
from abc import ABC, abstractmethod

class BaseSession(ABC):
    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def postprocess(self):
        pass


class OrtModelSessionInterface(BaseSession):
    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def compute(self):
        print("ort model session run")

    @abstractmethod
    def postprocess(self):
        pass


class OrtModelSession():
    def __init__(self):
        pass

OrtModelSessionInterface.register(OrtModelSession)


class RN50Session(OrtModelSession):
    def __init__(self):
        super(RN50Session, self).__init__()

o = OrtModelSession()
o.compute()

r = RN50Session()
r.compute()
