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
from .device import Device
from .session import BaseSession
from .options import Options, new_options

class ModelNotSetOptionException(Exception):
    def __init__(self):
        print("please set option to model before run")

class ModelNotSetDeviceException(Exception):
    def __init__(self):
        print("please set device to model before run")

class ModelNotInitException(Exception):
    def __init__(self):
        print("If __init__ of concrete model is overrided, 
              __init__ of super class must be called in concrete model")

class Model(ABC):
    __slot__ = ['device', 'options']

    def __init__(self):
        self.options = new_options()

    def set_options(self, options: Options):
        # setattr(self, 'options', options)
        self.options = options

    def get_options(self) -> Options:
        if not hasattr(self, 'options'):
            raise ModelNotInitException()
        return self.options

    def set_device(self, device: Device):
        # setattr(self, 'device', device)
        self.device = device

    def get_device(self):
        if not hasattr(self, 'device'):
            raise ModelNotSetDeviceException()
        return self.device

    # @abstractmethod
    # def create_session(self) -> BaseSession:
    #     pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def postprocess(self):
        pass
