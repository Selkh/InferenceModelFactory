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
from .options import Options, get_default_options

class ModelNotSetOptionException(Exception):
    def __init__(self):
        print("please set option to model before run")

class ModelNotSetDeviceException(Exception):
    def __init__(self):
        print("please set device to model before run")

class Model(ABC):

    __options = get_default_options()

    def set_options(self, options: Options):
        self.__options = options

    def get_options(self) -> Options:
        return self.__options

    def set_device(self, device: Device):
        self.__device = device

    def get_device(self):
        if not hasattr(self, 'device'):
            raise ModelNotSetDeviceException()
        return self.__device

    def create_session(self) -> BaseSession:
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def run_internal(self, session: BaseSession):
        pass

    def run(self, *args, **kwargs):
        sess = self.create_session()
        output = self.run_internal(sess)
        return output

    @abstractmethod
    def postprocess(self):
        pass
