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
from abc import ABC, abstractmethod
from .device import Device
from .options import Options, get_default_options


class ModelNotSetOptionException(Exception):
    def __init__(self):
        print("Error: Please set option to model before run")


class ModelNotSetDeviceException(Exception):
    def __init__(self):
        print("Error: Please set device to model before run")


class ModelNotInitException(Exception):
    def __init__(self):
        print(
            "Error: If '__init__' is overrided, that of super class must"
            " be derived in concrete model"
        )


class ModelArgumentException(Exception):
    def __init__(self, name: str, args: str):
        print("method '{}' has additional arguments: {}".format(name, args))


class Model(ABC):
    __slot__ = ["_device", "_options"]

    def __init__(self):
        options = get_default_options()
        subparsers = options.get_subparsers()
        if self.__class__.__name__ in subparsers.choices:
            self._options = Options(options.get_parser(), freeze=True)
        else:
            subparser = subparsers.add_parser(self.__class__.__name__.lower())
            self._options = Options(subparser)

        # add 'device' as default argument considering its general usage
        self._options.add_argument('--device', default='gcu', type=str,
                                   help='on which device to execute model,'
                                   'partial or full support cpu/gpu/gcu')

    def set_options(self, options: Options):
        # setattr(self, 'options', options)
        self._options = options

    def get_options(self) -> Options:
        if not hasattr(self, "_options"):
            raise ModelNotInitException()
        return self._options

    def set_device(self, device: Device):
        # setattr(self, 'device', device)
        self._device = device

    def get_device(self):
        if not hasattr(self, "_device"):
            raise ModelNotSetDeviceException()
        return self._device

    @abstractmethod
    def create_dataset(self):
        raise NotImplementedError(
            "Must implement method of create dataset for model")

    @abstractmethod
    def load_data(self, path):
        raise NotImplementedError(
            "Must implement method of load data from path of each item"
        )

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def postprocess(self):
        pass

    @abstractmethod
    def eval(self, collections):
        pass

    def __check_argument(self, func, allowed_number):
        argcount = func.__code__.co_argcount
        if argcount > allowed_number:
            raise ModelArgumentException(
                func.__name__,
                str(func.__code__.co_varnames[allowed_number:argcount])[1:-1],
            )

    def __check_rtn_value(self, f):
        import dis

        last_instr = None
        for instr in dis.get_instructions(f):
            if instr.opcode != 83:
                last_instr = instr
        if last_instr and last_instr.argval:
            pass
        else:
            raise ValueError(
                "No return value in function {}.".format(f.__name__))

    def sanity_check(self):
        # check argument and return value for method 'create_dataset'
        self.__check_argument(self.create_dataset, 1)
        self.__check_rtn_value(self.create_dataset)

        # check argument and return value for method 'load_data'
        self.__check_argument(self.load_data, 2)
        self.__check_rtn_value(self.load_data)

    @staticmethod
    def make_batch(batch):
        import numpy as np

        type_name = type(batch[0]).__name__
        if type_name == "ndarray":
            # numpy array
            return np.stack(batch, 0)
        elif type_name in ["int", "float", "str"]:
            # scalar type
            return np.array(batch)
        elif type_name == "list":
            return [Model.make_batch(b) for b in zip(*batch)]
        elif type_name == "dict":
            return {key: Model.make_batch([b[key] for b in batch])
                    for key in batch[0]}
        else:
            raise TypeError("Dataset has data with unsupported type")

    @staticmethod
    def assignment(item, value, key):
        if not key:
            raise ValueError("key for value cannot be None")

        if hasattr(item, key):
            print("Override key: {}".format(key))

        try:
            setattr(item, key, value)
        except AttributeError:
            raise AttributeError(
                "item with type '{}' cannot be setattr".format(type(item)))
