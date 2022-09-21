"""
/* Copyright 2022 The Enflame Tech Company. All Rights Reserved.

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

__all__ = [
    "ModelFactory",
    "ModelUnimplementException",
    "ModelNameConflictException",
    "ModelNameUndefinedException",
    "ModelNotCompleteException",
]


class ModelUnimplementException(Exception):
    pass


class ModelNameConflictException(Exception):
    def __init__(self, name: str):
        print("\nname: {} has been registered, please check or rename\n".format(name))


class ModelNameUndefinedException(Exception):
    pass


class ModelNotCompleteException(Exception):
    def __init__(self, name: str, method: str):
        print(
            "\nmodel: {} does not have method: {}, please realize it\n".format(
                name, method
            )
        )


class BaseModelFactory(type):
    __framework_name = "name"
    __model_name = "model"
    __registered_map = {}

    __abstractmethods = set(["new_model"])

    def __new__(mcs, *args, **kwargs):
        cls = super(BaseModelFactory, mcs).__new__(mcs, *args, **kwargs)

        if cls.__name__ == "ModelFactory" or cls.model == "undefined":
            return cls

        # A similar implement of abc.abstractmethod, as each derived model class
        # must realize its create method: new_model. We tend to catch error
        # earlier than instantiate.
        for method_name in mcs.__abstractmethods:
            if method_name not in cls.__dict__.keys():
                raise ModelNotCompleteException(cls.model, method_name)

        mcs.__register_new_model(cls)
        return cls

    @classmethod
    def __register_new_model(mcs, cls):
        framework_name = getattr(cls, mcs.__framework_name, None)
        if not framework_name:
            return

        model_name = getattr(cls, mcs.__model_name, None)
        if not model_name:
            raise ModelNameUndefinedException()

        cls_name = framework_name + "-" + model_name
        if cls_name in mcs.__registered_map:
            raise ModelNameConflictException(cls_name)

        mcs.__registered_map[cls_name] = cls

    def display_all(cls):
        return type(cls).__registered_map

    def get(cls, name: str):
        if not cls.is_registered(name):
            raise ModelUnimplementException()
        return cls.__registered_map[name]

    def is_registered(cls, name: str):
        all_models = cls.display_all()
        if name in all_models:
            return True
        else:
            return False

    def reset(cls):
        cls.__registered_map.clear()


class ModelFactory(metaclass=BaseModelFactory):
    pass
