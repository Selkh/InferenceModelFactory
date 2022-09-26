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
        print("\nmodel: {} has been registered, please rename\n".format(name))


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
    _framework_name = "name"
    _model_name = "model"
    _registered_map = {}
    _develop_map = {"done": [], "beta": [], "alpha": []}

    _abstractmethods = set(["new_model"])

    def __new__(mcs, *args, **kwargs):
        cls = super(BaseModelFactory, mcs).__new__(mcs, *args, **kwargs)

        if cls.__name__ == "ModelFactory" or cls.model == "undefined":
            return cls

        # Similar implement to abc.abstractmethod, as each derived model class
        # must realize its create method: new_model. We tend to catch error
        # earlier than instantiate.
        for method_name in mcs._abstractmethods:
            if method_name not in cls.__dict__.keys():
                raise ModelNotCompleteException(cls.model, method_name)

        mcs.__register_new_model(cls)
        return cls

    @classmethod
    def __register_new_model(mcs, cls):
        framework_name = getattr(cls, mcs._framework_name, None)
        if not framework_name:
            return

        model_name = getattr(cls, mcs._model_name, None)
        if not model_name:
            raise ModelNameUndefinedException()

        cls_name = framework_name + "-" + model_name
        if cls_name in mcs._registered_map:
            raise ModelNameConflictException(cls_name)

        mcs._registered_map[cls_name] = cls

        if not hasattr(cls, '_develop_stage'):
            mcs._develop_map["done"].append(cls_name)
        else:
            assert cls._develop_stage in ['alpha', 'beta', 'done']
            mcs._develop_map[cls._develop_stage].append(cls_name)

    def display_all(cls):
        return type(cls)._registered_map

    def display_all_stages(cls):
        return type(cls)._develop_map

    def display_by_stage(cls, stage: str):
        assert stage in ['alpha', 'beta', 'done']
        return type(cls)._develop_map[stage]

    def display_by_model(cls, model: str):
        if model in type(cls)._develop_map["done"]:
            return "done"
        elif model in type(cls)._develop_map["beta"]:
            return "beta"
        elif model in type(cls)._develop_map["alpha"]:
            return "alpha"
        else:
            raise ValueError(
                "model: {} seems not supported by now, please check."
                "Call 'display_all' to get all models support'".format(model))

    def get(cls, name: str):
        if not cls.is_registered(name):
            raise ModelUnimplementException()
        return cls._registered_map[name]

    def is_registered(cls, name: str):
        all_models = cls.display_all()
        if name in all_models:
            return True
        else:
            return False

    def reset(cls):
        cls._registered_map.clear()


class ModelFactory(metaclass=BaseModelFactory):
    pass


def Completeness(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return Completeness("done")(args[0])

    if "stage" in kwargs:
        stage = kwargs["stage"]
        # Divide the development of a model into three stages:
        #    alpha: run pass on cpu & gpu, satisfy correctness
        #    beta: run pass on gcu
        #    done: run pass on gcu, satisfy correctness and deliver to QA
        assert stage in ['alpha', 'beta', 'done']
    elif kwargs:
        raise KeyError("Unkonwn key in kwargs: {}".format(kwargs.keys()))
    else:
        stage = "done"

    def wrap(obj):
        if not obj.__doc__:
            obj.__doc__ = ""

        obj.__doc__ += ("\n Development in '{}' stage".format(stage))
        setattr(obj, '_develop_stage', stage)
        return obj
    return wrap
