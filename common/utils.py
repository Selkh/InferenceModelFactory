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
import os
import sys
import traceback
from types import ModuleType
from importlib.machinery import ModuleSpec
from importlib.abc import Loader
from importlib._bootstrap_external import PathFinder


class PseudoModule(ModuleType):
    def __init__(self, name):
        self.name = name

    def __getattribute__(self, key):
        global count
        if key in ["name",
                   "__name__",
                   "__loader__",
                   "__package__",
                   "__path__"]:
            return super(PseudoModule, self).__getattribute__(key)
        else:
            sys.meta_path.pop()
            raise ModuleNotFoundError(
                f"No module named {self.name!r}", name=self.name)


class PseudoLoader(Loader):
    def create_module(self, spec):
        module = PseudoModule(spec.name)
        return module

    def exec_module(self, module):
        pass


class PseudoFinder(PathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # Be called only when module not found
        parent = fullname.rpartition(".")[0]

        # where requires this not found module
        fname = None
        fno = 0
        fline = ""
        for frame in reversed(traceback.extract_stack()):
            if frame.name == "find_spec":
                # function itself
                continue

            if frame.filename.startswith("<"):
                # system frozen module
                continue
            # '<module>' is not the only standard for call stack
            # if frame.name == '<module>':
            # record the first real import
            fname = frame.filename
            fno = frame.lineno
            fline = frame.line
            break

        if fname and fno and fline:
            if fname.startswith("/usr"):  # need by system lib
                return None
            print(
                "\nWarning: No module named {}, ignore during register but "
                "will raise error if called.\nFirst required by\n"
                "  File '{}', line {}, in <module>\n    {}\n".format(
                    fullname, fname, fno, fline)
            )

        if not parent:
            # module in sys path
            return ModuleSpec(fullname, PseudoLoader())
        else:
            # suppose to error file import
            return super().find_spec(fullname, path, target)


def pseudo_hook(path):
    return PseudoFinder()


# sys.meta_path.insert(3, ExampleFinder())
sys.meta_path.append(PseudoFinder())
# sys.path_hooks.insert(2, example_hook)
# sys.path_importer_cache.clear()


black_list = ["__pycache__", "__init__.py"]


def traversal(path):
    for p in os.listdir(path):
        if p in black_list:
            continue
        leaf_path = os.path.join(path, p)
        if os.path.isdir(leaf_path):
            traversal(leaf_path)
        else:
            if leaf_path.endswith(".py"):
                leaf_path = leaf_path[:-3]
                module = leaf_path.replace("/", ".")
                # print("import module: ", module)
                __import__(module)
