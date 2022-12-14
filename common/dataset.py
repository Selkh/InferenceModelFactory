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

import typing
import ray
from ray.data.context import DatasetContext
from ray.data.datasource import Datasource, ReadTask
from ray.data.block import Block, BlockMetadata, T
from typing import Union, Iterator, List
from types import MethodType
import sys
abc = sys.modules['abc']


if not hasattr(abc, '_get_dump'):
    import weakref

    def _get_dump(cls):
        registry_weakrefs = set(weakref.ref(obj) for obj in cls._abc_registry)
        return (registry_weakrefs,
                cls._abc_cache,
                cls._abc_negative_cache,
                cls._abc_negative_cache_version)

    abc._get_dump = _get_dump

if sys.version_info < (3, 7):
    def wrap(f):
        def wrapper(type_constructor, name, bases,
                    type_kwargs, class_tracker_id, extra):
            for b in bases:
                if hasattr(b, '_is_protocol'):
                    b._is_protocol = True
            return f(type_constructor, name, bases,
                     type_kwargs, class_tracker_id, extra)
        return wrapper

    ray.cloudpickle.cloudpickle_fast._make_skeleton_class = wrap(
        ray.cloudpickle.cloudpickle_fast._make_skeleton_class)


if not ray.is_initialized():
    ray.init(num_cpus=8)


def wrap_map(f):
    def wrapper(self, *args, **kwargs):
        r = f(*args, **kwargs)
        r.map = self.map
        r.map_batches = self.map_batches
        r.window = self.window
        r.repeat = self.repeat
        return r

    return wrapper


def wrap_map_batches(f):
    def wrapper(self, *args, **kwargs):

        if "batch_size" in kwargs.keys() and "drop_last" in kwargs.keys():
            drop_last = kwargs.pop("drop_last")

            if drop_last:
                batch_size = kwargs["batch_size"]
                total_rows = self.count()

                if total_rows % batch_size != 0:
                    d, _ = self.split_at_indices(
                        [total_rows - total_rows % batch_size])
                    r = d.map_batches(*args, **kwargs)
                    r.map = self.map
                    r.window = self.window
                    r.repeat = self.repeat
                    return r

        r = f(*args, **kwargs)
        r.map = self.map
        r.map_batches = self.map_batches
        r.window = self.window
        r.repeat = self.repeat
        return r

    return wrapper


def wrap_pipe_map(f, total_rows):
    def wrapper(self, *args, **kwargs):
        pipe = f(*args, **kwargs)
        pipe.map = MethodType(wrap_pipe_map(pipe.map, total_rows), pipe)
        pipe.map_batches = MethodType(
            wrap_pipe_map_batches(pipe.map_batches, total_rows), pipe
        )
        return pipe

    return wrapper


def wrap_pipe_map_batches(f, total_rows):
    def wrapper(self, *args, **kwargs):

        if "batch_size" not in kwargs.keys():
            return f(*args, **kwargs)

        # When calling ???map_batches??? of DatasetPipeline, data is expected to
        # be organized in the form of "N-windows/1-block/1-row". When a block
        # contains multiple rows, it is considered to have been re-planned
        # which will not be handled again. Therefore, argument 'batch_size'
        # of multiple 'map_batches' must keep same in one single pipeline
        try:
            rows_per_block = self._base_iterable._splits[0].get_metadata()[
                0].num_rows
            if rows_per_block > 1:
                return f(*args, **kwargs)
        except AttributeError:
            pass

        drop_last = kwargs.pop(
            "drop_last") if "drop_last" in kwargs.keys() else False

        batch_size = kwargs["batch_size"]

        context = DatasetContext.get_current()

        # reorg window-block-row
        if total_rows % batch_size != 0 and drop_last:
            dataset_pipe = self.rewindow(blocks_per_window=total_rows)
            context.optimize_fuse_stages = False
            dataset_pipe = dataset_pipe.foreach_window(
                lambda ds: ds.split_at_indices(
                    [total_rows - total_rows % batch_size])[0]
            )
            dataset_pipe = dataset_pipe.rewindow(blocks_per_window=batch_size)
            context.optimize_fuse_stages = True
        else:
            dataset_pipe = self.rewindow(blocks_per_window=batch_size)

        dataset_pipe = dataset_pipe.foreach_window(
            lambda ds: ds.repartition(num_blocks=1)
        )

        pipe = dataset_pipe.map_batches(*args, **kwargs)
        # r = f(*args, **kwargs)
        pipe.map = MethodType(wrap_pipe_map(pipe.map, total_rows), pipe)
        pipe.map_batches = MethodType(
            wrap_pipe_map_batches(pipe.map_batches, total_rows), pipe
        )

        return pipe

    return wrapper


def wrap_transform(f):
    def wrapper(self, *args, **kwargs):
        pipe = f(*args, **kwargs)
        total_rows = self.count()
        pipe.map = MethodType(wrap_pipe_map(pipe.map, total_rows), pipe)
        pipe.map_batches = MethodType(
            wrap_pipe_map_batches(pipe.map_batches, total_rows), pipe
        )
        pipe.repeat = MethodType(wrap_transform(pipe.repeat), pipe)
        return pipe

    return wrapper


def method_wrap(f):
    def wrapper(self, *args, **kwargs):
        dl = f(*args, **kwargs)
        if isinstance(dl, list):
            dl = dl
        else:
            dl = [dl]
        rdl = []
        for d in dl:
            d.map = MethodType(wrap_map(d.map), d)
            d.map_batches = MethodType(wrap_map_batches(d.map_batches), d)

            for name in d.__dir__():
                if not name.startswith("_"):
                    method = getattr(d, name)
                    if method.__name__ == 'wrapper':
                        continue
                    if hasattr(method, "__annotations__"):
                        return_type = method.__annotations__["return"]
                        if return_type == 'DatasetPipeline[T]':
                            setattr(d, name, MethodType(
                                wrap_transform(method), d))
                        elif return_type in ['Dataset[T]', 'Dataset[U]'] \
                                or isinstance(return_type, typing.GenericMeta):
                            setattr(d, name, MethodType(
                                method_wrap(method), d))
                        else:
                            continue
            rdl.append(d)

        if len(dl) == 1:
            return rdl[0]
        else:
            return rdl
    return wrapper


def func_wrap(f):
    def wrapper(*args, **kwargs):
        ds = f(*args, **kwargs)
        ds = ds.repartition(ds.count())
        ds.map = MethodType(wrap_map(ds.map), ds)
        ds.map_batches = MethodType(wrap_map_batches(ds.map_batches), ds)

        # ds.window = MethodType(wrap_transform(ds.window), ds)
        # ds.repeat = MethodType(wrap_transform(ds.repeat), ds)

        for name in ds.__dir__():
            if not name.startswith("_"):
                method = getattr(ds, name)

                if method.__name__ == "wrapper":
                    continue

                if hasattr(method, "__annotations__"):
                    return_type = method.__annotations__["return"]
                    if return_type == "DatasetPipeline[T]":
                        setattr(ds, name, MethodType(
                            wrap_transform(method), ds))
                    elif return_type in ['Dataset[T]', 'Dataset[U]'] \
                            or isinstance(return_type, typing.GenericMeta):
                        setattr(ds, name, MethodType(
                            method_wrap(method), ds))
                    else:
                        continue

        return ds

    return wrapper


class Item:
    pass


class DatasetDatasource(Datasource[T]):
    def prepare_read(self, parallelism, dataset_factory) -> List[ReadTask]:
        def read_fn() -> Iterator[Block]:
            block = list(dataset_factory())
            yield block

        metadata = BlockMetadata(
            num_rows=None,
            size_bytes=None,
            schema=None,
            input_files=None,
            exec_stats=None,
        )
        return [ReadTask(read_fn, metadata)]


def read_dataset(dataset):
    def dataset_factory():
        if isinstance(dataset, Iterator):
            return [d for d in dataset]
        elif hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
            return [dataset[i] for i in range(len(dataset))]
        else:
            raise ValueError("Dataset is neither iterable nor subscriptable")

    ds = ray.data.read_datasource(
        DatasetDatasource(), parallelism=1, dataset_factory=dataset_factory
    )


    return ds


def read_json(paths: Union[str, List[str]]):
    return ray.data.read_json(paths)


def read_csv(paths: Union[str, List[str]]):
    return ray.data.read_csv(paths)


def read_text(paths: Union[str, List[str]]):
    return ray.data.read_text(paths)


def read_numpy(paths: Union[str, List[str]]):
    return ray.data.read_numpy(paths)
