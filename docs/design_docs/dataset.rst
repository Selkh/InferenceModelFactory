Dataset
==================================

Overview
----------------------------------

*Dataset* mainly relies on scaffolding of 'ray.data', a high-performance library for distributed data collection (Refer to https://github.com/ray-project/ray/tree/master/doc/source/data for details).
Meanwhile, we do some encapsulations to adapt to specific scenarios with low overhead, although not very **Actor** but practical. 

Usage
----------------------------------
Dataset could only be created via the 'read_*()', following the design method of 'Ray', including 'read_json', 'read_csv', 'read_text', 'read_numpy' and an extension, 'read_dataset', transforming an existing dataset to a 'Ray' one.
Here we assume the exsiting dataset is *Iterable* or *List-like* where elements could be retrieved by index.
Therefore, model developers will definitely call one or more 'read_*()', usually in the overrided method of 'preprocess'.

Design
----------------------------------
Before introducing our design, it's necessary to briefly introduce 'ray.data.Dataset' and 'ray.data.DatasetPipeline'.
In 'Ray', each piece of data will be stored in a ’*row*‘. Data is a general representation which can value or structure.
A '*block*' is composed of rows, in other words '*row*' is the smallest granularity in the '*block*'.
'*Dataset*' is implemented as list of reference of '*block*'.
'*DatasetPipeline*' only undertakes scheduling execution, recording operations as stages, optimizing and fusing, and performing on '*Dataset*' when 'iter_*'.
'*DatasetPipeline*' is very similar to 'future' for non-blocking asynchronous execution.
In this method, operations should not be aware of object entity during async-execution but exclusive with our 'Enflame framework'.
Methods of both '*Dataset*' and '*DatasetPipeline*' will generate a new object as return value not itself, which offers possibility and difficulty for our decorating.
Therefore, we rearranged the dataset at block-level with some additional operations to be compatible for more '*Session*'.
To chain the pipeline, we replace original method with decorated with the same name at every return object.

Compared to serial execution, example as

>>> [load1]
>>>        [prep1]
>>>               [load2]
>>>                      [prep2]
>>>                             [batch_infer1]
>>>                                           [load3]
>>>                                                  [prep3]
>>>                                                         [load4]
>>>                                                                [prep4]
>>>                                                                       [batch_infer2]
>>> Time ------------------------------------------------------------------------------>

there is obvious gains under pipeline mode

>>> [load1][load2][load3][load4]
>>>        [prep1][prep2][prep3][prep4]
>>>                      [batch_infer1][batch_infer2]
>>> Time ------------------------------------------->