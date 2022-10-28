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

from onnx_models.base import OnnxModelFactory, OnnxModel
from common.dataset import read_dataset, Item
import numpy as np
import torch
import torch.nn as nn
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.data.audio_to_text import AudioToCharDataset
from nemo.collections.asr.metrics.wer import WER
from common.model import Model
from collections import OrderedDict


class ConformerFactory(OnnxModelFactory):
    model = "conformer"

    def new_model():
        return Conformer()


class ConformerItem(Item):
    def __init__(self, processed_signal, processed_signal_len, target, target_length):
        self.processed_signal = processed_signal
        self.processed_signal_len = processed_signal_len
        self.target = target
        self.target_length = target_length


class NemoWrapper:
    def __init__(self, restore_path):
        self.restore_path = restore_path
        self.nemo = nemo_asr.models.EncDecCTCModel.restore_from(
            self.restore_path)

    def __getstate__(self):
        return {"restore_path": self.restore_path}

    def __setstate__(self, values):
        self.restore_path = values['restore_path']
        self.nemo = nemo_asr.models.EncDecCTCModel.restore_from(
            self.restore_path)


class Conformer(OnnxModel):
    def __init__(self):
        super(Conformer, self).__init__()
        self.options = self.get_options()
        self.options.add_argument('--model_path',
                                  default='./model/conformer_small-asr-nvidia-op13-fp32-N.onnx',
                                  help='onnx path')
        self.options.add_argument('--data_path',
                                  default='./data/test_manifest.json',
                                  type=str,
                                  help='dataset path')
        self.options.add_argument('--nemo_path',
                                  default='./model/stt_en_conformer_ctc_small.nemo',
                                  type=str,
                                  help='dataset path')

        self.options.add_argument("--padding_mode",
                                  type=bool,
                                  default=False,
                                  help="use the padding mode when use dtu")
        self.options.add_argument("--batch_size",
                                  type=int,
                                  default=10,
                                  help="batch_size")
        self.options.add_argument("--max_padding",
                                  type=int,
                                  default=99200,
                                  help="batch_size")

    def create_dataset(self):

        test_manifest = self.options.get_data_path()

        self.model = NemoWrapper(self.options.get_nemo_path())
        dataset = AudioToCharDataset(
            manifest_filepath=test_manifest,
            labels=self.model.nemo.decoder.vocabulary,
            sample_rate=16000,
        )

        return read_dataset(dataset)

    def load_data(self, data):

        return ConformerItem(data[0], data[1], data[2], data[3])

    def _padding(self, item):

        x = item.processed_signal
        length = x.size(0)
        zero_padding = torch.zeros([self.options.get_max_padding() - length])
        item.processed_signal = (torch.cat((x, zero_padding), 0))

    def preprocess(self, item):

        self._padding(item)
        processed_signal, processed_signal_len = self.model.nemo.preprocessor(
            input_signal=item.processed_signal.unsqueeze(0),
            length=item.processed_signal_len.unsqueeze(0),
        )

        if self.options.get_device() == 'dtu' and self.options.get_padding_mode():
            padding_len = 1000 - processed_signal.shape[2]
            processed_signal = torch.nn.functional.pad(
                processed_signal, (0, padding_len), "constant", 0)

        item.processed_signal = processed_signal.squeeze(0)
        item.processed_signal_len = processed_signal_len.squeeze(0)

        return item

    def to_numpy(self, tensor):
        """
        convert tensor to numpy
        """
        return tensor.detach().numpy() if tensor.requires_grad else tensor.numpy()

    def make_batch(self, batch):

        import numpy as np

        type_name = type(batch[0]).__name__

        if type_name == 'Tensor':
            # tensor type

            return torch.stack(batch, 0)
        elif type_name == "list":
            # for b in zip(*batch):
            #     print(b, type(b), len(b))
            return [self.make_batch(b) for b in zip(*batch)]
        else:
            raise TypeError("Dataset has data with unsupported type")

    def run_internal(self, sess, items):

        datas = self.make_batch(
            [[item.processed_signal, item.processed_signal_len] for item in items])

        processed_signal = datas[0]
        processed_signal_len = datas[1]

        ort_inputs = {
            sess.get_inputs()[0].name: self.to_numpy(processed_signal)}

        if "conformer" in self.options.get_nemo_path():
            ort_inputs[sess.get_inputs()[1].name] = self.to_numpy(
                processed_signal_len
            )

        res = sess.run(None, ort_inputs)

        for z in zip(items, res[0]):
            Model.assignment(*z, 'result')

        return items

    def postprocess(self, item):
        results = []

        res = item.result
        ologits = [res]
        alogits = np.asarray(ologits)
        logits = torch.from_numpy(alogits[0])
        greedy_predictions = logits.argmax(dim=-1, keepdim=False).unsqueeze(0)

        # compute wer score

        targets = item.target.unsqueeze(0)
        targets_lengths = item.target_length.unsqueeze(0)

        self.model.nemo._wer.update(
            greedy_predictions, targets, targets_lengths)
        _, wer_num, wer_denom = self.model.nemo._wer.compute()

        result = {
            'wer_num': self.to_numpy(wer_num),
            'wer_denom': self.to_numpy(wer_denom),
        }

        results.append(result)

        return results

    def eval(self, collections):
        wer_nums = []
        wer_denoms = []
        for item in collections:
            wer_nums.append(item[0]['wer_num'])
            wer_denoms.append(item[0]['wer_denom'])
        wer_score = sum(wer_nums) / sum(wer_denoms)
        print("The inference done successful!")
        print("WER= {}".format(wer_score))
        return {}
