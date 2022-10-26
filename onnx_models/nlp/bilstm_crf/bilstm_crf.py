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
from common.model import Model
import copy
import os
import pickle as cPickle


class Bilstm_CrfFactory(OnnxModelFactory):
    model = "bilstm_crf"

    def new_model():
        return Bilstm_Crf()


class Bilstm_CrfItem(Item):
    def __init__(self, data):
        self.sentence = data[0]
        self.label = data[1]
        self.length = 0


class Bilstm_Crf(OnnxModel):
    def __init__(self):
        super(Bilstm_Crf, self).__init__()
        self.options = self.get_options()
        self.options.add_argument('--model_path',
                                  default='./model/BiLSTM_CRF-seqlen96-op13-fp32-N.onnx',
                                  help='onnx path')
        self.options.add_argument('--data_path',
                                  default='./data/dev',
                                  type=str,
                                  help='dataset path')
        self.options.add_argument('--pre_model',
                                  default='./model/BiLSTM_CRF_data.pkl',
                                  type=str,
                                  help='pkl path')
        self.options.add_argument("--batch_size",
                                  type=int,
                                  default=64,
                                  help="batch_size")
        self.options.add_argument("--max_length",
                                  type=int,
                                  default=96,
                                  help="length")

    def create_dataset(self):
        data = []
        word_to_ix = {"unk": 0, "pad": 1}
        self.ix_to_word = {}
        tag_to_ix = {"O": 0, "START": 1, "STOP": 2}
        self.ix_to_tag = {}
        with open(self.options.get_pre_model(), "rb") as f:
            data_map = cPickle.load(f)
            word_to_ix = data_map.get("word_to_ix", {})  # word_to_ix
            tag_to_ix = data_map.get("tag_to_ix", {})  # tag_to_ix
        sentence = []  # 一句话中的所有word
        target = []  # 一句话中的所有tag
        with open(self.options.get_data_path(), encoding="utf-8") as f:
            
            ix = -1
            for line in f:
                ix += 1  # 正在处理第ix行数据,便于后续DEBUG
                line = line[:-1]  # 去掉一行末尾的'\n'

                if line == '':  # 一句话结束了
                    # 当数据类型为train时, sentence中不应该有0
                    data.append([sentence, target])
                    sentence = []
                    target = []
                    continue
                try:
                    word, tag = line.split(" ")  # 以空格划分word和tag
                except Exception:  # line为空时发生异常
                    continue

                # self.data_type == 'dev'时，有可能word不在self.word_to_ix中
                # 此时约定这个word为'unk',对应的word_id为0
                sentence.append(word_to_ix.get(word, 0))  # 在这一步直接word->number了
                target.append(tag_to_ix.get(tag, 0))  # tag->number

        for k, v in word_to_ix.items():
            self.ix_to_word[v] = k

        for k, v in tag_to_ix.items():
            self.ix_to_tag[v] = k
    

        return read_dataset(data)

    def load_data(self, data):

        return Bilstm_CrfItem(data)

    def pad_data(self, item):
        i = []

        i.append(item.sentence)
        i.append(item.label)
        max_length = self.options.get_max_length()

        # append之前i[0]:word2id(list), i[1]:tag2id(list)
        # append之后i[0]:word2id, i[1]:tag2id, i[2]:len(i[0])
        item.length = len(item.sentence)  # len(i[0])为这句话的真实长度
        if (len(i[0]) >= max_length):
            i[0] = i[0][:max_length]
            i[1] = i[1][:max_length]
        else:
            i[0] = i[0] + (max_length - len(i[0])) * [1]  # word2id(list)中缺少的部分补1 self.word_to_ix中'pad'对应1
            i[1] = i[1] + (max_length - len(i[1])) * [0]  # tag2id(list)中缺少的部分也补0 self.tag_to_ix中'O'对应0
        item.sentence = np.array(i[0])
        item.label = i[1]

    def preprocess(self, item):
        self.pad_data(item)
        return item

    def run_internal(self, sess, items):
        
        datas = Model.make_batch([item.sentence for item in items])
        a = np.array(datas).astype(np.int64)
        input_data = {"tokens": a,
                      }

        res = sess.run(None, input_data)

        for z in zip(items, res[0]):
            Model.assignment(*z, 'score')
        for z in zip(items, res[1]):
            Model.assignment(*z, 'predict_paths_tmp')

        return items

    def path_to_entity(self, seq_of_word, seq_of_tag, ix_to_word, ix_to_tag):
        entity = []
        res = []
        try:
            for ix in range(len(seq_of_word)):
                if ix_to_tag[seq_of_tag[ix]][0] == 'B':
                    entity = [str(ix), ix_to_word[seq_of_word[ix]] + '/' + ix_to_tag[seq_of_tag[ix]]]  # 起始下标
                elif ix_to_tag[seq_of_tag[ix]][0] == 'M' and len(entity) != 0 \
                        and entity[-1].split('/')[1][1:] == ix_to_tag[seq_of_tag[ix]][1:]:
                    entity.append(ix_to_word[seq_of_word[ix]] + '/' + ix_to_tag[seq_of_tag[ix]])
                elif ix_to_tag[seq_of_tag[ix]][0] == 'E' and len(entity) != 0 \
                        and entity[-1].split('/')[1][1:] == ix_to_tag[seq_of_tag[ix]][1:]:
                    entity.append(ix_to_word[seq_of_word[ix]] + '/' + ix_to_tag[seq_of_tag[ix]])
                    entity.append(str(ix))
                    res.append(entity)
                    entity = []
                else:
                    entity = []
        except Exception as e:
            print('error\t', e)
        return res

    def postprocess(self, item):
        length = item.length

        unpadded_sentence = item.sentence[:length]
        unpadded_label = item.label[:length]
        predict_path = item.predict_paths_tmp.tolist()[:length]

        extracted_entities = self.path_to_entity(seq_of_word=unpadded_sentence, seq_of_tag=predict_path,
                                                 ix_to_word=self.ix_to_word,
                                                 ix_to_tag=self.ix_to_tag,
                                                 )
        correct_entities = self.path_to_entity(seq_of_word=unpadded_sentence, seq_of_tag=unpadded_label,
                                               ix_to_word=self.ix_to_word,
                                               ix_to_tag=self.ix_to_tag,
                                               )
        results = [extracted_entities, correct_entities]
        return results

    def eval(self, collections):
        extracted_entities = []
        correct_entities = []
        for item in collections:
            extracted_entities.extend(item[0])
            correct_entities.extend(item[1])

        intersection_entities = [i for i in extracted_entities if i in correct_entities]

        if len(intersection_entities) != 0:
            accuracy = float(len(intersection_entities)) / len(extracted_entities)
            recall = float(len(intersection_entities)) / len(correct_entities)
            f1 = (2 * accuracy * recall) / (accuracy + recall)
        else:
            f1, accuracy, recall = 0, 0, 0

        print("The F1 is ", f1)
        print("The prediction is ", accuracy)
        print("The recall is ", recall)
        return {}
