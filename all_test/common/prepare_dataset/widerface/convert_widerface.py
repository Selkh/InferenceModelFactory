#!/usr/bin/python3
#
# Copyright 2022 Enflame. All Rights Reserved.
#

from shutil import rmtree, move
from os import mkdir
from os.path import exists, join
from absl import flags, app
from zipfile import ZipFile

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_path', default = None, help = 'where the directory containing the downloaded files')
  flags.DEFINE_string('output_path', default = 'data', help = 'where to extract the dataset to')

def main(unused_argv):
  assert exists(join(FLAGS.input_path, 'WIDER_val.zip'))
  assert exists(join(FLAGS.input_path, 'wider_easy_val.mat'))
  assert exists(join(FLAGS.input_path, 'wider_face_val.mat'))
  assert exists(join(FLAGS.input_path, 'wider_hard_val.mat'))
  assert exists(join(FLAGS.input_path, 'wider_medium_val.mat'))
  if exists(FLAGS.output_path): rmtree(FLAGS.output_path)
  mkdir(FLAGS.output_path)
  mkdir(join(FLAGS.output_path, 'annotations'))
  zip_file = ZipFile(join(FLAGS.input_path, 'WIDER_val.zip'), 'r')
  zip_file.extractall(FLAGS.output_path)
  move(join(FLAGS.input_path, 'wider_easy_val.mat'), join(FLAGS.output_path, 'annotations'))
  move(join(FLAGS.input_path, 'wider_face_val.mat'), join(FLAGS.output_path, 'annotations'))
  move(join(FLAGS.input_path, 'wider_hard_val.mat'), join(FLAGS.output_path, 'annotations'))
  move(join(FLAGS.input_path, 'wider_medium_val.mat'), join(FLAGS.output_path, 'annotations'))

if __name__ == "__main__":
  add_options()
  app.run(main)

