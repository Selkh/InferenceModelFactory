#!/usr/bin/python3

from absl import flags, app
from zipfile import ZipFile

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to compressed dataset')
  flags.DEFINE_string('output', default = 'data', help = 'path to output directory')

def main(unused_argv):
  zip_file = ZipFile(FLAGS.dataset)
  zip_file.extractall(FLAGS.output)

if __name__ == "__main__":
  add_options()
  app.run(main)
