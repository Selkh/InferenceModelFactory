#!/usr/bin/python3

from os import listdir, mkdir
from os.path import join, exists
import json
import pickle
from absl import flags, app
import onnxruntime as rt
from itertools import product
import numpy as np
from skimage.transform import SimilarityTransform
import cv2

FLAGS = flags.FLAGS

class RetinaFace(object):
  def __init__(self, model_path):
    self.sess = rt.InferenceSession(model_path, providers = ['CPUExecutionProvider'])
    self.input_names = [node.name for node in self.sess.get_inputs()]
    self.output_names = [node.name for node in self.sess.get_outputs()]
  def preprocess(self, image):
    img = np.float32(image)
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_height, im_width, _ = img.shape
    scale_h = 640 / im_height
    scale_w = 640 / im_width
    im_scale = min(scale_w, scale_h)
    img = cv2.resize(img, dsize = None, fx = im_scale, fy = im_scale, interpolation = cv2.INTER_LINEAR)
    img -= np.array([104,117,123])
    im_height, im_width, _ = img.shape
    img = np.pad(img, [[0, 640 - im_height],[0, 640 - im_width],[0,0]], "constant", constant_values = 0)
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, axis = 0)
    return img, im_scale
  def get_priorbox(self,):
    anchors = list()
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    for k, step in enumerate([8, 16, 32]):
      _min_sizes = min_sizes[k]
      for i, j in product(range(int(np.ceil(640/step))), range(int(np.ceil(640/step)))):
        for min_size in _min_sizes:
          s_kx = min_size / 640
          s_ky = min_size / 640
          dense_cx = [x * step / 640 for x in [j + 0.5]]
          dense_cy = [y * step / 640 for y in [i + 0.5]]
          for cy, cx in product(dense_cy, dense_cx):
            anchors += [cx, cy, s_kx, s_ky]
    anchors = np.reshape(anchors, (-1, 4))
    return anchors
  def postprocess(self, loc, conf, landms, im_scale, conf_thres = 0.45, iou_thres = 0.5):
    priorbox = self.get_priorbox()
    loc = np.squeeze(loc, axis = 0)
    conf = np.squeeze(conf, axis = 0)
    landms = np.squeeze(landms, axis = 0)
    # 1) decode boxes
    boxes = np.concatenate([priorbox[:,:2] + loc[:,:2] * .1 * priorbox[:,2:],
                            priorbox[:,2:] * np.exp(loc[:,2:] * .2)], axis = 1)
    boxes[:,:2] -= boxes[:,2:] / 2
    boxes[:,2:] += boxes[:,:2]
    boxes /= im_scale
    scores = conf[:,1]
    # 2) decode landmarks
    landms = np.concatenate([priorbox[:,:2] + landms[:,:2] * .1 * priorbox[:,2:],
                             priorbox[:,:2] + landms[:,2:4] * .1 * priorbox[:,2:],
                             priorbox[:,:2] + landms[:,4:6] * .1 * priorbox[:,2:],
                             priorbox[:,:2] + landms[:,6:8] * .1 * priorbox[:,2:],
                             priorbox[:,:2] + landms[:,8:10] * .1 * priorbox[:,2:]], axis = 1)
    landms /= im_scale
    # 3) filter with confidence
    inds = np.where(scores > conf_thres)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]
    # 4) nms
    order = scores.argsort()[::-1]
    boxes = boxes[order] # boxes.shape = (target number, 4)
    landms = landms[order]
    scores = scores[order] # scores.shape = (target number)

    dets = np.concatenate([boxes, np.expand_dims(scores, axis = -1)], axis = -1) # dets.shape = (target number, 5)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = list()
    while order.size > 0:
      i = order[0]
      keep.append(i)
      xx1 = np.maximum(x1[i], x1[order[1:]])
      yy1 = np.maximum(y1[i], y1[order[1:]])
      xx2 = np.minimum(x2[i], x2[order[1:]])
      yy2 = np.minimum(y2[i], y2[order[1:]])

      w = np.maximum(0., xx2 - xx1 + 1)
      h = np.maximum(0., yy2 - yy1 + 1)
      inter = w * h
      ovr = inter / (areas[i] + areas[order[1:]] - inter)

      inds = np.where(ovr <= iou_thres)[0]
      order = order[inds + 1]
    results = list()
    for idx in keep:
      face_rect = boxes[idx] * 640
      score = scores[idx]
      pts = np.reshape(landms[idx], (5,2)) * 640
      results.append({'area': face_rect, 'score': score, 'landmarks': pts})
    return results
  def predict(self, raw_image, conf_thres = 0.45, iou_thres = 0.5):
    image = raw_image.copy()
    inputs, im_scale = self.preprocess(image)
    loc, conf, landms = self.sess.run(self.output_names, {self.input_names[0]: inputs})
    return self.postprocess(loc, conf, landms, im_scale, conf_thres = conf_thres, iou_thres = iou_thres)
  def visualize(self, raw_image, results):
    img_raw = raw_image.copy()
    if len(results) != 0:
      for target in results:
        pt1 = target['area'][:2].astype(np.int32).tolist()
        pt2 = target['area'][2:].astype(np.int32).tolist()
        cv2.rectangle(img_raw, pt1 = tuple(pt1), pt2 = tuple(pt2), color = (0, 0, 255), thickness = 2)
        text = "{:.4f}".format(target['score'])
        cv2.putText(img_raw, text, (int(pt1[0]), int(pt1[1]) + 12), cv2.FONT_HERSHEY_DUPLEX, .5, (225,225,225))
        for pt in target['landmarks']:
          cv2.circle(img_raw, (int(pt[0]), int(pt[1])), 1, (0,255,255), 4)
    return img_raw

def norm_crop(img, landmark, image_size=112):
    ARCFACE_SRC = np.array([[
        [122.5, 141.25],
        [197.5, 141.25],
        [160.0, 178.75],
        [137.5, 225.25],
        [182.5, 225.25]
    ]], dtype=np.float32)

    def estimate_norm(lmk):
        assert lmk.shape == (5, 2)

        tform = SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = np.inf
        src = ARCFACE_SRC

        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]

        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))

        if error < min_error:
            min_error = error
            min_M = M
            min_index = i

        return min_M, min_index

    M, pose_index = estimate_norm(landmark)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def load_imgs(dfdc_root, retinaface_onnx):
  retinaface = RetinaFace(retinaface_onnx)
  with open(join(dfdc_root, 'train_sample_videos', 'metadata.json'), 'r') as  f:
    labels = json.loads(f.read())
  for fname, meta_info in labels.items():
    path = join(dfdc_root, 'train_sample_videos', fname)
    label = True if meta_info['label'] == 'REAL' else False
    reader = cv2.VideoCapture(path)
    face_count = 0
    while True:
      for _ in range(9):
        reader.grab()
      success, img = reader.read()
      if not success: break
      results = retinaface.predict(img)
      if len(results) == 0: continue
      boxes = np.array([result['area'] for result in results]) # boxes.shape = (target_num, 4)
      landms = np.array([result['landmarks'] for result in results]) # landms.shape = (target_num, 5, 2)
      areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
      order = areas.argmax()
      boxes = boxes[order] # boxes.shape = (4)
      landms = landms[order] # landms.shape = (5, 2)
      # crop faces
      landmarks = landms.reshape(5, 2).astype(np.int)
      img = norm_crop(img, landmarks, image_size=320)[:,:,::-1] / 255.
      yield img, label

def add_options():
  flags.DEFINE_string('dfdc_root', default = None, help = 'path to deepfake detection directory')
  flags.DEFINE_string('retinaface', default = 'retinaface-rn50-op13-fp32.onnx', help = 'path to retinaface onnx')
  flags.DEFINE_float('iou_thres', default = 0.5, help = 'iou threshold')
  flags.DEFINE_float('conf_thres', default = 0.8, help = 'conf threshold')
  flags.DEFINE_string('output', default = 'data', help = 'output directory')

def main(unused_argv):
  if not exists(FLAGS.output): mkdir(FLAGS.output)
  with open(join(FLAGS.output, 'dataset.pkl'), 'wb') as f:
    for img, label in load_imgs(FLAGS.dfdc_root, FLAGS.retinaface):
      pickle.dump((img, label), f)

if __name__ == "__main__":
  add_options()
  app.run(main)

