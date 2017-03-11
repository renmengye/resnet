from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import os
import numpy as np
import pickle as pkl
import tarfile
import time
import tensorflow as tf
import threading

from resnet.data.synset import get_index, get_label
from resnet.data.vgg_preprocessing import preprocess_image as vgg_preproc
from resnet.data.inception_preprocessing import preprocess_image as inception_preproc
from resnet.utils import logger

DEFAULT_FOLDER = "data/imagenet"

log = logger.get()


def read_image(path):
  return cv2.imread(path)


def read_image_rgb(path):
  img = read_image(path)
  img = img[:, :, [2, 1, 0]]
  return img


class ImageNetDataset(object):

  def __init__(self,
               folder=DEFAULT_FOLDER,
               split="train",
               data_aug=True,
               resize_side_min=256,
               resize_side_max=480,
               crop=224,
               preprocessor="vgg"):
    """
    Args:
      folder: 
      split: train or valid or test
      data_aug: True (random scale, random crop) or False (single center crop)
    """
    self._split = split
    self._folder = folder
    self._img_ids = None
    self._labels = None
    self._bbox_dict = None
    self._data_aug = data_aug
    self._resize_side_min = resize_side_min
    self._resize_side_max = resize_side_max
    self._crop = crop
    self._preprocessor = preprocessor
    self._mutex = threading.Lock()
    with tf.device("/cpu:0"):
      self._image_preproc_inp = tf.placeholder(tf.uint8, [None, None, 3])
      if preprocessor == "vgg":
        self._image_preproc_out = vgg_preproc(
            self._image_preproc_inp,
            crop,
            crop,
            is_training=data_aug,
            resize_side_min=resize_side_min,
            resize_side_max=resize_side_max)
      elif preprocessor == "inception":
        self._image_bbox = tf.placeholder(tf.float32, [1, None, 4])
        self._image_preproc_out = inception_preproc(
            self._image_preproc_inp,
            crop,
            crop,
            is_training=data_aug,
            bbox=self._image_bbox)
        b = self.bbox_dict
      self._session = tf.Session()
    pass

  @property
  def folder(self):
    return self._folder

  @property
  def split(self):
    return self._split

  @property
  def crop(self):
    return self._crop

  @property
  def session(self):
    return self._session

  @property
  def resize_side_min(self):
    return self._resize_side_min

  @property
  def resize_side_max(self):
    return self._resize_side_max

  @property
  def image_preproc_inp(self):
    return self._image_preproc_inp

  @property
  def image_preproc_out(self):
    return self._image_preproc_out

  @property
  def image_bbox(self):
    return self._image_bbox

  @property
  def img_ids(self):
    if self._img_ids is None:
      self._mutex.acquire()
      cache_file = os.path.join(self.folder, "{}_ids.txt".format(self.split))
      if os.path.exists(cache_file):
        _img_ids, _labels = self.read_image_ids(cache_file)
      else:
        _img_ids, _labels = self.extract_image_ids()
      self._img_ids = _img_ids
      self._labels = _labels
      self._mutex.release()
    return self._img_ids

  @property
  def labels(self):
    if self._labels is None and self.split != "test":
      a = self.img_ids
    return self._labels

  def read_image_ids(self, cache_file):
    """Read Image IDs and labels from cache file."""
    _img_ids = []
    _labels = []
    log.info("Reading cache from \"{}\".".format(cache_file))
    with open(cache_file, "r") as f:
      lines = f.readlines()
    for line in lines:
      parts = line.strip("\n").split(",")
      _img_ids.append(parts[0])
      _labels.append(int(parts[1]))
    _labels = np.array(_labels)
    return _img_ids, _labels

  def extract_image_ids(self):
    """Re-populate image IDs and labels from folder structure."""
    _img_ids = []
    _labels = []
    image_folder = os.path.join(self.folder, self.split)
    folders = os.listdir(image_folder)
    for ff in folders:
      subfolder = os.path.join(image_folder, ff)
      image_fnames = os.listdir(subfolder)
      _img_ids.extend(image_fnames)
      _labels.extend([get_index(ff, False)] * len(image_fnames))
    _labels = np.array(_labels)
    with open(cache_file, "w") as f:
      for ii, jj in zip(_img_ids, _labels):
        f.write("{},{}\n".format(ii, jj))
    return _img_ids, _labels

  def get_size(self):
    return len(self.img_ids)

  @property
  def bbox_dict(self):
    if self._bbox_dict is None:
      cache_file = os.path.join(self.folder, "bbox_cache.pkl")
      log.info("Reading cache from \"{}\".".format(cache_file))
      with open(cache_file, "rb") as f:
        self._bbox_dict = pkl.load(f)
    return self._bbox_dict

  def get_batch_idx(self, idx, **kwargs):
    start_time = time.time()
    img = None
    label = np.zeros([len(idx)], dtype="int32")
    for kk, ii in enumerate(idx):
      label_name = get_label(self.labels[ii], False)
      img_fname = os.path.join(self.folder, self.split, label_name,
                               self.img_ids[ii])
      # img_ = read_image(img_fname)
      img_ = read_image_rgb(img_fname)
      if img_ is None:
        raise Exception("Cannot read \"{}\"".format(img_fname))

      if self._preprocessor == "vgg":
        img_ = self.session.run(self.image_preproc_out,
                                feed_dict={self.image_preproc_inp: img_})
      elif self._preprocessor == "inception":
        if self._data_aug:
          bbox = np.expand_dims(self.bbox_dict[self.img_ids[ii]], 0)
          bbox = np.minimum(np.maximum(0.0, bbox), 1.0)
          img_ = self.session.run(
              self.image_preproc_out,
              feed_dict={self.image_preproc_inp: img_,
                         self.image_bbox: bbox})
        else:
          img_ = self.session.run(self.image_preproc_out,
                                  feed_dict={self.image_preproc_inp: img_})
      if img is None:
        img = np.zeros(
            [len(idx), img_.shape[0], img_.shape[1], img_.shape[2]],
            dtype="float32")
      img[kk] = img_
      label[kk] = self.labels[ii]
    results = {"img": img, "label": label}
    return results


if __name__ == "__main__":
  labels = ImageNetDataset(split="train").labels
  print(labels.max())
  print(labels.min())
  print(len(labels))
  labels = ImageNetDataset(split="valid").labels
  print(labels.max())
  print(labels.min())
  print(len(labels))
  img_ids = ImageNetDataset(split="valid").img_ids
  print(img_ids[0])
  print(img_ids[-1])
  print(ImageNetDataset(split="train").get_batch_idx(np.arange(5)))
  print(ImageNetDataset(
      split="valid", data_aug=False).get_batch_idx(np.arange(5)))
