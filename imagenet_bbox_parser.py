from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import gzip
import numpy as np
import pickle as pkl
import tarfile
from tqdm import tqdm
import xml.etree.ElementTree as ET
from resnet.data.imagenet import ImageNetDataset


def read_bbox(folder, split, label_name, out_dict):
  split_folder = os.path.join(folder, split)
  label_folder = os.path.join(split_folder, label_name)
  img_id_list = os.listdir(label_folder)
  tar_fname = os.path.join(folder, "annotation", label_name + ".tar.gz")
  with tarfile.open(tar_fname, "r:gz") as tar:
    for img_id in tqdm(img_id_list):
      img_id_idx = int(img_id.split("_")[-1].split(".")[0])
      box_list = []
      fname = "".join([
          "Annotation/", label_name, "/", label_name, "_", str(img_id_idx),
          ".xml"
      ])
      found = True
      try:
        member = tar.getmember(fname)
      except:
        found = False
      if found:
        member_f = tar.extractfile(member)
        tree = ET.parse(member_f)
        root = tree.getroot()
        size = tree.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        for ii, obj in enumerate(tree.findall("object")):
          name = obj.find("name").text
          bndbox = obj.find("bndbox")
          xmin = int(bndbox.find("xmin").text)
          ymin = int(bndbox.find("ymin").text)
          xmax = int(bndbox.find("xmax").text)
          ymax = int(bndbox.find("ymax").text)
          if name == label_name:
            box_list.append(
                [ymin / height, xmin / width, ymax / height, xmax / width])
      if len(box_list) == 0:
        bbox = np.array([[0., 0., 1., 1.]], dtype=np.float32)
      else:
        bbox = np.array(box_list, dtype=np.float32)
      out_dict[img_id] = bbox


def read_all(folder, split):
  split_folder = os.path.join(folder, split)
  label_name_list = os.listdir(split_folder)
  with open(os.path.join(folder, "bbox.txt"), "w") as f:
    bbox_dict = {}
    for label_name in tqdm(label_name_list):
      read_bbox(folder, split, label_name, bbox_dict)
      # for key in bbox_dict.keys():
      #   f.write(key)
      #   for bb in

  with gzip.open(os.path.join(folder, "bbox_cache.pklz"), "wb") as f:
    pkl.dump(bbox_dict, f)


def main():
  read_all("/ais/gobi4/mren/data/imagenet", "train")


if __name__ == "__main__":
  main()
