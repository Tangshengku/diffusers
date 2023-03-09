import json
import jsonlines
import random
import shutil
import os


train_json_path = "/home/dongk/dkgroup/tsk/projects/data/coco/annotations/captions_train2014.json"
train_img_path = "/home/dongk/dkgroup/tsk/projects/data/coco/train2014"
train_out = "/home/dongk/dkgroup/tsk/projects/data/coco/train/train"

with open(train_json_path, "r") as f:
    data = json.load(f)
    samples = data["annotations"]
    with jsonlines.open("/home/dongk/dkgroup/tsk/projects/data/coco/train/train/metadata.jsonl", "w") as wf:
        for sample in samples:
          line = {}

          sample["caption"] = sample["caption"].strip()
          line["text"] = sample["caption"]
          # wf.write(sample["caption"]+"\n")
          image_id = sample["image_id"]
          line["file_name"] = "COCO_train2014_" + "{:012}".format(image_id) + ".jpg"
          
          wf.write(line)
          image_name = os.path.join(train_img_path, "COCO_train2014_" + "{:012}".format(image_id) + ".jpg")
          out_name = os.path.join(train_out, "COCO_train2014_" + "{:012}".format(image_id) + ".jpg")
          # while (True):
          #   if not os.path.exists(out_name):
          #     break
          #   else:
          #     out_name = out_name[:-5] + "_dup.jpg"
          shutil.copyfile(image_name, out_name)
