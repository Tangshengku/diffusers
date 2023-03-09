import json
import random
import shutil
import os


val_json_path = "/home/dongk/dkgroup/tsk/projects/data/coco/annotations/captions_val2014.json"
val_img_path = "/home/dongk/dkgroup/tsk/projects/data/coco/val2014"
val_out = "/home/dongk/dkgroup/tsk/projects/data/coco/val_30000"

# with open(val_json_path, "r") as f:
#     data = json.load(f)
#     anno = data["annotations"]
#     samples = random.sample(anno, 30000)
with open("/home/dongk/dkgroup/tsk/projects/data/coco/annotations/val_30000.txt", "r") as wf:
    samples = wf.readlines()
    for sample in samples:
      # if sample["caption"][-1] == "\n":
      #   sample["caption"])
      # else:
      sample["caption"] = sample["caption"].strip()
      wf.write(sample["caption"]+"\n")
      image_id = sample["image_id"]
      image_name = os.path.join(val_img_path, "COCO_val2014_" + "{:012}".format(image_id) + ".jpg")
      out_name = os.path.join(val_out, "COCO_val2014_" + "{:012}".format(image_id) + ".jpg")
      if os.path.exists(out_name):
          out_name = out_name[:-5] + "_dup.jpg" 
      shutil.copyfile(image_name, out_name)
