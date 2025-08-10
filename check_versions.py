#import torch
#import detectron2

#print("Torch version:", torch.__version__)
#print("Detectron2 version:", detectron2.__version__)

import hashlib
import os

def md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

cfg_path = "/Users/mariakhan/Desktop/FruitWiseModel/output_files/cfg.pkl"
weights_path = os.path.expanduser("~/model_files/output/model_final.pth") 

cfg_hash = md5(cfg_path)
weights_hash = md5(weights_path)

print("Local cfg.pkl MD5:", cfg_hash)
print("Local model_final.pth MD5:", weights_hash)
