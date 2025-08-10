from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import pickle
from PIL import Image
import io
import binascii
import traceback
from detectron2.data.datasets import register_coco_instances
import os

app = Flask(__name__)

# Verify paths and load model configuration
cfg_path = "/Users/mariakhan/Desktop/FruitWiseModel/output_files/cfg.pkl"
weights_path = os.path.expanduser("~/model_files/output/model_final.pth") 

def verify_paths():
    paths = {
        "Model config": cfg_path,
        "Model weights": weights_path,
        "Train COCO JSON": "/Users/mariakhan/Desktop/FruitWiseModel/output_files/train_coco.json",
        "Train images": "/Users/mariakhan/Desktop/FruitWiseModel/output_files/kaggle/input/allergic-fruit-computer-vision/train",
        "Valid COCO JSON": "/Users/mariakhan/Desktop/FruitWiseModel/output_files/valid_coco.json",
        "Valid images": "/Users/mariakhan/Desktop/FruitWiseModel/output_files/kaggle/input/allergic-fruit-computer-vision/valid",
        "Test COCO JSON": "/Users/mariakhan/Desktop/FruitWiseModel/output_files/test_coco.json",
        "Test images": "/Users/mariakhan/Desktop/FruitWiseModel/output_files/kaggle/input/allergic-fruit-computer-vision/test"
    }
    
    for description, path in paths.items():
        print(f"Checking {description}: {path}")
        if not os.path.exists(path):
            print(f"Error: {description} does not exist at {path}")
            return False
    return True

if not verify_paths():
    raise RuntimeError("One or more required paths are incorrect or missing.")

# Register the dataset
register_coco_instances("allergic_fruit_dataset_train", {}, "/Users/mariakhan/Desktop/FruitWiseModel/output_files/train_coco.json", "/Users/mariakhan/Desktop/FruitWiseModel/output_files/kaggle/input/allergic-fruit-computer-vision/train")
register_coco_instances("allergic_fruit_dataset_valid", {}, "/Users/mariakhan/Desktop/FruitWiseModel/output_files/valid_coco.json", "/Users/mariakhan/Desktop/FruitWiseModel/output_files/kaggle/input/allergic-fruit-computer-vision/valid")
register_coco_instances("allergic_fruit_dataset_test", {}, "/Users/mariakhan/Desktop/FruitWiseModel/output_files/test_coco.json", "/Users/mariakhan/Desktop/FruitWiseModel/output_files/kaggle/input/allergic-fruit-computer-vision/test")

# Ensure metadata is correctly set
MetadataCatalog.get("allergic_fruit_dataset_train").thing_classes = ["Apple", "Banana", "Cantaloupe", "Common fig", "Grape", "Grapefruit", "Lemon", "Mango", "Orange", "Peach", "Pear", "Pineapple", "Pomegranate", "Strawberry", "Watermelon"]

# Load the model and metadata
with open(cfg_path, 'rb') as f:
    cfg = pickle.load(f)
cfg.MODEL.WEIGHTS = weights_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = 'cpu'  # Use CPU
predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get("allergic_fruit_dataset_train")

def read_image(file) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(file))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("Error reading image:", e)
        print(traceback.format_exc())
        raise

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        img = read_image(file.read())
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img_out = out.get_image()[:, :, ::-1]
        _, im_arr = cv2.imencode('.jpg', img_out)
        im_bytes = im_arr.tobytes()
        detected_classes = outputs["instances"].pred_classes
        class_names = [metadata.thing_classes[i] for i in detected_classes]

        return jsonify({"detected_fruits": class_names, "image": binascii.hexlify(im_bytes).decode()})
    except Exception as e:
        print("Error processing image:", e)
        print(traceback.format_exc())
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
