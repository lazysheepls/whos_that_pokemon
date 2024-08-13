from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
import os
import supervision as sv
import numpy as np

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg = get_cfg()  
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # Load the config file used for training  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (gengar). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Set path to the trained weights  
cfg.MODEL.DEVICE = "cuda"  # Use GPU; for CPU, set to "cpu"  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set the detection threshold  
predictor = DefaultPredictor(cfg)

DATASET_DIR="../../datasets/pokemon/v5/"
SOURCE_VID_DIR="../../datasets/pokemon/video/"
COCO_JSON_FILE_NAME="_annotations.coco.json"
CLASS_NAMES=["Gengar"]

# supervision callback function
def process_frame(frame: np.ndarray, _) -> np.ndarray:
    result = predictor(frame)
    detections = sv.Detections.from_detectron2(result)
    
    box_annotator = sv.BoxAnnotator(thickness=4, color=sv.Color.BLUE)
    mask_annotator = sv.MaskAnnotator(color=sv.Color.BLUE)
    label_annotator = sv.LabelAnnotator(color=sv.Color.BLUE)

    labels = [f"{CLASS_NAMES[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _, _ in detections]
    detections.class_id
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = mask_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame,detections=detections, labels=labels)

    return frame

VIDEO_PATH=SOURCE_VID_DIR + "gengar2.mp4"
TARGET_PATH=SOURCE_VID_DIR + "gengar_result2.mp4"
sv.process_video(source_path=VIDEO_PATH, target_path=TARGET_PATH, callback=process_frame)