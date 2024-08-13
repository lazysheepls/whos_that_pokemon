from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.datasets import load_coco_json
from detectron2.data import MetadataCatalog
import time, cv2, os, random

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
COCO_JSON_FILE_NAME="_annotations.coco.json"

# Predict
dataset_dicts = load_coco_json(DATASET_DIR + "valid/" +COCO_JSON_FILE_NAME, DATASET_DIR + "valid", "pokemon_valid")
pokemon_metadata = MetadataCatalog.get("pokemon_valid")

for d in random.sample(dataset_dicts, 3 if len(dataset_dicts)>3 else len(dataset_dicts)):    
    # im = cv2.imread(d["file_name"])
    im = cv2.imread("../../datasets/pokemon/" + "3.png")
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=pokemon_metadata, 
                   scale=0.5, 
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image_name = str(round(time.time()*1000))+'.jpg'
    cv2.imshow(image_name, out.get_image()[:, :, ::-1])
    if cv2.waitKey(0) == 27:
        break  # esc to quit
    cv2.destroyAllWindows()