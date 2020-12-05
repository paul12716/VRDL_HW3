import cv2
from mrcnn import model as modellib, utils
from mrcnn.config import Config
from pycocotools.coco import COCO
import json
from pycocotools import mask as maskUtils

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    STEPS_PER_EPOCH = 100
    
    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 21   # COCO has 80 classes


class InferenceConfig(CocoConfig):
  # Set batch size to 1 since we'll be running inference on
  # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
  DETECTION_MIN_CONFIDENCE = 0
config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", config=config, model_dir='logs/')
model_path = model.find_last()
model.load_weights(model_path, by_name=True)

cocoGt = COCO("../test.json")

coco_dt = []

for imgid in cocoGt.imgs:
  image = cv2.imread("../test_images/" + cocoGt.loadImgs(ids=imgid)[0]['file_name'])[:,:,::-1] # load image
  # masks, categories, scores = model.detect(image) # run inference of your model
  r = model.detect([image], verbose=0)[0]
  masks = r["masks"].astype(np.uint8)
  categories = r["class_ids"]
  scores = r["scores"]
  n_instances = len(scores)    
  if len(categories) > 0: # If any objects are detected in this image
    for i in range(n_instances): # Loop all instances
      # save information of the instance in a dictionary then append on coco_dt list
      mask = masks[:, :, i]
      pred = {}
      pred['image_id'] = imgid # this imgid must be same as the key of test.json
      pred['category_id'] = int(categories[i])
      pred['segmentation'] = binary_mask_to_rle(mask)
      # pred['segmentation'] = maskUtils.encode(np.asfortranarray(mask)) # save binary mask to RLE, e.g. 512x512 -> rle
      pred['score'] = float(scores[i])
      coco_dt.append(pred)

with open("0860903.json", "w") as f:
    json.dump(coco_dt, f)