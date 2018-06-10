import os
import sys
import random
import math
import numpy as np
import skimage.i
import matplotlib
import Pyro4
from PIL import Image
from urllib import request  
import matplotlib.pyplot as plt
from skimage.measure import find_contours

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

@Pyro4.expose
class ForecastImage(object):
    def get_edges(self,urlstr):
        import keras
        keras.backend.clear_session() 
        path =os.path.dirname(os.getcwd())+"/images/test.jpg"
        request.urlretrieve(urlstr, path)
        
        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
        config.display()

        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                    'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                    'teddy bear', 'hair drier', 'toothbrush']



        # Load a random image from the images folder
        file_names = next(os.walk(IMAGE_DIR))[2]
        image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

        # Run detection
        results = model.detect([image], verbose=1)

        r = results[0]
        masks =r['masks']
        print(r['class_ids'])
        typelist = r['class_ids']
        namelist = []
        for i in typelist:
            namelist.append(class_names[i])
        N =r['rois'].shape[0]
        list_dic = {}
        for i in range(N):
            if masks is not None:
                mask = masks[:, :, i]
                # Mask Polygon
                # Pad to ensure proper polygons for masks that touch image edges.
                padded_mask = np.zeros(
                    (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = mask
                contours = find_contours(padded_mask, 0.5)
                templist = []
                for verts in contours:
                    verts = np.fliplr(verts) - 1 
                    for m in range(len(verts)):
                        list_tem = verts[m]
                        templist.append(list_tem[0])
                        templist.append(list_tem[1])
            list_dic[i]=templist  
        returndic ={}
        returndic[0]=namelist
        returndic[1]=list_dic  
        return(returndic)

daemon = Pyro4.Daemon()                # make a Pyro daemon
ns = Pyro4.locateNS()                  # find the name server
uri = daemon.register(ForecastImage)   # register the greeting maker as a Pyro object
ns.register("Edgeforecast", uri)   # register the object with a name in the name server
print("Ready.")
daemon.requestLoop()                   # start the event loop of the server to wait for calls
