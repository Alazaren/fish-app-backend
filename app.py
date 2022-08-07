from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin

from fastai.vision.all import *

from PIL import Image
import pathlib
import numpy as np

from fastai.learner import load_learner

from icevision.all import *
from icevision.models import *



# from fastai.vision import load_image
app = Flask(__name__)
CORS(app, support_credentials=True)


# load the classifier
learn = load_learner('export.pkl')
classes = learn.dls.vocab

# load the object detection
modelPath = pathlib.Path('Fish_checkpoint.pth')
checkpoint_and_model = model_from_checkpoint(modelPath)


def predict_single(img_file):
    img_file = Image.open(img_file)
    model = checkpoint_and_model["model"]
    model_type = checkpoint_and_model["model_type"]
    class_map = checkpoint_and_model["class_map"]
    img_size = checkpoint_and_model["img_size"]

    img_size = checkpoint_and_model["img_size"]
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(img_size), tfms.A.Normalize()])
    
    # img_file = get_image_files(img_file)

    preds = model_type.end2end_detect(img_file, valid_tfms, model, class_map=class_map, detection_threshold=0.5)

 
    result = [];
    for bbox in preds['detection']['bboxes']:
        img1 = img_file.crop(bbox.xyxy)
        
        # img_byte_arr = io.BytesIO()
        # img1.save(img_byte_arr, format='PNG')
        # img_byte_arr = img_byte_arr.getvalue()

        prediction = learn.predict(np.asarray(img1))
        probs_list = prediction[2].numpy()
        result.append(
            {
                'bbox': bbox.xywh,
                'category': classes[prediction[1].item()],
                'probs': {c: round(float(probs_list[i]), 5) for (i, c) in enumerate(classes)},
                'originalWidth': img_file.width,
            }
        )
    return result;

# route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(predict_single(request.files['image'].stream))

if __name__ == '__main__':
    app.run()