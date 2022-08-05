from flask import Flask, request, jsonify
from fastai.vision.all import *
from icevision.all import *
from icevision.models import *
from numpy import asarray
from fastai.learner import load_learner
import torchvision.transforms as transforms
# from fastai.vision import load_image
from flask_cors import CORS,cross_origin
app = Flask(__name__)
CORS(app, support_credentials=True)

# load the learner
learn = load_learner('export.pkl')
classes = learn.dls.vocab

modelPath = Path('/home/andrey/Deep learning/Fish_project/fish/fish.v3i.coco/dataset/model')
checkpoint_path = os.path.join(modelPath / 'Fish_checkpoint.pth')
checkpoint_and_model = model_from_checkpoint(checkpoint_path)



def predict_single(img_file):
    
    model = checkpoint_and_model["model"]
    model_type = checkpoint_and_model["model_type"]
    class_map = checkpoint_and_model["class_map"]
    img_size = checkpoint_and_model["img_size"]

    img_size = checkpoint_and_model["img_size"]
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(img_size), tfms.A.Normalize()])
    original_image = img_file
    img_file = load_image(img_file)

    preds = model_type.end2end_detect(img_file, valid_tfms, model, class_map=class_map, detection_threshold=0.5)
    # transform = transforms.Compose([
    #     transforms.PILToTensor()
    # ])
 
    result = [];
    for bbox in preds['detection']['bboxes']:
        img1 = img_file.crop(bbox.xyxy)
        # img1 = tensor(img1)
        numpydata = asarray(img1)
        prediction = learn.predict(numpydata)
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
    return jsonify(predict_single(request.files['image']))

if __name__ == '__main__':
    app.run()





    # def predict_single(img_file):
    # 'function to take image and return prediction'
    # img_file = load_image(img_file)
    # prediction = learn.predict(tensor(img_file))
    # probs_list = prediction[2].numpy()
    # return {
    #     'category': classes[prediction[1].item()],
    #     'probs': {c: round(float(probs_list[i]), 5) for (i, c) in enumerate(classes)}
    # }


        #     # img1 = tensor(img1)
        # transform = transforms.Compose([
        #     transforms.PILToTensor()
        # ])
        
        # img_tensor = transform(img1)