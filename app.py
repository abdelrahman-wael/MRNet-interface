from flask import Flask,render_template, jsonify,request
# import torchvision.models as models
import numpy as np
import torch 
from skimage.transform import resize
from model import MRNet

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/classifcation")
def classifcation():
    return render_template("classification.html")

@app.route('/predict', methods=['POST'])
def predict():
    
    patient = request.files['file']
    patient=load_transform_image(patient)
    # print(patient.shape)
    model = get_model()
    pred = model(patient)
    pred=pred.item()
    print()
    print()
    print()
    print(pred)
    if(pred>0.5):
        label = "found tear"
    else:
        label="no tear"

    return jsonify({'pred': str(pred),"label":str(label)})


def load_transform_image(patient):
    
    vol=np.load(patient).astype(np.int32)
    vol=np.transpose(vol)
    vol = resize(vol, (224, 224))
    vol=np.transpose(vol)
    vol = np.stack((vol,)*3, axis=1)
    vol_tensor = torch.FloatTensor(vol)
    return vol_tensor


def get_model():
    model=MRNet()
    state_dict = torch.load("/home/abdel/Documents/web developlement/flusk knee MRI/model weights/f1score_val0.7155_train0.8096_epoch14", map_location=(None))
    model.load_state_dict(state_dict)
    return model


@app.route("/image captioning")
def image_captioning():
    return render_template("image captioning.html")


if __name__ == "__main__":
    app.run(debug=True)