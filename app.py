from flask import Flask,render_template, jsonify,request
# import torchvision.models as models
import numpy as np
import torch 
from skimage.transform import resize
from model import MRNet
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
import base64

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
    patient, images =load_transform_image(patient)
    fig = Figure(figsize=(10,10))

    # columns = 4
    # rows = 5
    # for i in range(1,images.shape[0]):
    #     im = images[i]
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(im[0], cmap="gray")
    # # ax = fig.subplots()
    # # ax.imshow(images[1][0], cmap="gray")
    # plt.show()

    ax=fig.subplots(ncols=4,nrows=4)
    
    i = 0
    for row in ax:
        for col in row:
            col.imshow(images[i][0], cmap="gray")
            col.axis('off')
            i+=1
    plt.show()
    print(i)
     # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    img = base64.b64encode(buf.getbuffer()).decode("ascii")
    # print(patient.shape)
    print(images.shape)
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

    data = jsonify({'pred': str(pred),"label":str(label)}) 
    data1 = {'pred': str(pred), "label": str(label), "img": img}
    print("! "+ str(data1))
    print(str(data))
    # return jsonify({'pred': str(pred),"label":str(label)})
    return render_template("classification_result.html", data=data1)


def load_transform_image(patient):
    
    vol=np.load(patient).astype(np.int32)
    vol=np.transpose(vol)
    vol = resize(vol, (224, 224))
    vol=np.transpose(vol)
    vol = np.stack((vol,)*3, axis=1)
    vol_image = vol
    vol_tensor = torch.FloatTensor(vol)
    return vol_tensor, vol_image


def get_model():
    model=MRNet()
    state_dict = torch.load("/mnt/g/Grad Projects/MRNet-interface/model weights/f1score_val0.7155_train0.8096_epoch14", map_location=(torch.device('cpu')))
    model.load_state_dict(state_dict)
    return model


@app.route("/image captioning")
def image_captioning():
    return render_template("image captioning.html")


@app.route('/generate_caption', methods=['POST'])
def generate():
    print("CAPTIOOOOOOOOOOOON")
    patient = request.files['file']
    sentence = get_caption(patient)
    print(sentence)
    return render_template("classification_result.html")

if __name__ == "__main__":
    app.run(debug=True)