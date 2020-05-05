from flask import Flask,render_template, jsonify
import torchvision.models as models
import torch 

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/classifcation")
def classifcation():
    return render_template("classification.html")

@app.route('/predict', methods=['POST'])
def predict():
    densenet = models.densenet161(pretrained=True)
    # file_ = request.files['file']
    print(file_)
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})


@app.route("/image captioning")
def image_captioning():
    return render_template("image captioning.html")


if __name__ == "__main__":
    app.run(debug=True)