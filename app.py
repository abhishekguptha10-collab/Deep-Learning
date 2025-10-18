from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle
import cv2
from PIL import Image, ImageChops, ImageEnhance
import os
import itertools
app = Flask(__name__)

dependencies = {
    'auc_roc': AUC
}

class_names = {
0: 'Tampered (Fake)',
1: 'Authentic (Real)',

}

model = load_model('casia.h5')
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

image_size = (200, 200)

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0


 
@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')   
    
@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		# model = request.form['model']
		# print(model)
	
		img_path = "static/tests/" + img.filename	
		img.save(img_path)
		#plt.imshow(img)

		

		
		image = prepare_image(img_path)
		image = image.reshape(-1, 200, 200, 3)
		y_pred = model.predict(image) 
		y_pred_class = np.argmax(y_pred, axis = 1)[0]
		predict_result =  class_names[y_pred_class]
        #  print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')
	return render_template("prediction.html", prediction = predict_result, img_path = img_path)

@app.route("/performance")
def performance():
	return render_template('performance.html')   

@app.route("/chart")
def chart():
	return render_template('chart.html')   
	
if __name__ =='__main__':
	app.run(debug = True)


	

	


