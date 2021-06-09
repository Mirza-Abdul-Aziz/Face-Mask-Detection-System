import os
import numpy as np
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2
import pickle

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

model = pickle.load(open('RF_trained_model.pkl', 'rb'))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        userimage = cv2.imread( 'static/uploads/{}'.format(filename))
        userinput_img_resized = cv2.resize(userimage, (224, 224))
        userinput_img_grayscale = cv2.cvtColor(
        userinput_img_resized, cv2.COLOR_BGR2GRAY)
        userimg_array = np.array(userinput_img_grayscale, dtype='float32')
        userimg_array = userimg_array.flatten()
        userimage = userimg_array/255
        userimage = userimage.reshape(1, -1)
        predicted_recurrence = model.predict(userimage)

        if(predicted_recurrence == 1):
                prediction = "Without-Mask"
        if(predicted_recurrence == 0):
                prediction = "With-Mask"
        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename, prediction= prediction)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
if __name__ == "__main__":
    app.run(debug = True)