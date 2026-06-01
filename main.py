from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained breast cancer model
model = load_model('models/breast_cancer_clean.keras', compile=False)
# Breast cancer class labels
class_labels = ['benign', 'malignant', 'normal']

# Upload folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Prediction function
def predict_cancer(image_path):
    IMAGE_SIZE = 128

    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    predicted_label = class_labels[predicted_class_index]

    if predicted_label == 'normal':
        return "Normal Tissue", confidence_score
    elif predicted_label == 'benign':
        return "Benign Tumor", confidence_score
    else:
        return "Malignant Tumor", confidence_score


# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        file = request.files['file']

        if file:
            file_location = os.path.join(
                app.config['UPLOAD_FOLDER'],
                file.filename
            )

            file.save(file_location)

            result, confidence = predict_cancer(file_location)

            return render_template(
                'index.html',
                result=result,
                confidence=f"{confidence*100:.2f}%",
                file_path=f'/uploads/{file.filename}'
            )

    return render_template('index.html', result=None)


# Serve uploaded images
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )


if __name__ == '__main__':
    app.run(debug=True)