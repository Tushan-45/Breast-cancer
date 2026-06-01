from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from tensorflow.keras.layers import Dense

# Patch Dense to ignore quantization_config
old_dense_init = Dense.__init__

def patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    old_dense_init(self, *args, **kwargs)

Dense.__init__ = patched_dense_init

# Initialize Flask app
app = Flask(__name__)

# Load trained breast cancer model
model = load_model('models/breast_cancer_new.keras', compile=False)
# Breast cancer class labels
class_labels = ['benign', 'malignant', 'normal', 'unrelated']

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

    # --- Grayscale check ---
    # Ultrasound images are grayscale; color variance across channels should be minimal
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    channel_diff = np.max([
        np.mean(np.abs(r - g)),
        np.mean(np.abs(g - b)),
        np.mean(np.abs(r - b))
    ])
    if channel_diff > 0.05:  # Colorful image → not an ultrasound
        return "Please upload a valid breast ultrasound image", 0

    # --- Brightness / contrast check ---
    mean_pixel = np.mean(img_array)
    std_pixel = np.std(img_array)

    if mean_pixel > 0.85:          # Too bright (blank/white image)
        return "Please upload a valid breast ultrasound image", 0
    if mean_pixel < 0.05:          # Too dark (blank/black image)
        return "Please upload a valid breast ultrasound image", 0
    if std_pixel < 0.08:           # No texture (flat/uniform image)
        return "Please upload a valid breast ultrasound image", 0

    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)

    predicted_class_index = np.argmax(predictions)
    confidence_score = float(np.max(predictions))

    # --- Use the 'unrelated' class your model already predicts ---
    predicted_label = class_labels[predicted_class_index]
    if predicted_label == 'unrelated':
        return "Please upload a valid breast ultrasound image", confidence_score

    # --- Entropy check: reject low-certainty predictions ---
    # High entropy = model is unsure = likely unrelated image
    entropy = -np.sum(predictions * np.log(predictions + 1e-9))
    max_entropy = np.log(len(class_labels))  # worst case entropy
    if entropy / max_entropy > 0.5:          # more than 50% uncertain
        return "Uncertain or unrelated image", confidence_score

    # --- Confidence threshold ---
    if confidence_score < 0.85:
        return "Uncertain or unrelated image", confidence_score

    # --- Return actual diagnosis ---
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)