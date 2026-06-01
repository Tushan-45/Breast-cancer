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

app = Flask(__name__)

model = load_model('models/breast_cancer_new.keras', compile=False)
class_labels = ['benign', 'malignant', 'normal', 'unrelated']

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def predict_cancer(image_path):
    IMAGE_SIZE = 128

    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0

    # ── 1. Grayscale check ──────────────────────────────────────────
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    channel_diff = np.max([
        np.mean(np.abs(r - g)),
        np.mean(np.abs(g - b)),
        np.mean(np.abs(r - b))
    ])
    if channel_diff > 0.05:
        return "Please upload a valid breast ultrasound image", 0.0, {}, "invalid"

    # ── 2. Brightness / contrast check ─────────────────────────────
    mean_pixel = np.mean(img_array)
    std_pixel  = np.std(img_array)

    if mean_pixel > 0.85:
        return "Please upload a valid breast ultrasound image", 0.0, {}, "invalid"
    if mean_pixel < 0.05:
        return "Please upload a valid breast ultrasound image", 0.0, {}, "invalid"
    if std_pixel < 0.08:
        return "Please upload a valid breast ultrasound image", 0.0, {}, "invalid"

    # ── 3. Model prediction ─────────────────────────────────────────
    img_array   = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)[0]  # shape: (4,)

    predicted_class_index = np.argmax(predictions)
    confidence_score      = float(predictions[predicted_class_index])
    predicted_label       = class_labels[predicted_class_index]

    all_scores = {class_labels[i]: float(predictions[i]) for i in range(len(class_labels))}

    # ── 4. Unrelated class check ────────────────────────────────────
    if predicted_label == 'unrelated':
        return "Please upload a valid breast ultrasound image", confidence_score, {}, "invalid"

    # ── 5. Entropy check ────────────────────────────────────────────
    entropy     = -np.sum(predictions * np.log(predictions + 1e-9))
    max_entropy = np.log(len(class_labels))
    if entropy / max_entropy > 0.50:
        return "Uncertain image — please consult a specialist", confidence_score, all_scores, "uncertain"

    # ── 6. Confidence gap check (fixes benign↔malignant confusion) ──
    sorted_preds   = np.sort(predictions)[::-1]
    confidence_gap = float(sorted_preds[0] - sorted_preds[1])

    if confidence_gap < 0.20:
        top2_indices = np.argsort(predictions)[::-1][:2]
        top2_labels  = [class_labels[i] for i in top2_indices]
        top2_scores  = {class_labels[i]: float(predictions[i]) for i in top2_indices}
        msg = (
            f"Ambiguous result between '{top2_labels[0]}' and '{top2_labels[1]}' — "
            "please consult a radiologist"
        )
        return msg, confidence_score, top2_scores, "ambiguous"

    # ── 7. Hard confidence threshold ────────────────────────────────
    if confidence_score < 0.85:
        return "Uncertain image — please consult a specialist", confidence_score, all_scores, "uncertain"

    # ── 8. Final result ─────────────────────────────────────────────
    display_scores = {k: v for k, v in all_scores.items() if k != 'unrelated'}

    if predicted_label == 'normal':
        return "Normal Tissue", confidence_score, display_scores, "valid"
    elif predicted_label == 'benign':
        return "Benign Tumor", confidence_score, display_scores, "valid"
    else:
        return "Malignant Tumor", confidence_score, display_scores, "valid"


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

            result, confidence, breakdown, status = predict_cancer(file_location)

            return render_template(
                'index.html',
                result=result,
                confidence=f"{confidence * 100:.2f}%",
                breakdown=breakdown,
                status=status,
                file_path=f'/uploads/{file.filename}'
            )

    return render_template('index.html', result=None)


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)