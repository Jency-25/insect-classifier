import os
import json
import base64
import numpy as np
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins (GitHub Pages frontend -> Render backend)

# --- Load TFLite Model and Labels ---
print("Loading TFLite model and labels...")
MODEL_PATH = 'model/insect_model.tflite'
LABELS_PATH = 'model/labels.npy'

interpreter = None
labels = []
insect_db = {}
TARGET_SIZE = (224, 224)
input_index = None
output_index = None

try:
    # Use tflite_runtime if available (smaller), else fall back to tensorflow.lite
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        print("Using tflite_runtime interpreter")
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        print("Using tensorflow.lite interpreter")

    interpreter.allocate_tensors()

    # Get input/output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    # Determine target image size from model input
    shape = input_details[0]['shape']  # e.g. [1, 224, 224, 3]
    TARGET_SIZE = (int(shape[1]), int(shape[2]))

    labels = np.load(LABELS_PATH, allow_pickle=True)

    with open('insect_details.json', 'r', encoding='utf-8') as f:
        insect_db = json.load(f)

    print(f"TFLite model loaded! Input size: {TARGET_SIZE}, Labels: {len(labels)}")

except Exception as e:
    print(f"ERROR loading model: {e}")
    interpreter = None


@app.route('/')
def home():
    """Serve the Home page."""
    return render_template('home.html', species_json=json.dumps(insect_db))

@app.route('/scanner')
def scanner():
    """Serve the Scanner interface page."""
    return render_template('scanner.html')

@app.route('/api/species')
def api_species():
    """Return the full species database as JSON."""
    return jsonify(insect_db)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image analysis using TFLite interpreter."""
    if interpreter is None:
        return jsonify({'success': False, 'error': 'Model failed to load on the server.'}), 500

    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No valid image provided'}), 400

        # Decode base64 image from frontend canvas
        image_data = data['image']
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]

        image_bytes = base64.b64decode(image_data)

        # Open with PIL and convert to RGB
        img = Image.open(BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize to model's expected input size
        img = img.resize(TARGET_SIZE)
        img_array = np.array(img).astype('float32')
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Run inference with TFLite interpreter
        interpreter.set_tensor(input_index, img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_index)

        # Get top prediction
        class_index = np.argmax(predictions[0])
        confidence_val = float(predictions[0][class_index])
        pred_class_name = str(labels[class_index])

        # Lookup details from DB
        details = insect_db.get(pred_class_name, "Details not found for this species.")

        return jsonify({
            'success': True,
            'prediction': pred_class_name,
            'confidence': confidence_val * 100,
            'details': details
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Flask Server...")
    port = int(os.environ.get("PORT", 5000))
    print(f"Server running at: http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
