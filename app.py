from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

try:
    # Load the quantized TFLite model
    interpreter = tf.lite.Interpreter(model_path="D:/rice_api/model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load class labels
    with open("D:/rice_api/labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]

    print("✅ Model and labels loaded successfully")

except Exception as e:
    print(f"❌ Error loading model/labels: {e}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']

        # 🖼️ Load and preprocess image as UINT8 for quantized model
        image = Image.open(image_file).convert('RGB').resize((224, 224))
        image_np = np.array(image, dtype=np.uint8)
        image_np = np.expand_dims(image_np, axis=0)  # Shape: [1, 224, 224, 3]

        # Set tensor and invoke model
        interpreter.set_tensor(input_details[0]['index'], image_np)
        interpreter.invoke()

        # Get prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        max_index = int(np.argmax(output_data))
        confidence = float(output_data[max_index])
        prediction = labels[max_index]

        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })

    except Exception as e:
        print(f"❌ Internal error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5002)
