# from PIL import Image
# from flask import Flask, request, jsonify
# import os 
# from model.pneumonia_detect import predict_image  # make sure this is fixed
# from werkzeug.utils import secure_filename
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/predict', methods=['POST'])
# def predict():
#     print("🛰 Received a request!")

#     if 'image' not in request.files:
#         print("❌ No 'image' in request.files")
#         return jsonify({'error': 'No image uploaded'}), 400

#     image_file = request.files['image']
#     print(f"📁 Received file: {image_file.filename}")

#     if image_file.filename == '':
#         print("❌ Empty filename")
#         return jsonify({'error': 'Empty file'}), 400

#     try:
#         # Optional: Save uploaded file (for debugging/audit)
#         filename = secure_filename(image_file.filename)
#         file_path = os.path.join(UPLOAD_FOLDER, filename)
#         image_file.save(file_path)
#         print(f"✅ Saved uploaded file to {file_path}")

#         # Load image and predict
#         image = Image.open(file_path).convert('RGB')
#         label, confidence = predict_image(image)

#         return jsonify({
#             'label': label,
#             'confidence': round(confidence, 2)
#         })

#     except Exception as e:
#         print(f"⚠️ Error: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)

from PIL import Image
from flask import Flask, request, jsonify, make_response,render_template
import os
from model.pneumonia_detect import predict_image
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/')
def render_html():
    return render_template('x-ray.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(file_path)

        label, confidence = predict_image(file_path)

        result = {
            'label': label,
            'confidence': round(confidence * 100, 2)
        }

        response = make_response(jsonify(result), 200)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
