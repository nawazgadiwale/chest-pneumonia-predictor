# 🫁 Chest X-ray Pneumonia Detection

This is a Flask-based web application that allows users to upload chest X-ray images and uses a deep learning model (ResNet50) to detect whether the X-ray shows signs of **Pneumonia** or is **Normal**.

---

## 🔍 Features

- Upload a chest X-ray image from your device
- Pretrained ResNet50 model for binary classification (NORMAL vs PNEUMONIA)
- Image preprocessing with `torchvision.transforms`
- Flask API backend
- Frontend UI to upload and view prediction result
- CORS-enabled for cross-origin requests

---

## 📁 Project Structure

project/
├── app.py # Flask app
├── model/
│ ├── pneumonia_detect.py # Prediction logic
│ └── saved_models/
│ └── pneumonia_model.pth # Trained PyTorch model
├── templates/
│ └── x-ray.html # Frontend HTML
├── static/
│ └── styles.css # Optional CSS
├── uploads/ # Temp folder for uploaded images
└── README.md

yaml
Copy
Edit

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- torchvision
- Flask
- Flask-CORS
- PIL (Pillow)

You can install dependencies using:

```bash
pip install -r requirements.txt
Example requirements.txt:

text
Copy
Edit
torch
torchvision
flask
flask-cors
pillow
🚀 How to Run the App
Clone the repository:


git clone https://github.com/yourusername/pneumonia-xray-flask.git
cd pneumonia-xray-flask
Ensure the model file is in place:

Place your trained model as:


model/saved_models/pneumonia_model.pth
Start the Flask server:


python app.py
Visit the web app:

Open in browser:

http://127.0.0.1:5000


🧠 Model Details
Architecture: ResNet50

Fine-tuned for binary classification

Input image size: 224x224

Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

📷 Sample Output
Image Uploaded	Prediction	Confidence
xray_001.jpg	PNEUMONIA	94.2%
xray_002.jpg	NORMAL	88.6%

🛠 Future Improvements
Grad-CAM heatmap visualizations

PDF report generation

Drag-and-drop image support

Docker containerization

REST API documentation (Swagger/Postman)

🤝 Contributing
Feel free to open issues or pull requests if you'd like to contribute!

📄 License
This project is open-source under the MIT License.

👨‍⚕️ Made with ❤️ for Medical AI
