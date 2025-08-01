from flask import Flask, request, send_file, render_template
import cv2
import os
from roboflow import Roboflow

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Roboflow setup
rf = Roboflow(api_key="48mujrRECZ7HD2H20S5m")
project = rf.workspace("bytebhajis").project("custom-workflow-object-detection-dvean")
model = project.version("2").model

# 🔹 Route: Landing page
@app.route("/")
def home():
    return render_template("index.html")

# 🔹 Route: Upload form page
@app.route("/upload", methods=["GET"])
def upload_page():
    return render_template("upload.html")

# 🔹 Route: Prediction logic
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]
    if file.filename == "":
        return "No file selected", 400

    # Save uploaded image
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded.jpg")
    file.save(image_path)

    # Resize image for Roboflow
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (640, 640))
    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], "temp.jpg")
    cv2.imwrite(temp_path, image_resized)

    # Run inference
    result = model.predict(temp_path, confidence=0.3).json()

    # Draw boxes
    for pred in result.get("predictions", []):
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        class_name = pred["class"]
        confidence = pred["confidence"]
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(image_resized, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save and return result
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], "output.jpg")
    cv2.imwrite(output_path, image_resized)
    return send_file(output_path, mimetype='image/jpeg')

# ✅ Only one app.run block!
if __name__ == "__main__":
    app.run(debug=True)