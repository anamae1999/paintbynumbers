from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests (e.g., Shopify, HTML forms)

@app.route("/generate", methods=["POST"])
def generate():
    if 'image' not in request.files:
        return jsonify({"error": "Missing 'image' in form data"}), 400

    try:
        file = request.files['image']
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Resize for simplicity
        img = cv2.resize(img, (400, int(img.shape[0] * (400 / img.shape[1]))))

        # K-means clustering to reduce colors
        Z = img.reshape((-1, 3)).astype(np.float32)
        _, labels, centers = cv2.kmeans(Z, 8, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)

        labels = labels.reshape(img.shape[:2])

        # Create a blank white canvas
        canvas = np.ones(img.shape[:2], dtype=np.uint8) * 255

        for i in range(8):
            mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) < 100:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(canvas, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
                cv2.drawContours(canvas, [cnt], -1, 0, 1)

        # Save and send back the output image
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(temp.name, canvas)
        return send_file(temp.name, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500
