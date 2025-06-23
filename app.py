from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tempfile

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (e.g., from Shopify or localhost)

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Resize & blur
        img = cv2.resize(img, (600, int(img.shape[0] * (600 / img.shape[1]))))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blurred = cv2.GaussianBlur(img_rgb, (5, 5), 0)

        # Posterization (K-means)
        k = 8
        Z = blurred.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        clustered = centers[labels.flatten()].reshape(img.shape)
        label_map = labels.reshape(img.shape[:2])

        # Create outline canvas
        canvas = np.ones(img.shape[:2], dtype=np.uint8) * 255

        for i in range(k):
            mask = (label_map == i).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) < 500:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(canvas, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
                cv2.drawContours(canvas, [cnt], -1, 0, 1)

        # Save and send the result
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(temp.name, canvas)
        return send_file(temp.name, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
