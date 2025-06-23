from flask import Flask, request, send_file
import cv2, numpy as np, tempfile

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (400, int(img.shape[0] * (400 / img.shape[1]))))

    Z = img.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(Z, 8, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape(img.shape[:2])

    canvas = np.ones(img.shape[:2], dtype=np.uint8) * 255
    for i in range(8):
        mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(canvas, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
            cv2.drawContours(canvas, [cnt], -1, 0, 1)

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp.name, canvas)
    return send_file(temp.name, mimetype='image/png')
