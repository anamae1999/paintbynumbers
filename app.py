import cv2
import numpy as np
import os

def posterize_and_generate_outline(image_path, output_prefix="output", k=8):
    # Load and resize image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be loaded. Check the file path.")

    image = cv2.resize(image, (600, int(image.shape[0] * (600 / image.shape[1]))))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Posterize using K-means
    blurred = cv2.GaussianBlur(image_rgb, (5, 5), 0)
    Z = blurred.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    clustered = centers[labels.flatten()].reshape(image_rgb.shape)
    label_map = labels.reshape(image_rgb.shape[:2])

    # Create white canvas for outlines
    outline_canvas = np.ones(image_rgb.shape[:2], dtype=np.uint8) * 255

    for i in range(k):
        mask = (label_map == i).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(outline_canvas, str(i + 1), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
            cv2.drawContours(outline_canvas, [cnt], -1, 0, 1)

    # Save both outputs
    posterized_file = f"{output_prefix}_posterized.png"
    outline_file = f"{output_prefix}_outline.png"

    cv2.imwrite(posterized_file, cv2.cvtColor(clustered, cv2.COLOR_RGB2BGR))
    cv2.imwrite(outline_file, outline_canvas)

    print("✅ Posterized image saved to:", posterized_file)
    print("✅ Outline image saved to:", outline_file)

# Example usage
posterize_and_generate_outline("your_image.jpg", output_prefix="paint_by_number", k=8)
