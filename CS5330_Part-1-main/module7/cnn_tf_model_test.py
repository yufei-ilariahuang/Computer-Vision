import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Specify the model file path and the input image file path
model_path = "scene_classifier_model.h5"
image_path = "./3scenes/coast/coast_bea1.jpg"  # Update this with the path to your test image

# Load the trained model
print("[INFO] loading model...")
model = load_model(model_path)

# Load and preprocess the input image
print("[INFO] loading and preprocessing image...")
image = cv2.imread(image_path)
image = cv2.resize(image, (32, 32))
image = image.astype("float32") / 255.0
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Make predictions
print("[INFO] making predictions...")
predictions = model.predict(image)
predicted_class = np.argmax(predictions, axis=1)[0]

# Assuming the same LabelBinarizer is used to encode the labels
class_labels = ['coast', 'forest', 'highway']
print(f"[INFO] predicted class: {class_labels[predicted_class]}")

# Add text to the image
text = f"Pred: {class_labels[predicted_class]}"
output_image = cv2.imread(image_path)
cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

# Display the input image with prediction
cv2.imshow("Output Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
