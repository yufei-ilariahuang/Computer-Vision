import cv2
import torch
import torchvision.transforms as transforms
import argparse
from torchvision import models
import torch.nn as nn
from PIL import Image
import time


class FPSCalculator:
    def __init__(self, update_interval = 1):
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.update_interval = update_interval

    def update(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time

        if elapsed_time > self.update_interval:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

        return self.fps


def load_label_mapping(file_path):
    idx_to_class = {}
    with open(file_path, 'r') as f:
        for line in f:
            idx, class_name = line.strip().split(',')
            idx_to_class[int(idx)] = class_name
    return idx_to_class

def main():
    # read model path from argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the model')
    args = parser.parse_args()
    model_path = args.model_path
    
    if not model_path.endswith(".pt"):
        print("Invalid model file")
        return
    
    # load the model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")), )
    model.eval()

    # load label mapping
    idx_to_class = load_label_mapping("./label_mapping.txt")

    # define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # open webcam
    cap = cv2.VideoCapture(0)
    frame_count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    output = ""
    confidence_threshold = 0.6

    fps_calculator = FPSCalculator()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Please check your camera connection")
            break

        # predict 1 time each 10 frames
        frame_count += 1
        if frame_count % 10 == 0:

            # preprocess the frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transform(img).unsqueeze(0).to(device)

            # make prediction
            with torch.no_grad():
                outputs = model(img)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # print confidence for each label
                print(f"Confidence: {confidence.item()}, Predicted: {idx_to_class[predicted.item()]}")

                if confidence.item() > confidence_threshold:
                    output = "Class: {}, Confidence Level: {:.2f}".format(idx_to_class[predicted.item()], confidence.item())
                else:
                    output = "No target object detected"

        # display the result on frame
        cv2.putText(frame, output, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # update fps calculator
        fps = fps_calculator.update()

        # Write FPS
        cv2.putText(frame, "FPS = {:.2f}".format(fps), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # show the frame
        cv2.imshow('Live Classification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

