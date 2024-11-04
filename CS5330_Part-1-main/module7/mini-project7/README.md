# Mini-project 7: Live Classifier
The goal of mini-project7 is to develop a real-time object classification system using a Convolutional Neural Network (CNN). The system will classify objects in live video feed, frame by frame, into predefined classes such as remote control, cell phone, TV, and at least one additional class of your choosing. The project involves collecting and preprocessing a dataset, designing and training a CNN model, applying the model to a live webcam feed, and displaying the classified objects in real-time.

# Team Members
- Siyi Gao
- Yufei Huang
- Xuedong Pan
- Tongle Yao

# Install Dependencies
`pip install -r requirements.txt`
## Packages We Used (and Versions)
- numpy==2.1.2
- opencv-python==4.10.0.84
- pillow==11.0.0
- torch==2.5.1
- torchaudio==2.5.1
- torchvision==0.20.1
- seaborn==0.13.2
- scikit-learn==1.5.2
- matplotlib==3.9.2

# Dataset Information
We collected 218 images for each type of object we wanted to discuss, including cellphones, TVs, remote controls, and Mugs. Those images are mostly from websites such as Google Images and Facebook Marketplace. We captured only a small portion of the images. We will apply data augmentation techniques in the model training stage.

Dataset Link: https://drive.google.com/file/d/1UMMdDyqrA4qrosjIxn3oxGvdWTA9vwMw/view?usp=sharing

# CNN Model
We tried different ways to train our CNN model, and we found that if we use our own 4-layer CNN, the maximum test accuracy that we can achieve is around 70%, with a test loss of roughly 0.8. Hence, we choose to train a model on top of a base model. Therefore, our final CNN architecture is based on ResNet-18, featuring multiple convolutional layers with ReLU activation functions and skip connections to enhance gradient flow. The final layer is a fully connected layer that outputs class probabilities using softmax via cross-entropy loss. Data augmentation techniques, such as random flips and color jitter, are applied to improve generalization. The model is trained using the Adam optimizer with a learning rate of 0.001, iterating through multiple epochs while evaluating performance on training and validation sets.

Model Link: https://drive.google.com/file/d/15mg2gTnMbf-7Yuy8bFD_EG-48NmOP-KM/view?usp=sharing

# Model Evaluation on Test Dataset
- Accuracy: 96.95%
- Loss: 0.1313
- Precision: 0.9015
- Recall: 0.9000
- F1-score: 0.9006
- Confusion Matrix:
[![confusion-matrix.png](https://i.postimg.cc/7hNtFtqh/confusion-matrix.png)](https://postimg.cc/jnLhNXwY)

- The testing accuracy and test loss are pretty high on the test dataset, but when we tested it on the real-world scenario, we found that sometimes it was not very accurate, especially in low-light or overexposure scenarios. We believe it is because of the limited dataset size, and it would be better if we could collect more data in each class.

# If You Want to Train a Model
`python train_model.py -d <dataset_path> -m <model_save_path>`

OR

`python train_model.py --data_dir <dataset_path> --model_path <model_save_path>`

Example:

`python train_model.py -d ./dataset -m ./model_best.pt`

# If You Want to Test a Model Using Webcam
`python run_classifier.py -m <model_path>`

OR

`python run_classifier.py --model_path <model_path>`

Example:

`python run_classifier.py -m model_best.pt `

# Note
- Before you run the `run_classifier.py` program, make sure both the model you want to use (i.e. `model_best.pt`) and its corresponding `label_mapping.txt` are under the current program directory. (ps: the default `label_mapping.txt` in the repository is for `model_best.pt`)
Make sure your network is stable. When you first run either the `train_model.py` or `run_classifier.py` program, it will need to download the base model, ResNet18, from the PyTorch server.
- When the camera feed is displayed, click 'q' to quit the program.

# Video Demonstration
Link: https://drive.google.com/file/d/1VzX7IeUj6PWo06IhRn0j22FkWui1J9Qh/view?usp=sharing
