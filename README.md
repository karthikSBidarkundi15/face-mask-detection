# Face Mask Detection using CNN

## Overview

This project detects whether a person is wearing a face mask in real time using a webcam. It uses Computer Vision and a Convolutional Neural Network (CNN) model to classify faces as **Mask** or **No Mask**. The system captures live video, detects faces, and displays predictions instantly.

---

## Technologies Used

- Python
- OpenCV
- TensorFlow
- Keras
- NumPy
- CNN (Convolutional Neural Network)

---

## Dataset

Face Mask Dataset from Kaggle.

Download and place the dataset inside:

```text
dataset/
├── with_mask/
└── without_mask/
```

---

## Features

- Real-time face detection
- Face mask classification
- Webcam integration
- CNN-based image classification
- Live prediction display

---

## Project Structure

```text
Face-Mask-Detection/
│
├── dataset/
├── train_mask_detector.py
├── detect_mask.py
├── requirements.txt
└── README.md
```

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Train the Model

```bash
python train_mask_detector.py
```

### Run Real-Time Detection

```bash
python detect_mask.py
```

---

## Applications

- Public safety monitoring
- Healthcare facilities
- Educational institutions
- Offices and workplaces
- Smart surveillance systems

---

## Future Improvements

- Improve accuracy with larger datasets
- Detect multiple faces simultaneously
- Deploy on Raspberry Pi or edge devices
- Add alert and notification systems
- Develop a web-based version

---

## Learning Outcomes

Through this project, I gained experience in:

- Computer Vision
- OpenCV
- Deep Learning Fundamentals
- CNN-based Image Classification
- Real-Time Video Processing
- TensorFlow Integration

---

## Acknowledgements

This project was developed for learning purposes using publicly available datasets, tutorials, and open-source resources to understand Computer Vision and Deep Learning concepts.

---

## Author

Karthik Bidarkundi

Artificial Intelligence & Machine Learning Engineering Student
