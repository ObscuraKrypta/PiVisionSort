# Metal-Color-Detection


# PiVisionSort: Intelligent Material Recognition System Using Color Detection

PiVisionSort is a cost-effective and efficient image processing and machine learning-based system designed to detect and classify different metal materials (Aluminum, Copper, and Brass) on a conveyor belt using their color features in real-time. This project aims to enhance the material recognition process in industrial recycling settings using Raspberry Pi and a simple camera setup.

---

##  Project Objectives

- Improve the accuracy and speed of metal detection in recycling industries.
- Provide a low-cost, portable, and hardware-friendly solution using Raspberry Pi.
- Classify metallic particles based on color in HSV space using KMeans and Decision Tree.

---

##  Core Features

- **Real-Time Video Processing** using OpenCV.
- **Image Preprocessing** with grayscale conversion, Canny edge detection, Gaussian blur, and contour extraction.
- **Color Clustering** using KMeans to find dominant colors in detected contours.
- **Classification** using a trained Decision Tree based on HSV values to label materials.
- **Live Labeling** on video frames with annotated object types: `aluminum`, `copper`, or `messing (brass)`.

---

##  Technologies Used

- Python 3
- OpenCV
- scikit-learn (KMeans, Decision Tree)
- NumPy, Pandas, Matplotlib
- Raspberry Pi (tested)
- ELP Camera or any USB webcam

---

##  Dataset

The project uses an HSV-based labeled dataset stored in `hsv_data.csv` where:
- `h`, `s`, `v`: HSV components of color
- `label`: integer label (0: aluminum, 1: copper, 2: messing)

You can extend or modify this file for retraining or improving accuracy.

---

##  How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/ObscuraKrypta/Metal-Color-Detection.git
   cd Metal-Color-Detection

