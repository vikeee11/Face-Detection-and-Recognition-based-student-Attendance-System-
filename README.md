# Face-Detection-and-Recognition-based-student-Attendance-System-

📌 Overview

This project is a Face Detection and Recognition based Student Attendance System developed as part of the final year B.E. Electronics and Telecommunication Engineering program at SSGMCE, Shegaon.

The system automates attendance recording using image processing and machine learning techniques. It leverages OpenCV, TensorFlow, TFlearn, and LBPH (Local Binary Pattern Histogram) for robust and accurate face recognition. Attendance is logged automatically into an Excel file and stored in a MySQL database with real-time feedback.

🚀 Features

👤 Face Detection & Recognition using OpenCV and LBPH

📸 Dataset Generation with live image capture

🧑‍💻 User Registration (Name, Age, Address linked to database)

📊 Attendance Logging in Excel with timestamps

🔔 Audio Feedback with user name confirmation

🖥️ GUI Interface built with Tkinter for easy interaction

⏳ Automated Attendance Scheduling using Python’s scheduler

🛠️ Tech Stack

Programming Language: Python

Libraries/Frameworks:

OpenCV (Image Processing & Face Detection)

TensorFlow & TFlearn (Deep Learning)

LBPH (Feature Extraction & Recognition)

Tkinter (GUI)

Schedule & Threading (Automated tasks)

Database: MySQL

Tools Used: PyCharm IDE

Output Storage: Excel Sheets (attendance.xlsx), Image Datasets

📂 Project Structure
Face-Attendance-System/
│── data/                  # Collected face datasets
│── trainer/               # Trained LBPH classifier
│── face_recognition.py    # Main script for recognition
│── dataset_generator.py   # Script for capturing faces
│── train_classifier.py    # Script to train model
│── attendance.xlsx        # Attendance log file
│── requirements.txt       # Dependencies
│── README.md              # Documentation

⚙️ Installation & Setup

1️⃣ Clone Repository
git clone https://github.com/vikeee11/Face-Detection-and-Recognition-based-student-Attendance-System-.git
cd face-attendance-system

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Setup Database

Create a MySQL database and table for storing student details:

CREATE DATABASE attendance_system;
USE attendance_system;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    age INT,
    address VARCHAR(255)
);

4️⃣ Run Modules

Generate Dataset:

python dataset_generator.py


Train Model:

python train_classifier.py


Start Attendance Recognition:

python face_recognition.py

📸 Screenshots

GUI for Registration & Training

Real-time Face Detection

Attendance Excel Sheet

(Add screenshots here after running project)

🔮 Future Scope

Improve accuracy under low-light & occlusion conditions

Deploy on Raspberry Pi / IoT devices for portability

Mobile application integration for remote attendance

Cloud-based scaling & ERP system integration

AI-powered liveness detection to prevent spoofing


📜 Publication

This work was accepted for presentation at the
National Level Students Conference IEEE TECHNICOKNOCKDOWN-2025 (TKD-25).
