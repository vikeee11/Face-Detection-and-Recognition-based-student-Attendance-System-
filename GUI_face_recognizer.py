import cv2
import tkinter as tk
from tkinter import messagebox
import os
import numpy as np
import mysql.connector
import pandas as pd
from datetime import datetime
import schedule
import threading
import time
window = tk.Tk()
window.title("Face Recognition system")

l1 = tk.Label(window, text="Name", font=("Algerian", 20))
l1.grid(column=0, row=0)
t1 = tk.Entry(window, width=50, bd=5)
t1.grid(column=1, row=0)

l2 = tk.Label(window, text="Age", font=("Algerian", 20))
l2.grid(column=0, row=1)
t2 = tk.Entry(window, width=50, bd=5)
t2.grid(column=1, row=1)

l3 = tk.Label(window, text="Address", font=("Algerian", 20))
l3.grid(column=0, row=2)
t3 = tk.Entry(window, width=50, bd=5)
t3.grid(column=1, row=2)


def train_classifier():
    data_root = "D:/final_project3/data/"  # Root directory containing user ID folders

    if not os.path.exists(data_root):
        messagebox.showerror("Error", "Data directory does not exist!")
        return

    faces = []
    ids = []

    # Loop through each user ID folder
    for user_id in os.listdir(data_root):
        user_path = os.path.join(data_root, user_id)

        if not os.path.isdir(user_path):  # Skip files, only process directories
            continue

        print(f"Processing user ID folder: {user_id}")

        try:
            user_id_int = int(user_id)  # Ensure the folder name is a valid integer
        except ValueError:
            print(f"Skipping non-numeric folder: {user_id}")
            continue

        # Collect all image paths in the current user ID folder
        image_paths = [os.path.join(user_path, f) for f in os.listdir(user_path)
                       if os.path.isfile(os.path.join(user_path, f))]

        for image_path in image_paths:
            try:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
                if img is None:
                    print(f"Skipping unreadable image: {image_path}")
                    continue

                # Extract ID from filename (ensuring format: user_id.img_number.extension)
                filename = os.path.basename(image_path)
                parts = filename.split(".")

                if len(parts) < 2 or not parts[0].isdigit():
                    print(f"Skipping invalid file format: {filename}")
                    continue

                id = int(parts[0])  # Extracting user ID

                if id != user_id_int:  # Ensure ID matches folder name
                    print(f"Skipping mismatched ID in file: {filename}")
                    continue

                faces.append(img)
                ids.append(id)
            except Exception as e:
                print(f"Error processing file {image_path}: {e}")

    if not faces or not ids:
        messagebox.showerror("Error", "No valid images found for training!")
        return

    ids = np.array(ids)

    try:
        # Train the classifier
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)
        clf.write("classifier.xml")

        messagebox.showinfo('Result', 'Training dataset completed successfully!')
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train classifier: {e}")



b1 = tk.Button(window, text="Training", font=("Algerian", 20), bg="orange", fg="red", command=train_classifier)
b1.grid(column=0, row=4)

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

    for (x, y, w, h) in features:
        face_roi = gray_img[y:y + h, x:x + w]  # Extract the face region
        id, confidence_score = clf.predict(face_roi)  # Predict ID

        confidence = int(100 * (1 - confidence_score / 300))  # Convert to percentage

        label = "UNKNOWN"


        try:
            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                passwd="",
                database="authorized_user"
            )
            mycursor = mydb.cursor()

            mycursor.execute("SELECT name FROM users WHERE id = %s", (id,))
            result = mycursor.fetchone()

            if result and confidence > 70:  # Only display name if confidence is above 80%
                label = result[0]

            mycursor.close()
            mydb.close()

        except mysql.connector.Error as e:
            print(f"Database error: {e}")

        # Draw rectangle & label
        rect_color = color if label != "UNKNOWN" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), rect_color, 2)
        cv2.putText(img, f"{label} ({confidence}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, rect_color, 2, cv2.LINE_AA)

    return img


def log_attendance(user_id, user_name):
    """
    Logs the attendance of a recognized user into an Excel sheet.
    """
    file_path = "attendance.xlsx"

    # Get current date and time
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Check if file exists
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df = pd.DataFrame(columns=["User ID", "Name", "Date", "Time"])

    # Avoid duplicate attendance for the same user on the same date
    if ((df["User ID"] == user_id) & (df["Date"] == date)).any():
        return  # Skip logging if already recorded

    # Append new attendance entry
    new_entry = pd.DataFrame([[user_id, user_name, date, time]], columns=["User ID", "Name", "Date", "Time"])
    df = pd.concat([df, new_entry], ignore_index=True)

    # Save to Excel
    df.to_excel(file_path, index=False)
    print(f"Attendance logged for {user_name} at {time}")

def detect_face():
    """
    Detects faces in real-time and logs attendance.
    """
    faceCascade = cv2.CascadeClassifier('haarcascadefrontalfacedefault.xml')
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Unable to access the webcam.")
        return

    while True:
        ret, img = video_capture.read()
        if not ret or img is None:
            print("Failed to capture frame. Exiting...")
            break

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray_img, 1.1, 10)

        for (x, y, w, h) in faces:
            face_roi = gray_img[y:y + h, x:x + w]
            id, confidence_score = clf.predict(face_roi)

            confidence = int(100 * (1 - confidence_score / 300))
            label = "UNKNOWN"


            try:
                mydb = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    passwd="",
                    database="authorized_user"
                )
                mycursor = mydb.cursor()
                mycursor.execute("SELECT name FROM users WHERE id = %s", (id,))
                result = mycursor.fetchone()

                if result and confidence > 70:
                    label = result[0]
                    log_attendance(id, label)  # Log attendance

                mycursor.close()
                mydb.close()

            except mysql.connector.Error as e:
                print(f"Database error: {e}")

            # Draw rectangle & label
            rect_color = (0, 255, 0) if label != "UNKNOWN" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), rect_color, 2)
            cv2.putText(img, f"{label} ({confidence}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, rect_color, 2, cv2.LINE_AA)

        cv2.imshow("Face Detection", img)

        if cv2.waitKey(1) == 13:  # Press Enter to exit
            break

    video_capture.release()
    cv2.destroyAllWindows()



b2 = tk.Button(window, text="Detect the faces", font=("Algerian", 20), bg="green", fg="orange", command=detect_face)
b2.grid(column=1, row=4)


def generate_dataset():
    if t1.get() == "" or t2.get() == "" or t3.get() == "":
        messagebox.showinfo('Result', 'Please provide complete details of the user')
        return

    try:
        # Connect to the MySQL database
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="",
            database="authorized_user"
        )
        mycursor = mydb.cursor()

        # Get the next user ID
        mycursor.execute("SELECT MAX(id) FROM users")
        result = mycursor.fetchone()
        new_id = result[0] + 1 if result[0] else 1  # Increment ID or start with 1

        # Insert the new user record into the database
        sql = "INSERT INTO users (id, name, age, address) VALUES (%s, %s, %s, %s)"
        val = (new_id, t1.get(), t2.get(), t3.get())
        mycursor.execute(sql, val)
        mydb.commit()
        mycursor.close()
        mydb.close()

        # Load Haarcascade face detector
        face_classifier = cv2.CascadeClassifier('haarcascadefrontalfacedefault.xml')

        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                return None
            for (x, y, w, h) in faces:
                return img[y:y + h, x:x + w]  # Return the first detected face

        # Create directory with user ID instead of name
        dataset_dir = f"D:/final_project3/data/{new_id}"
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)

        cap = cv2.VideoCapture(0)
        img_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                continue

            face = face_cropped(frame)
            if face is not None:
                img_id += 1
                face = cv2.resize(face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # Save with correct ID and unique image number
                file_name_path = f"{dataset_dir}/{new_id}.{img_id}.jpg"
                cv2.imwrite(file_name_path, face)

                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Cropped Face", face)

                if cv2.waitKey(1) == 13 or img_id == 200:  # Stop when Enter key (13) is pressed or 200 images collected
                    break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Dataset generation completed successfully!')

    except mysql.connector.Error as e:
        messagebox.showerror("Database Error", f"Failed to insert user: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")



b3 = tk.Button(window, text="Generate dataset", font=("Algerian", 20), bg="pink", fg="black", command=generate_dataset)
b3.grid(column=2, row=4)

window.geometry("800x200")



def scheduled_attendance():
    while True:
        schedule.run_pending()
        time.sleep(1)  # Prevents CPU overload

        # Schedule face detection every hour
        schedule.every().hour.at(":00").do(detect_face)

        # Run the scheduler in a separate thread
        thread = threading.Thread(target=scheduled_attendance, daemon=True)
        thread.start()


window.mainloop()
