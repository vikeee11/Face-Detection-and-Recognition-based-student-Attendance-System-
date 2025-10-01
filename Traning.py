import os
import numpy as np
from PIL import Image
import cv2


def train_classifier(data_dir):
    # Get all file paths in the directory
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    faces = []
    ids = []

    for image in path:
        try:
            # Open and convert image to grayscale
            img = Image.open(image).convert('L')
            imageNp = np.array(img, 'uint8')

            # Extract ID from filename
            id = int(os.path.split(image)[1].split(".")[1])

            # Append face and ID to respective lists
            faces.append(imageNp)
            ids.append(id)
        except Exception as e:
            print(f"Error processing file {image}: {e}")

    ids = np.array(ids)

    # Train and save classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")


# Call the function
train_classifier("data")

