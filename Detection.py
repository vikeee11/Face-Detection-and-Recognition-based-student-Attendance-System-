import cv2

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        face = gray_img[y:y + h, x:x + w]
        if face.size > 0:
            id, pred = clf.predict(face)
            confidence = int(100 * (1 - pred / 300))

            if confidence > 70:
                name = "VIVEK" if id == 1 else "Manish" if id == 2 else "Unknown"
                cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return img

# Load Haar Cascade
faceCascade = cv2.CascadeClassifier('haarcascadefrontalfacedefault.xml')
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

# Open the default camera
video_capture = cv2.VideoCapture(0)
while True:
    ret, img = video_capture.read()
    if not ret:
        break
    img = draw_boundary(img, faceCascade, 1.3, 6, (255, 255, 255), "Face", clf)
    cv2.imshow("Face Detection", img)
    if cv2.waitKey(1) == 13:  # Press Enter to exit
        break

video_capture.release()
cv2.destroyAllWindows()
