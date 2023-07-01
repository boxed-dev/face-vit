import cv2
import numpy as np
import face_recognition
import pickle
import os

path = 'Images1'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

encoding_file = "EncodeFile1.p"
if os.path.exists(encoding_file):
    with open(encoding_file, 'rb') as file:
        encodeListKnown = pickle.load(file)
else:
    encodeListKnown = []

cap = cv2.VideoCapture(0)


# Font configuration
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

bestMatchName = None
bestMatchProb = 0

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        faceProbs = face_recognition.face_distance(encodeListKnown, encodeFace)
        minProbIndex = np.argmin(faceProbs)
        minProb = faceProbs[minProbIndex]

        if bestMatchName is None or minProb < bestMatchProb:
            bestMatchProb = minProb
            bestMatchName = classNames[minProbIndex].upper()

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        (text_width, text_height), _ = cv2.getTextSize(bestMatchName, font, font_scale, font_thickness)
        text_x = x1 + (x2 - x1) // 2 - text_width // 2
        text_y = y2 - 6
        cv2.putText(img, bestMatchName, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Best match:", bestMatchName, "with probability:", 1 - bestMatchProb)
