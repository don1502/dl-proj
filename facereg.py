#Importing the required libraries
import face_recognition
import cv2
import os
from google.colab.patches import cv2_imshow

#Image reading
img = cv2.imread("img.jpg")

def read_images(path):
    img = cv2.imread(path)
    (h, w) = img.shape[:2]
    width = 500
    ratio = width / float(w)
    height = int(h * ratio)
    return cv2.resize(img, (width, height))

#for known dataset
knownencodings = []
knownnames = []
known_dir = "known"#Add the path to your known images
for name in os.listdir(known_dir):
    img = read_images(f"{known_dir}/{name}")
    img_enc = face_recognition.face_encodings(img)[0]
    knownencodings.append(img_enc)
    knownnames.append(os.path.split('.')[0])
print(knownnames)

#for unkown dataset & face recognition
unknown_dir = "unknown"#Add the path to your unknown images 
for name in os.listdir(unknown_dir):
    img = read_images(f"{unknown_dir}/{name}")
    img_enc = face_recognition.face_encodings(img)[0]
    results= face_recognition.compare_faces(knownencodings, img_enc)
    print(results)
    print(face_recognition.face_distance(knownencodings, img_enc))
    res = [i for i, val in enumerate(results) if val]
    name = knownnames[res[0]]
    print(name)

#for drawing rectangle around the face & putting name on it
(top,right,bottom,left) = face_recognition.face_locations(img)[0]
cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 100), 2)
cv2.putText(img, name, (left+2, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)
cv2_imshow(img)