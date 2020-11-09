import cv2
import os

# wget https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml
def viola_jones_anime(img, cascade_file = "./models/lbpcascade_animeface.xml"):
    

    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    return faces
    
def viola_jones_combine(img, 
        class_face=cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml"), 
        class_eye=cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_eye.xml")):
    faces = class_face.detectMultiScale(img, minNeighbors=9, minSize=(100,100))
    eyes = class_eye.detectMultiScale(img, minNeighbors=9, minSize=(100,100))
    # faces = class_face.detectMultiScale(img, minNeighbors=5)
    # eyes = class_eye.detectMultiScale(img, minNeighbors=5)


    faces_keep = []

    if len(eyes) >= 2 and len(faces) >= 1:
        for i in range(0, len(faces)):
            eye_count = 0
            for j in range(0, len(eyes)):
                if eyes[j][0] > faces[i][0] and eyes[j][1] > faces[i][1]\
                        and eyes[j][2] < faces[i][2] and eyes[j][3] < faces[i][3]:
                    eye_count += 1

            if eye_count == 2:
                faces_keep.append(i)

    if len(faces_keep) is 0:
        return None
    return faces[faces_keep]



