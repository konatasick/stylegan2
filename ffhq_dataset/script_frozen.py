import cv2
import os
    
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




if __name__ == "__main__":

    # classifiers
    face_cascade = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_eye.xml")
    face_dir = '/data2/kkwu/cartoon_face/frozen2013raw/'
    output_dir = '../output/frozen2013raw/'
    try:
        os.system(f'rm ../output/frozen2013raw -rf')
    except:
        pass
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(face_dir):
        img = cv2.imread(f"{face_dir}{filename}")
        if img is not None:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = resize_img(img, 700)
            faces = viola_jones_combine(img, face_cascade, eye_cascade)
            if faces is not None:
                for i, (x,y,w,h) in enumerate(faces):
                    # img = cv2.rectangle(img, (x,y),(x+w, y+h), (4,8,170), 2)
                    if h<100 or w<100:
                        continue
                    im_w, im_h = img.shape[:2]
                    try:
                        y_ = y - int(im_w/100*10)
                        h_ = h + int(im_w/100*10)*2
                        x_ = x - int(im_w/100*10)
                        w_ = w + int(im_w/100*10)*2
                        tmp_img = img[y_:y_+h_, x_:x_+w_, :]
                        cv2.imwrite(f"{output_dir}{i}_{i+1}_{filename}", tmp_img)
                        y_ = y - int(im_w/100*15)
                        h_ = h + int(im_w/100*15)*2
                        x_ = x - int(im_w/100*15)
                        w_ = w + int(im_w/100*15)*2
                        tmp_img = img[y_:y_+h_, x_:x_+w_, :]
                        cv2.imwrite(f"{output_dir}{i}_{i+2}_{filename}", tmp_img)
                    except:
                        tmp_img = img[y:y+h, x:x+w, :]
                        cv2.imwrite(f"{output_dir}{i}_{filename}", tmp_img)
                    print(f'Saved {output_dir}{i}_{filename}.')