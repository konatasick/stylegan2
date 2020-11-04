import dlib
from ffhq_dataset.script_frozen import viola_jones_combine


class LandmarksDetector:
    def __init__(self, predictor_model_path, mode='real'):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        mode:
        'real'--real people, use dlib detector
        'cartoon'-- cartoon face
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)
        self.mode = mode


    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)

        if self.mode == 'real':
            dets = self.detector(img, 1)
        elif self.mode == 'cartoon':
            dets = dlib.rectangles()
            faces = viola_jones_combine(img[:,:,::-1])
            if faces is not None:
                for i, (x,y,w,h) in enumerate(faces):
                    rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                    dets.append(rect)


        for detection in dets:
            if detection.height() > 100 and detection.width() > 100:
                try:
                    face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
                    yield face_landmarks
                except:
                    print("Exception in get_landmarks()!")
