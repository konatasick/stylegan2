# from https://github.com/rolux

import os
import sys
import bz2
from tensorflow.keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
import re


LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path
def lndmrk_str2digital(pts_file):
  # pts_file = '/content/gdrive/MyDrive/face-of-art/examples/out_pred_landmarks/4k-wreckitralph2-animationscreencaps.com-1950_01.pts'
  with open(pts_file, 'r') as f:
      lndmk_str = f.read()
  f.close()
  pattern = re.compile('[0-9]+')
  lndmk_str_lst = lndmk_str[lndmk_str.index('{'):].split('\n')
  lndmk_lst = [eval(f"[{x.strip().replace(' ', ',')}]") for x in lndmk_str_lst if pattern.findall(x)!=[]]
  return lndmk_lst

if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """

    # landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                              #  LANDMARKS_MODEL_URL, cache_subdir='temp'))
    landmarks_model_path = './models/shape_predictor_68_face_landmarks.dat'
    RAW_IMAGES_DIR = sys.argv[1]
    ALIGNED_IMAGES_DIR = sys.argv[2]
    LANDMARKS_DIR = f'{RAW_IMAGES_DIR}-out_pred_landmarks'

    for img_name in [x for x in os.listdir(RAW_IMAGES_DIR) if x[0] not in '._']:
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        print(f'processing {img_name}')

        face_img_name = '%s.png' % (os.path.splitext(img_name)[0])
        aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
        os.makedirs(ALIGNED_IMAGES_DIR, exist_ok=True)
        face_landmarks_name = '%s.pts' % (os.path.splitext(img_name)[0])
        face_landmarks_path = os.path.join(LANDMARKS_DIR, face_landmarks_name)
        face_landmarks = lndmrk_str2digital(face_landmarks_path)
        if not os.path.exists(aligned_face_path):
          image_align(raw_img_path, aligned_face_path, face_landmarks)