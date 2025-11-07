# Download OpenCV DNN face detector model files if not present
import os
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'opencv_models')
os.makedirs(MODEL_DIR, exist_ok=True)

PROTO_PATH = os.path.join(MODEL_DIR, 'deploy.prototxt')
CAFFE_MODEL_PATH = os.path.join(MODEL_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')


# Updated URLs (as of 2025)
DEPLOY_URL = 'https://github.com/opencv/opencv/raw/4.x/samples/dnn/face_detector/deploy.prototxt'
CAFFE_MODEL_URL = 'https://github.com/opencv/opencv_3rdparty/raw/4.x/dnn_models/res10_300x300_ssd_iter_140000.caffemodel'

if not os.path.exists(PROTO_PATH):
    print('[SETUP] Downloading deploy.prototxt...')
    urllib.request.urlretrieve(DEPLOY_URL, PROTO_PATH)

if not os.path.exists(CAFFE_MODEL_PATH):
    print('[SETUP] Downloading res10_300x300_ssd_iter_140000.caffemodel...')
    urllib.request.urlretrieve(CAFFE_MODEL_URL, CAFFE_MODEL_PATH)

print('[SETUP] OpenCV DNN face detector model files are ready.')
