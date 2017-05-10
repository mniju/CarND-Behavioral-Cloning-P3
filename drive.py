import argparse
import base64 #For encoding/decoding data
from datetime import datetime
import os
import shutil # High level File Operations

import numpy as np
import socketio # Lowlevel networking interface
import eventlet #concurrent networking library similiar to threading
import eventlet.wsgi
from PIL import Image # Python Umaging Library
from flask import Flask # Miniwebframe work for python
from io import BytesIO
import cv2 # Open CV

from keras.models import load_model

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def img_resize(image,new_size):
    return cv2.resize(image,(new_size[0],new_size[1]), interpolation=cv2.INTER_AREA)

def img_crop(image,y_crop):
    # img[y:y+h, x:x+w] -> 35 from top until 130 in bottom
    image = image[y_crop[0]:y_crop[1], 0:image.shape[1]]
    return image

@sio.on('telemetry')
def telemetry(sid, data):
    y_crop = (50,140)  # Cropping Size
    new_size = (200,66) # New size after Resizing
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_raw = np.asarray(image)
        image_crop = img_crop(image_raw,y_crop)
        image_rz = img_resize(image_crop,new_size)
        steering_angle = float(model.predict(image_rz[None, :, :, :], batch_size=1))
        throttle = 0.15
        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
