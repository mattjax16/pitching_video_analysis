'''
pitch_pose_estimate.py

This is a script using opencv and googles mediapipe module

The goal of this script is to train a model to reconize the different poses of each pitch.

'''




import cv2
import mediapipe as mp
import time
from hand_detector import handDetector

#gopro imports
from goprocam import GoProCamera
from goprocam import constants

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()


cap = cv2.VideoCapture(2)
hand_detect = handDetector(detectionCon= 0.5, trackCon= 0.5)

prev_time = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)



    img = hand_detect.findHands(img)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


class PitchPoseDetector():
    def __init__(self):
        pass

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import requests
# import bs4
# import requests_html
# import json
# import cv2
# import csv
# import time
# import concurrent.futures
# import os
# import mediapipe as mp
#
#
# webcam1_cap = cv2.VideoCapture(2)
#
# while True:
#     ret, img = webcam1_cap.read()
#
#     cv2.imshow('webcam1',img)
#     cv2.waitKey(1)