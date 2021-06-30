'''
gopro_capture.py

This is a script to control my go pro hero 9

'''

# initial importing of libraries needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import bs4
import requests_html
import json
import cv2
import csv
import time
import concurrent.futures
import os

#gopro imports
from goprocam import GoProCamera
from goprocam import constants

#motion detection lkibraries
from hand_detector import handDetector

from pose_detector import poseDetector





def record_clip(time_to_record = 13, save_path = 'data', clip_name = 'test_clip1.mp4'):
    '''
    This is a method to record a clip from the gopro and save it to the sd card on the gopro
    :return:
    '''

    gopro = GoProCamera.GoPro()

    #make gopro start beeping
    gopro.locate('1')

    #delete all video on sd card
    gopro.delete('all')

    # #capture video for certian number of time
    # gopro.shutter(constants.start)
    # print(f'started recording')
    # time.sleep(time_to_record)
    # gopro.shutter(constants.stop)
    # print(f'finished recording')

    # gopro_mode = gopro.getStatus(param='status',value='42')

    # change the resolution adn frame rate of the stream
    gopro.video_settings(res='1080p', fps='30')


    #have go pro record clip:
    # clip_url = gopro.shoot_video(time_to_record)

    #get the clip url
    clip_url = gopro.getMedia()

    #save the clip to the path provided
    gopro.downloadLastMedia(clip_url,'gopro_test.MP4')

    # make gopro stop beeping
    gopro.locate('0')


def live_capture():
    '''
    This function is used to get a live capture feed from the gopro
    :return:
    '''
    gopro = GoProCamera.GoPro()
    gopro.gpControlSet(constants.Stream.BIT_RATE, constants.Stream.BitRate.B2_4Mbps)

    # change the resolution adn frame rate of the stream
    gopro.video_settings(res='1080p', fps='120')

    # gopro.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.W480)
    # gopro.stream("udp://127.0.0.1:10000")

    # make video capture object from go pro source
    cap = cv2.VideoCapture("udp://127.0.0.1:10000")

    # make hand detection object
    hand_detect = handDetector(detectionCon=0.4, trackCon=0.5)

    #make pose dectector object
    pose_detect = poseDetector(detectionCon=0.5,trackCon=0.5)
    # set var for calculating FPS
    prev_time = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect the hands in the image
        # frame = hand_detect.findHands(frame)

        # detect the body in the image
        frame = pose_detect.findPose(frame)

        # showing the FPS (frames per second)
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(frame, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        # Display the resulting frame
        cv2.imshow("GoPro OpenCV", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()

    print(f'Done with live_capture()')
    return

'''The main test function for the class'''
def main():

    record_clip(time_to_record=5)

    print(f'Done with Main Function gopro_capture.py')
    # go_pro = GoProCamera.GoPro()
    #
    #
    # # test_list_media = json.load(go_pro.listMedia())
    #
    # # go_pro.video_setting(fps = '120')
    #
    # go_pro.video_settings(res = '24',fps='30')
    #
    # test_list_media = go_pro.listMedia()
    #
    # # go_pro.delete('all')
    #
    # # test_list_media = go_pro.listMedia()





if __name__ == "__main__":
    main()


