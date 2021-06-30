'''
gopro_capture.py

This is a class for controlling the gopro

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
from goprocam import constants as go_pro_constants

class GoProController():

    def __init__(self):
        self.go_pro = GoProCamera.GoPro()
        self.media_list = self.go_pro.listMedia()




'''The main test function for the class'''
def main():
    goPro = GoProController()

    print(f'Done with gopro_capture')




if __name__ == "__main__":
    main()
