# -*- coding: utf-8 -*-
"""
Created on Thu May 27 21:43:04 2021

@author: Lyle
"""


import cv2
import dataFrameTools
from os import path
import sys


def preprocessImages(color=0, enableProgressBar=1):
    df=dataFrameTools.loadScaledDF()
    dfmodels = df.index.values
    numModels=4800
    imPath='..\\Images\\'
    if color==0:
        newimPath='..\\Processed Images\\Grayscale\\'
    else:
        newimPath='..\\Processed Images\\Colored\\'
    if enableProgressBar==1:
            toolbar_width = 40
            sys.stdout.write("[%s]" % (" " * toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    count=0
    for modelNum in range(1,numModels+1):
        imName=imPath+"(" + str(modelNum) + ").png"
        # imName=imPath+ str(modelNum) + ".png"
        rowCount=0
        if modelNum in dfmodels:
            if path.exists(imName): #If file exists
                count=count+1
                image=cv2.imread(str(imName),color)
                image=cv2.resize(image, (260,120))
                cv2.imwrite(newimPath+"(" + str(modelNum) + ").png", image)
        #     else:
        #         print(str(modelNum)+" has df value but not image")
        # if path.exists(imName):
        #     if modelNum not in dfmodels:
        #         print(str(modelNum)+" has image but not df value")
        if enableProgressBar==1 and modelNum%10==0:
            sys.stdout.write("-")
            sys.stdout.flush()
    if enableProgressBar==1:
        sys.stdout.write("]\n")

if __name__ == '__main__':
    preprocessImages()