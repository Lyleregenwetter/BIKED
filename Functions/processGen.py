# -*- coding: utf-8 -*-
"""
Created on Thu May 27 00:01:48 2021

@author: Lyle
"""

import dataFrameTools
import pandas as pd
from pathlib import Path

def processGen(file="synthesized", genbcad=1, checkvalnum=0): 
    #Generate BikeCAD files from a datafram of synthesized parametric data
    #Function looks for a file in ../Data/file
    dataFrameTools.deNormalizeDF(file)
    
    #Functionality coming later
    # if checkvalnum!=0:
    #     checkdf.checkdf(file,0,checkvalnum)
    
    deOHdf=dataFrameTools.deOH(file)
    if genbcad==1:
        dataFrameTools.genBCAD(file)
        
if __name__ == '__main__':
    processGen()