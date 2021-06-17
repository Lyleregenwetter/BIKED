# -*- coding: utf-8 -*-
"""
Created on Thu May 27 00:01:48 2021

@author: Lyle
"""

import dataFrameTools
import convertParams
import pandas as pd
from pathlib import Path
import checkdf

def processGen(file="synthesized", genbcad=1, checkvalnum=0): 
    #Generate BikeCAD files from a datafram of synthesized parametric data
    #Function looks for a file in ../Data/file
    dataFrameTools.deNormalizeDF(file)
    if checkvalnum!=0:
        checkdf.checkdf(file,0,checkvalnum)
    
    deOHdf=dataFrameTools.deOH(file)
    reDF=dataFrameTools.convertOneHot(deOHdf, 0)
    reDF=dataFrameTools.standardizeReOH(reDF)
    reDF.to_csv(Path("../data/"+file+"_reDF.csv"))
    if genbcad==1:
        deOHdf=convertParams.deconvert(deOHdf)
        dataFrameTools.genBCAD(deOHdf)
        
if __name__ == '__main__':
    processGen()