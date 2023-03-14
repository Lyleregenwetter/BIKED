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

def processGen(file="synthesized", gen_name="", dataset="", genbcad=1, denorm = False, check=True, from_OH=True, intermediates=0, sourcepath = "PlainRoadbikestandardized.txt", targetpath = "../Generated BCAD Files/Files/"): 
    #Generate BikeCAD files from a datafram of synthesized parametric data
    #Function looks for a file in ../Data/file
    if isinstance(file, str):
        df=pd.read_csv(Path("../data/"+file+".csv"), index_col=0)
    else:
        df=file
    if denorm:
        df=dataFrameTools.deNormalizeDF(df, dataset, 1, intermediates)
    if check: #-1 for use all
        df = checkdf.checkdf(df, gen_name, 0, intermediates)
    if from_OH:
        
        df=dataFrameTools.deOH(df, dataset, intermediates)
        
    reDF=dataFrameTools.convertOneHot(df, dataset, 0)
    reDF=dataFrameTools.standardizeReOH(reDF, dataset, intermediates)    
    if genbcad==1:
        deOHdf=convertParams.deconvert(df, dataset)
        dataFrameTools.genBCAD(deOHdf, sourcepath, targetpath)
    return reDF

    