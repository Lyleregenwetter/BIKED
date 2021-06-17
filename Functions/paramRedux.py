# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:00:57 2020

@author: Lyle
"""
import dataFrameTools
import convertParams
import numpy as np
import pandas as pd
import time
from pathlib import Path

def paramRedux(mini=0, convert=1):
    start_time=time.time()
    df=pd.read_csv("../Data/BIKED_raw.csv", index_col=0, low_memory=False) 
    df=dropmodels(df) #Removes tandem and extra features and drops any models with these features

    if mini==0:
        df=dropcolumns(df) #Remove columns deemed to be irrelevant
    else:
        df=dropcolumnsmini(df)
    df=dataFrameTools.dropData(df)  #Remove columns and rows that only contain N/A's
    if convert==1:
        df=convertParams.convert(df)
    
    catlist=["Dimension units","ForkLengthMeasurement1","ForkLengthMeasurement0","Gearing analysis","HEADSETprofile","REARbrake kind","Rollout units","ShoeTipOrCleatX","nSeat stay mount location","spc type","HORIZONTAL_TOP","TRACK_ERGO","Seat stay mount location","PumpTube","PumpLocate","PhBlock styleTOPTUBE","PhBlock styleDOWNTUBE","DropoutParamOrStatic","SHIFTERtype","SEATTUBEBENDS","SEATSTAYSYMMETRY","Pumporient","Head tube type","Hand position","Fork type","FRONTbrake kind","EYELETS","Down tube type","Chain stay trad wish yoke","BarEndShiftType","CHAIN_GUARDSTYLE","CLAMPSTYLE","CRANK_POS","BRAKEMOUNT_TO", "Top tube type", "Stem kind", "Seat tube type", "STEM3_MM_RATIO","STEM1_MM_RATIO","SSAUX2_MM_RATIO","SSAUX1_MM_RATIO","CSAUX3_MM_RATIO","CSAUX2_MM_RATIO","CSAUX1_MM_RATIO","Seat stay Curv or Rake","TOPTUBEBENDS","nSeat stay Curv or Rake","nSSAUX2_MM_RATIO","nSSAUX1_MM_RATIO",'BELTorCHAIN',"nCSAUX1_MM_RATIO","nCSAUX2_MM_RATIO","nCSAUX3_MM_RATIO"]
    intlist=["FRONTROTORBOLTS","REARROTORBOLTS","Toe overlap bar angle","Shoe down angle","SPIDER_ARM_PHASE_SHIFT", "DOWNTUBEBENDS","DERAILLEURX","DERAILLEURL","Crouch","Cleat X","Cleat Y","Brake lever position","CRANK_SPIDER","STEMBENDS","Crank up angle","Crank down angle","Cadence","CSSIDEBENDS","CSAUXSLIDER","CSAUXBENDS","BEND_POSITION","DERAILLEUR_PULLEY_TEETH","DERAILLEUR_PULLEY2_TEETH","SSAUXBENDS","SSSIDEBENDS","SSAUXSLIDER","Shoe up angle","Shoe size","nCSAUXBENDS","nCSSIDEBENDS","nSSAUXBENDS","nSSSIDEBENDS", "nSSAUXSLIDER","nCSAUXSLIDER","SPOKE_PHASE_SHIFT rear", "SPOKE_PHASE_SHIFT front", "SPOKES rear","SPOKES front","SPOKES composite rear","SPOKES composite front","Number of cogs", "Number of chainrings","Teeth on cog 0","Dim A mmInch","Dim B mmInch","Dim C mmInch","Dim D mmInch","Dim E mmInch","Dim F mmInch","SELECTEDCOG","SELECTEDRING"]
    for col in df.columns:
        if df.dtypes[col]==np.int64:
            if col.endswith("style"):
                catlist.append(col)
            if col.startswith("CHEVRON"):
                intlist.append(col)
            if col.startswith("CORNER"):
                intlist.append(col)
    for i in range(12):
        intlist.append("Teeth on cog "+str(i))
        intlist.append("Teeth on chainring "+str(i))


    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1, inplace=True)

    for idx in df.index:
        # if df.at(idx, "Display WATERBOTTLES")==True:
        if df.at[idx, "Display WATERBOTTLES"]==False:
            df.at[idx, "bottle DOWNTUBE0 show"]=False
            df.at[idx, "bottle SEATTUBE0 show"]=False
        else:
            if df.at[idx, "bottle DOWNTUBE0 show"]!=True:
                df.at[idx, "bottle DOWNTUBE0 show"]=False
            if df.at[idx, "bottle SEATTUBE0 show"]!=True:
                df.at[idx, "bottle SEATTUBE0 show"]=False
    df.drop(columns="Display WATERBOTTLES",inplace=True)
    
    #models manually deemed not to be bikes
    modeldroplist=[38,751,1209,1239,1321,1457,1546,2019,2163,2880,2884,3202,3203,3207,3209,3504,4139]
    df.drop(modeldroplist, axis=0, inplace=True)
    
    #models that appear to be broken in some way, can be optionally dropped
    borderline=[199,240,327,566,630,754,985,1062,1065,1104,1125,1151,1154,1232,1282,1287,1344,1346,1355,1356,1368,1382,1387,1412,1416,1453,1464,1657,1669,1787,1863,1957,2047,2048,2065,2108,2292,2352,2354,2557,2560,2563,2564,2565,2566,2567,2568,2569,2570,2571,2572,2574,2575,2577,2578,2579,2580,2584,2588,2590,2638,2640,2641,2647,2703,2713,2770,2772,2843,2930,3032,3102,3125,3126,3127,3142,3155,3161,3211,3214,3289,3303,3499,3751,3779,3785,3925,4093,4265,4373,4383,4654,4670]
    # df.drop(borderline, axis=0, inplace=True)
    if mini==0:
        df=fixunits(df)
    
    dtypedf=df.dtypes
    for col in df.columns:
        if col in catlist:
            df[col] = df[col].astype(str)
            dtypedf[col]="str"
    for col in df.columns:
        if df.dtypes[col]==np.float64:
            if allint(df,col):
                # print(col)
                dtypedf[col]="int64"
    
    dtypedf.to_csv("../Data/BIKED_datatypes.csv",header=["type"])
    df.to_csv(Path("../Data/BIKED_reduced.csv"))
    print("Full Parameter Space Reduction completed in %s seconds" % (time.time() - start_time))    

def dropcolumns(df):
    start_time=time.time()
    endrules=(" offset", " offsetX", "show", "GREEN", "RED", "BLUE", "IMAGENAME", "IMAGEFitHeight","IMAGEFitWidth", "IMAGEaspectR", "TILED", "IMAGEYES", "TNDM0", "TNDM1", "TNDM2", "TNDM3", "TNDM4", "TNDM5", "carve", "RGB", "EXTR0", "EXTR1", "EXTR2", "EXTR3", "EXTR4", "EXTR5", "EXTR6", "EXTR7", "EXTR8", "EXTR9", "EXTR10")
    startrules=("Display", "DECAL", "OUTLINE","TNDM","CblGuide","FH","show","Show","GRADIENT","OUTGRADIENT", "Drawing", "Box","ET is displayed","User dim ","BOOM","carve", "bottle DOWNTUBE1","bottle SEATTUBE1","bottle TOPTUBE1","bottle SSSIDE1","bottle nSSSIDE1","bottle DOWNTUBE2", "bottle SEATTUBE2","bottle TOPTUBE2","bottle SSSIDE2","bottle nSSSIDE2","bottle DOWNTUBE3","bottle SEATTUBE3","bottle TOPTUBE3", "bottle SSSIDE3","bottle nSSSIDE3","bottle nSSSIDE0","bottle SSSIDE0","bottle FORK1","bottle FORK0","bottle TOPTUBE0","CROSS_PATTERN","Photo")
    substring_list=["RGB","PAINT"]
    overrideinclude=("Display FENDERS","Display FENDERS","Display SEATPOST_CLAMP","Display AEROBARS","Display CHAINGUARD","Display RACK","Display WATERBOTTLES","bottle SEATTUBE0 show","bottle DOWNTUBE0 show")
    droplist=["LOCKFRAME","KPH MPH","FRAMES_PER_ROTATION", "DIM_WHEEL","DIM_DOT_DIA","DIM_ARROW_WID","DIM_ARROW_LEN","EXTRA_LINKS","Dimension text size","Paper aspect ratio",'Title block position', 'Title block text size',"Note width","Logo width","Decimal places","Angular decimal places","WHEEL_DISP_STATE","MODEL_NAME","MORE_INFO", "Photo file name","Name","BRAND","Email","Notes","FRAME_NUMBER","MODEL_YEAR","Paint scheme","Notes drawing","Phone","Address line 0","Address line 1","Address line 2","Address line 3"]

    collist=[]
    for col in df.columns: #Main Parameter Space Reduction
        if col.startswith(startrules) or col.endswith(endrules) or any(substring in col for substring in substring_list) or col in droplist:
            if col not in overrideinclude:
                collist.append(col)
    df.drop(collist, axis=1, inplace=True)  
    print("Irrelevant features dropped in in %s seconds" % (time.time() - start_time))    
    # df.to_csv("../data/dropdf.csv")
    return df
    
def fixunits(df): #Convert units in inches to mm and remove the unit bool column
    for letter in ["A","B","C","D","E","F"]:
        for idx in df.index:
            col="Dim " + letter + " mmInch"
            if df.at[idx,col]==0:
                df.at[idx,"Dim " + letter +" TextField"]*=25.4
        df.drop(col, axis=1, inplace=True)   
    return df 

def allint(df,col):
    coldf=df[col].dropna()
    for idx in coldf.index:
        if int(df.at[idx,col])==df.at[idx,col]:
            pass
        else:
            return False
    return True

def dropmodels(indf):
    start_time = time.time()
    #Load the full xml dataframe 
    # df=pd.read_csv("../Data/fullxmldf.csv",index_col=0)
    substring_list=["TNDM","EXTRATUBE"]
    collist=[]
    for column in indf.columns:
        if any(substring in column for substring in substring_list):
            collist.append(column)
    booldf=indf[collist].notnull()
    newdf=indf[(booldf.T != False).any()]
    dropDF=indf.drop(newdf.index)
    dropDF.drop(collist, axis=1, inplace=True)
    dropDF.to_csv("../Data/BIKED_reduced.csv") #Dropdf 
    print("Bike Models with Tandem and Extra Members Dropped in %s seconds" % (time.time() - start_time))
    return dropDF

def dropcolumnsmini(df):
    miniparams=pd.read_csv("../data/minibikedParams.csv", index_col=0).index
    reddf=df[miniparams]
    return reddf
    
    
    
    
    
    
    
    
    
    