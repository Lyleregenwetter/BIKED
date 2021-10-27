# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 21:51:20 2021

@author: Lyle
"""

import dataFrameTools
import numpy as np
import pandas as pd
import time
from pathlib import Path


def convert(df):
    df["RDERD"]=df["Wheel diameter rear"]-df["ERD rear"]
    df["FDERD"]=df["Wheel diameter front"]-df["ERD front"]
    df["RDBSD"]=df["Wheel diameter rear"]-df["BSD rear"]
    df["FDBSD"]=df["Wheel diameter front"]-df["BSD front"]
    df.drop(["ERD rear"], axis=1, inplace=True)
    df.drop(["ERD front"], axis=1, inplace=True)
    df.drop(["BSD rear"], axis=1, inplace=True)
    df.drop(["BSD front"], axis=1, inplace=True)
    return df


def deconvert(df):
    if "RDERD" in df.columns:
        df["ERD rear"]=df["Wheel diameter rear"]-df["RDERD"]
        df.drop(["RDERD"], axis=1, inplace=True)
    if "FDERD" in df.columns:
        df["ERD front"]=df["Wheel diameter front"]-df["FDERD"]
        df.drop(["FDERD"], axis=1, inplace=True)
    if "RDBSD" in df.columns:
        df["BSD rear"]=df["Wheel diameter rear"]-df["RDBSD"]
        df.drop(["RDBSD"], axis=1, inplace=True)
    if "FDBSD" in df.columns:
        df["BSD front"]=df["Wheel diameter front"]-df["FDBSD"]
        df.drop(["FDBSD"], axis=1, inplace=True)
    df["nCHAINSTAYOFFSET"]=df["CHAINSTAYOFFSET"]
    df["nCHAINSTAYAUXrearDIAMETER"]=df["CHAINSTAYAUXrearDIAMETER"]
    df["nChain stay horizontal diameter"]=df["Chain stay horizontal diameter"]
    df["nChain stay position on BB"]=df["Chain stay position on BB"]
    df["nChain stay taper"]=df["Chain stay taper"]
    df["nChain stay back diameter"]=df["Chain stay back diameter"]
    df["nChain stay vertical diameter"]=df["Chain stay vertical diameter"]
    df["nSeat stay junction0"]=df["Seat stay junction0"]
    df["nSeat stay bottom diameter"]=df["Seat stay bottom diameter"]
    df["nSEATSTAY_HF"]=df["SEATSTAY_HF"]
    df["nSSTopZOFFSET"]=df["SSTopZOFFSET"]
    df["nSEATSTAY_HR"]=df["SEATSTAY_HR"]
    df["nSEATSTAYTAPERLENGTH"]=df["SEATSTAYTAPERLENGTH"]
    return df