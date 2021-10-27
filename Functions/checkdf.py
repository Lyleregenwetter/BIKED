# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 20:33:03 2021

@author: Lyle
"""

from pathlib import Path
import pandas as pd

def checkdf(file="vaegendf", printcodes=0, checkvalnum=1):
    df=pd.read_csv(Path("../data/"+file+"_Invsc.csv"), index_col=0)
    validmodels=[]
    for i in df.index:
        valid=1
        try:
            if df.at[i, "Saddle height"]<df.at[i, "Seat tube length"]+70:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " Saddle height too low")
        except:
            pass
        try:
            if df.at[i, "Saddle height"]>df.at[i, "Seat tube length"]+df.at[i, "Seatpost LENGTH"]:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " Seatpost too short")
        except:
            pass
        try:
            if df.at[i, "Wheel diameter front"]<df.at[i, "BSD front"]:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " Front Wheel OD smaller than rim OD")
        except:
            pass
        try:
            if df.at[i, "Wheel diameter rear"]<df.at[i, "BSD rear"]:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " Rear Wheel OD smaller than rim OD")
        except:
            pass
        try:
            if df.at[i, "BSD rear"]-df.at[i, "Rim depth rear"]*2>df.at[i, "ERD rear"]:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " Rear Spokes too short")
        except:
            pass
        try:
            if df.at[i, "BSD front"]-df.at[i, "Rim depth front"]*2>df.at[i, "ERD front"]:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " Front Spokes too short")
        except:
            pass
        try:
            if df.at[i, "Wheel diameter rear"]>df.at[i, "Wheel cut"]:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " Wheel cut too small")
        except:
            pass
        try:
            if df.at[i, "Wheel diameter rear"]<df.at[i, "ERD rear"]:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " Rear Spokes too long")
        except:
            pass
        try:
            if df.at[i, "Wheel diameter front"]<df.at[i, "ERD front"]:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " Front Spokes too long")
        except:
            pass
        try:
            if df.at[i, "FDERD"]<=0:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " FDERD<0")
        except:
            pass
        try:
            if df.at[i, "RDERD"]<=0:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " RDERD<0")
        except:
            pass
        try:
            if df.at[i, "FDBSD"]<=0:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " FDBSD<0")
        except:
            pass
        try:
            if df.at[i, "RDBSD"]<=0:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " RDBSD<0")
        except:
            pass
        if valid==1:
            validmodels.append(i)
    print(validmodels)
    print(str(len(validmodels))+" valid models")
    print(str(len(df.index))+" total models")
    print(str(float(len(validmodels))/float(len(df.index))) + " fraction valid")
    if len(validmodels[:])>=checkvalnum:
        validmodels=validmodels[:checkvalnum]
        sampled=df.loc[validmodels]
        sampled.index=[file+str(i) for i in range(checkvalnum)]
    else:
        print("WARNING: NOT ENOUGH VALID SAMPLES")
        sampled=df.loc[validmodels]
        sampled.index=[file+str(i) for i in range(len(validmodels[:]))]
    sampled.to_csv(Path("../data/"+file+"_Invsc.csv"))