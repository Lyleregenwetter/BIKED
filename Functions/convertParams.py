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


def convert(df, dataset=""):
    for idx in df.index:
        if df.at[idx, "Display WATERBOTTLES"]==False:
            df.at[idx, "bottle DOWNTUBE0 show"]=False
            df.at[idx, "bottle SEATTUBE0 show"]=False
        else:
            if df.at[idx, "bottle DOWNTUBE0 show"]!=True:
                df.at[idx, "bottle DOWNTUBE0 show"]=False
            if df.at[idx, "bottle SEATTUBE0 show"]!=True:
                df.at[idx, "bottle SEATTUBE0 show"]=False
    df.drop(columns="Display WATERBOTTLES",inplace=True)
    
    if dataset=="":
        df["RDERD"]=df["Wheel diameter rear"]-df["ERD rear"]
        df["FDERD"]=df["Wheel diameter front"]-df["ERD front"]
        df["RDBSD"]=df["Wheel diameter rear"]-df["BSD rear"]
        df["FDBSD"]=df["Wheel diameter front"]-df["BSD front"]
        df.drop(["ERD rear"], axis=1, inplace=True)
        df.drop(["ERD front"], axis=1, inplace=True)
        df.drop(["BSD rear"], axis=1, inplace=True)
        df.drop(["BSD front"], axis=1, inplace=True)
    
    if dataset in ["micro", "clip_s"]:
        for idx in df.index:
            BBD=df.at[idx, "BB textfield"]
            FCD=df.at[idx, "FCD textfield"]
            FTY=BBD
            FTX=np.sqrt(FTY**2+FCD**2)
            x=df.at[idx, "FORK0R"]
            fkl=df.at[idx, "FORK0L"]
            htlx=df.at[idx, "Head tube lower extension2"]
            lsth=df.at[idx, "Lower stack height"]
            y=fkl+htlx+lsth
            ha=df.at[idx, "Head angle"]*np.pi/180
            dtx=FTX-y*np.cos(ha)-x*np.sin(ha)
            dty=FTY+y*np.sin(ha)+x*np.cos(ha)
            df.at[idx, "DT Length"]=np.sqrt(dtx**2+dty**2)
            
            csbd=df.at[idx, "Chain stay back diameter"]
            csvd=df.at[idx, "Chain stay vertical diameter"]
            csd=(csbd+csvd)/2
            df.at[idx, "csd"]=csd
            
            ssbd=df.at[idx, "Seat stay bottom diameter"]
            sshr=df.at[idx, "SEATSTAY_HR"]
            ssd=(ssbd+sshr)/2
            df.at[idx, "ssd"]=ssd
            
            ttrd=df.at[idx, "Top tube rear diameter"]
            ttrd2=df.at[idx, "Top tube rear dia2"]
            ttfd=df.at[idx, "Top tube front diameter"]
            ttfd2=df.at[idx, "Top tube front dia2"]
            ttd=(ttrd+ttrd2+ttfd+ttfd2)/4
            df.at[idx, "ttd"]=ttd
            
            dtrd=df.at[idx, "Down tube rear diameter"]
            dtrd2=df.at[idx, "Down tube rear dia2"]
            dtfd=df.at[idx, "Down tube front diameter"]
            dtfd2=df.at[idx, "Down tube front dia2"]
            dtd=(dtrd+dtrd2+dtfd+dtfd2)/4
            df.at[idx, "dtd"]=dtd
            
            
            df.at[idx, "Wall thickness Bottom Bracket"]=2.0
            df.at[idx, "Wall thickness Head tube"]=1.1
        
    if dataset in ["mini"]:
        pass
    if dataset in ["clip", "clip_s"]:
        for column in list(df.columns):
            if column.endswith("sRGB"):
                vals = df[column].values
                df.drop(column, axis=1, inplace=True)
                vals=vals+2**24
                r = np.floor_divide(vals , 2**16)
                g = np.floor_divide(np.mod(vals, 2**16) , 2**8)
                b = np.mod(vals, 2**8)
                df[column.replace("sRGB", "R_RGB")]=r
                df[column.replace("sRGB", "G_RGB")]=g
                df[column.replace("sRGB", "B_RGB")]=b
        pass
    return df.copy()

    # if dataset in ["clip"]:
    #     rgbs=[]
    #     all_ser=[]
    #     for column in list(df.columns):
    #         if column.endswith("sRGB"):
    #             vals = df[column].values
    #             df.drop(column, axis=1, inplace=True)
    #             vals=vals+2**24
    #             r = np.floor_divide(vals , 2**16)
    #             g = np.floor_divide(np.mod(vals, 2**16) , 2**8)
    #             b = np.mod(vals, 2**8)
    #             all_ser.append(pd.Series(r, name=column.replace("sRGB", "R_RGB")))
    #             all_ser.append(pd.Series(g, name=column.replace("sRGB", "G_RGB")))
    #             all_ser.append(pd.Series(b, name=column.replace("sRGB", "B_RGB")))
    #             rgbs.append(column.replace("sRGB", "R_RGB"))
    #             rgbs.append(column.replace("sRGB", "G_RGB"))
    #             rgbs.append(column.replace("sRGB", "B_RGB"))
    #     df = pd.concat([df] + all_ser, axis=1)
    #     rgbs=pd.DataFrame(rgbs)
    #     rgbs.to_csv("rgbs.csv")
    #     return df
def deconvert(df, dataset=""):
    if dataset=="":
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

    
    
    if dataset in ["micro", "clip_s"]:
        if "csd" in df.columns:
            df["Chain stay back diameter"]=df["csd"]
            df["Chain stay vertical diameter"]=df["csd"]
        if "ssd" in df.columns:
            df["SEATSTAY_HR"]=df["ssd"]
            df["Seat stay bottom diameter"]=df["ssd"]
        if "ttd" in df.columns:
            df["Top tube rear diameter"]=df["ttd"]
            df["Top tube rear dia2"]=df["ttd"]
            df["Top tube front diameter"]=df["ttd"]
            df["Top tube front dia2"]=df["ttd"]
        if "dtd" in df.columns:
            df["Down tube rear diameter"]=df["dtd"]
            df["Down tube rear dia2"]=df["dtd"]
            df["Down tube front diameter"]=df["dtd"]
            df["Down tube front dia2"]=df["dtd"]
        for idx in df.index:    
            Stack=df.at[idx, "Stack"]
            HTL=df.at[idx, "Head tube length textfield"]
            HTLX=df.at[idx, "Head tube lower extension2"]
            HTA=df.at[idx, "Head angle"]*np.pi/180
            BBD=df.at[idx, "BB textfield"]
            DTL=df.at[idx, "DT Length"]
            DTJY=Stack-(HTL-HTLX)*np.sin(HTA)
            DTJX=np.sqrt(DTL**2-DTJY**2)
            FWX=DTJX+(DTJY-BBD)/np.tan(HTA)
            FCD=np.sqrt(FWX**2+BBD**2)
            df.at[idx, "FCD textfield"]=FCD
        df.drop(["DT Length"], axis=1, inplace=True)
        
    if dataset in ["mini"]:
        pass
    if dataset in ["clip", "clip_s"]:
        for column in list(df.columns):
            if column.endswith("R_RGB"):
                r = df[column].values
                g = df[column.replace("R_RGB", "G_RGB")].values
                b = df[column.replace("R_RGB", "B_RGB")].values
                df.drop(column, axis=1, inplace=True)
                df.drop(column.replace("R_RGB", "G_RGB"), axis=1, inplace=True)
                df.drop(column.replace("R_RGB", "B_RGB"), axis=1, inplace=True)
                val=r*(2**16)+g*(2**8)+b-(2**24)
                df[column.replace("R_RGB", "sRGB")]=val
    return df.copy()
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    