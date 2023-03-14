# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 20:33:03 2021

@author: Lyle
"""

from pathlib import Path
import pandas as pd
import numpy as np

def checkdf(df, genname, printcodes=0, intermediates=0):
    validmodels=[]
    for i in df.index:
        valid=1
        #List of params which should be positive
        collist = ['CS textfield', 'Stack', 'Head angle',
       'Head tube length textfield', 'Seat tube length',
       'Seat angle', 'DT Length', 'BB diameter', 'ttd', 'dtd', 'csd', 'ssd',
       'Chain stay position on BB', 'MATERIAL',
       'Head tube upper extension2', 'Seat tube extension2',
       'Head tube lower extension2', 'SEATSTAYbrdgshift', 'CHAINSTAYbrdgshift',
       'SEATSTAYbrdgdia1', 'CHAINSTAYbrdgdia1', 'SEATSTAYbrdgCheck',
       'CHAINSTAYbrdgCheck', 'Dropout spacing',
       'Wall thickness Bottom Bracket', 'Wall thickness Top tube',
       'Wall thickness Head tube', 'Wall thickness Down tube',
       'Wall thickness Chain stay', 'Wall thickness Seat stay',
       'Wall thickness Seat tube', 'ERD rear', 'Wheel width rear',
       'Dropout spacing style', 'BSD front', 'Wheel width front', 'ERD front',
       'BSD rear', 'Fork type', 'Stem kind', 'Display AEROBARS',
       'Handlebar style', 'Head tube type', 'BB length', 'Head tube diameter',
       'Wheel cut', 'Seat tube diameter', 'Top tube type',
       'bottle SEATTUBE0 show', 'bottle DOWNTUBE0 show',
       'Front Fender include', 'Rear Fender include', 'BELTorCHAIN',
       'Number of cogs', 'Number of chainrings', 'Display RACK',
       'FIRST color R_RGB', 'FIRST color G_RGB', 'FIRST color B_RGB',
       'RIM_STYLE front', 'RIM_STYLE rear', 'SPOKES composite front',
       'SBLADEW front', 'SBLADEW rear', 'Saddle length', 'Saddle height',
       'Down tube diameter', 'Seatpost LENGTH']
        intersection = list(set(collist).intersection(set(list(df.columns))))
        subset = df[intersection]
        if (subset < 0).any().any():
            valid=0
            if printcodes==1:
                print("Model " + str(i) + " has a negative value where it shouldnt")

        
        try:
            if df.at[i, "Saddle height"]<df.at[i, "Seat tube length"]+40:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " Saddle height too low")
        except:
            pass
        try:
            if df.at[i, "Saddle height"]>df.at[i, "Seat tube length"]+df.at[i, "Seatpost LENGTH"]+30:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " Seatpost too short")
        except:
            print("Couldn't check seatpost length too short")
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
            if df.at[i, "BSD rear"]<df.at[i, "ERD rear"]:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " BSD<ERD rear")
        except:
            pass
        try:
            if df.at[i, "BSD front"]<df.at[i, "ERD front"]:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " BSD<ERD front")
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
            if df.at[i, "Head tube lower extension2"]>=df.at[i, "Head tube length textfield"]:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " HTLX>HTL")
        except:
            pass
        
        try:
            if df.at[i, "Head tube upper extension2"] + df.at[i, "Head tube lower extension2"]>=df.at[i, "Head tube length textfield"]:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " HTLX + HTUX>HTL")
        except:
            pass
        
        try:
            if df.at[i, "RDBSD"]<=0:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " RDBSD<0")
        except:
            pass
        try:
            if df.at[i, "CS textfield"]<=0:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " CSL is <=0")
        except:
            pass
        try:
            assert valid==1
            Stack=df.at[i, "Stack"]
            HTL=df.at[i, "Head tube length textfield"]
            HTLX=df.at[i, "Head tube lower extension2"]
            HTA=df.at[i, "Head angle"]*np.pi/180
            DTL=df.at[i, "DT Length"]
            if HTA<np.pi/2:
                DTJY=Stack-(HTL-HTLX)*np.sin(HTA)
                if DTJY**2>=DTL**2:
                    valid=0
                    if printcodes==1:
                        print("Model " + str(i) + " Down Tube too short to reach head tube junction") 
        except:
            pass
        
        try:
            assert valid==1
            Stack=df.at[i, "Stack"]
            HTL=df.at[i, "Head tube length textfield"]
            HTLX=df.at[i, "Head tube lower extension2"]
            HTA=df.at[i, "Head angle"]*np.pi/180
            DTL=df.at[i, "DT Length"]
            BBD=df.at[i, "BB textfield"]
            DTJY=Stack-(HTL-HTLX)*np.sin(HTA)
            DTJX=np.sqrt(DTL**2-DTJY**2)
            FWX=DTJX+(DTJY-BBD)/np.tan(HTA)
            FCD=np.sqrt(FWX**2+BBD**2)
            FBSD=df.at[i, "BSD front"]
            DTOD = STOD=df.at[i, "Down tube diameter"]
            
            ang = np.arctan2(DTJY, DTJX)-np.arctan2(BBD, FWX)
            if ang<np.pi/2:
                if np.sin(ang)*FCD < FBSD/2-DTOD:
                    valid = 0
                    if printcodes==1:
                        print("Model " + str(i) + " Down Tube intersecting Front Wheel") 
        except:
            pass
        
        try:
            assert valid==1
            Stack=df.at[i, "Stack"]
            HTL=df.at[i, "Head tube length textfield"]
            HTLX=df.at[i, "Head tube lower extension2"]
            HTA=df.at[i, "Head angle"]*np.pi/180
            DTL=df.at[i, "DT Length"]
            BBD=df.at[i, "BB textfield"]
            DTJY=Stack-(HTL-HTLX)*np.sin(HTA)
            DTJX=np.sqrt(DTL**2-DTJY**2)
            FWX=DTJX+(DTJY-BBD)/np.tan(HTA)
            FBSD=df.at[i, "BSD front"]
            FCD=np.sqrt(FWX**2+BBD**2)
            if FCD<FBSD/2+172.5:
                valid = 0
                if printcodes==1:
                    print("Model " + str(i) + " toe overlap") 
        except:
            pass
        
        try:
            assert valid==1
            CSL=df.at[i, "CS textfield"]
            BBOD=df.at[i, "BB diameter"]
            FBSD=df.at[i, "BSD rear"]
            if CSL<FBSD/2+BBOD/2:
                valid = 0
                if printcodes==1:
                    print("Model " + str(i) + " Rear Wheel intersecting Bottom Bracket") 
        except:
            pass
        try:
            assert valid == 1
            CSL=df.at[i, "CS textfield"]
            BBD=df.at[i, "BB textfield"]
            STA=df.at[i, "Seat angle"]*np.pi/180
            RBSD=df.at[i, "BSD rear"]
            STOD=df.at[i, "Seat tube diameter"]
            ang = STA-np.arcsin(BBD/CSL)
            if ang<np.pi/2:
                if ang<=0 or np.sin(ang)*CSL < RBSD/2-STOD/2:
                    valid = 0
                    if printcodes==1:
                        print("Model " + str(i) + " Seat Tube intersecting Rear Wheel") 
        except:
            pass
        
        if valid==1:
            validmodels.append(i)
            
    print(str(float(len(validmodels))/float(len(df.index))) + " fraction valid")

    sampled=df.loc[validmodels]
    if intermediates!=0:
        sampled.index=[genname+str(i) for i in range(len(validmodels[:]))]
        sampled.to_csv(Path("../data/"+intermediates+"_sampled.csv"))
    return sampled
    
def checkstructureal(file="vaegendf", printcodes=0, checkvalnum=1):
    df=pd.read_csv(Path("../data/"+file+"_Invsc.csv"), index_col=0)
    validmodels=[]
    for i in df.index:
        valid=1
        try:
            if df.at[i, "Wall thickness Seat tube"]!=0.90:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " Seat tube thickness is not 0.9")
        except:
            pass
        try:
            if df.at[i, "Wall thickness Seat stay"]!=1.0:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " Seat stay thickness is not 1.0")
        except:
            pass
        try:
            if df.at[i, "Wall thickness Chain stay"]!=1.2:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " Chain stay thickness is not 1.2")
        except:
            pass
        try:
            if df.at[i, "OFFSET_AT_BB dt"]!=0:
                valid=0
                if printcodes==1:
                    print("Model " + str(i) + " Has DT offset")
        except:
            pass
        if valid==1:
            validmodels.append(i)
    