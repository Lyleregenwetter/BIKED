# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 20:20:31 2020

@author: Lyle
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import time
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from scipy.spatial import KDTree
# import checkdf
def normalizeDF(save=1): #Normalize BIKED Data
    start_time = time.time()
    ImpDF=loadImpDF()
    min_max_scaler = preprocessing.MinMaxScaler()
    # min_max_scaler = preprocessing.StandardScaler()
    x_scaled = min_max_scaler.fit_transform(ImpDF.values)
    # x_scaled=min_max_scaler.transform
    scdf = pd.DataFrame(x_scaled, columns=ImpDF.columns,index=ImpDF.index.values)
    if save==1:
        scdf.to_csv(Path('../Data/BIKED_normalized.csv')) 
        print("Scaled Dataframe Successfully exported to CSV in  %s seconds" % (time.time() - start_time))
    return scdf

    
def standardizeReOH(df):
    Impdf=loadImpDF()
    for col in Impdf.columns:
        if col not in df.columns:
            df[col]=[0]*len(df.index)
    return df

def deNormalizeDF(file="vaegendf", save=1):
    start_time = time.time()
    df=pd.read_csv(Path("../data/"+file+".csv"), index_col=0)
    ImpDF=loadImpDF()
    min_max_scaler = preprocessing.MinMaxScaler()
    # min_max_scaler = preprocessing.StandardScaler()
    min_max_scaler.fit(ImpDF.values)
    invscaled=min_max_scaler.inverse_transform(df)   
    invdf = pd.DataFrame(invscaled, columns=df.columns, index=df.index)
    if save==1:
        invdf.to_csv(Path("../data/"+file+"_Invsc.csv")) 
        print("Inverse Scaled Dataframe Successfully exported to CSV in  %s seconds" % (time.time() - start_time))
    return invdf


def deOH(file="vaegendf"): #Revert one-hot encoding back to categorical features
    df=pd.read_csv(Path("../data/"+file+"_Invsc.csv"), index_col=0) 
    newdf=pd.DataFrame()
# we will keep track of the most probable category for each model and categorical feature in a dict with keys containing model/feature pairs
    maxprobs={} 
    #Convert from one hot to non-onehot
    for column in df.columns:
        if ' OHCLASS: ' in column:
            front,back=column.split(' OHCLASS: ')
            for i in df.index:
                prob=df.at[i,column]
                if (i,front) in maxprobs:
                    if prob>maxprobs[(i,front)]:
                        maxprobs[(i,front)]=prob
                        newdf.at[i,front]=back
                else:
                    maxprobs[(i,front)]=prob
                    newdf.at[i,front]=back
        else:
            newdf[column]=df[column]
    #Make sure types of deohdf match the types of seldf
    dtypedf=pd.read_csv(Path("../Data/BIKED_datatypes.csv"), index_col=0).T
    # print(newdf.dtypes["DOWNTUBE1SnSCheck"])
    # print(newdf["DOWNTUBE1SnSCheck"])
    # print(newdf["DOWNTUBE1SnSCheck"].map({'False':False, 'True':True}))
    for column in newdf.columns:
        if dtypedf.at["type",column]=="bool":
            if newdf.dtypes[column]==np.float64:
                newdf[column] = newdf[column].round().astype('bool')
            else:
                newdf["DOWNTUBE1SnSCheck"].map({'False':False, 'True':True})
        if dtypedf.at["type",column]=="int64":
            if newdf.dtypes[column]==np.float64:
                newdf[column] = newdf[column].round().astype('int64')
            else:
                newdf[column] = pd.to_numeric(newdf[column]).astype('int64')
    newdf.to_csv(Path(Path("../data/"+file+"_DeOH.csv")))
    return newdf

#Take a dataframe of features and insert features into a baseline bikecad file to generate bcad files
#BikeCAD files are XML files with the .bcad extension
def genBCAD(file="vaegen"):
    df=pd.read_csv(Path("../data/"+file+"_DeOH.csv"), index_col=0) 
    # VAEFunctions.removeFiles("../Generated BCAD Files/Files")
    for modelidx in df.index: #loop over the models in the dataframe
        print(modelidx)
        count=0
        sourcefile = open(Path("PlainRoadbikestandardized.txt"), 'r') 
        targetfile= open(Path("../Generated BCAD Files/Files/" + str(modelidx) + ".bcad"), 'w')
        lines = sourcefile.readlines()
        linecount=0
        for line in lines: #Loop over the lines of the bcad file
            linecount+=1
            if linecount>4: #ignore first 4 lines of the bcad file
                param = find_between(line, "<entry key=\"", "\">")
                if param.endswith("mmInch"): #Manually set all units to mm
                    targetfile.writelines("<entry key=\""+param+"\">"+"1"+"</entry>\n")
                if param in df.columns: #if this line of the bcad file exists in the datafram column labels
                    if pd.isnull(df.at[modelidx,param]): #Don't want to insert nan values, leave blank instead
                        # targetfile.writelines("<entry key=\""+param+"\">"+"</entry>\n")
                        pass
                    elif type(df.at[modelidx,param])==np.bool_: #Bikecad wants "true" and "false" lower case
                        if df.at[modelidx,param]==True:
                            targetfile.writelines("<entry key=\""+param+"\">"+"true"+"</entry>\n")
                        else:
                            targetfile.writelines("<entry key=\""+param+"\">"+"false"+"</entry>\n")
                    # elif type(df.at[modelidx,param])==np.float64:
                    #     targetfile.writelines("<entry key=\""+param+"\">"+str(df.at[modelidx,param])+"</entry>\n")
                    elif type(df.at[modelidx,param])==np.float64 and df.at[modelidx,param].is_integer(): 
                        targetfile.writelines("<entry key=\""+param+"\">"+str(int(df.at[modelidx,param]))+"</entry>\n")
                    else:    #This is the default case, we insert the value into the bcad file
                        targetfile.writelines("<entry key=\""+param+"\">"+str(df.at[modelidx,param])+"</entry>\n")
    #                 df=df.drop(param,axis=1)
                    count+=1
                else:
                    targetfile.writelines(line)
            else:
                targetfile.writelines(line)
        sourcefile.close()
        targetfile.close()

def loadScaledDF():
    start_time = time.time()
    df=pd.read_csv(Path("../Data/BIKED_normalized.csv"), index_col=0) 
    print("Loaded Scaled Dataframe in  %s seconds" % (time.time() - start_time))
    return df
    
def loadVAEGenDF():
    start_time = time.time()
    df=pd.read_csv(Path("../Data/VAEGendf.csv"), index_col=0)
    print("Loaded VAE-Generated Dataframe in  %s seconds" % (time.time() - start_time))
    return df


def exportCorrDF(fvs=0, method='cosine'):
    start_time = time.time()
    ImpDF=loadImpDF()
    if fvs==1:
        ImpDF=ImpDF.T
    if method=="pearson" or method=="kendall" or method=="spearman":
        corrdf=ImpDF.corr(method =method)
    else:
        corrarr=cosine_similarity(ImpDF)
        corrdf=pd.DataFrame(data=corrarr,index=ImpDF.index.values, columns=ImpDF.index.values)
    filepath=Path('../Data/corrdf.csv')
    if fvs==0:
        corrdf.to_csv(filepath) 
    else:
        corrdf.to_csv(filepath) 
    print("Correlation Dataframe Successfully exported to CSV in  %s seconds" % (time.time() - start_time))
    
def loadDF():
    start_time = time.time()
    df=pd.read_csv(Path("../Data/df.csv", index_col=0)) 
    print("Loaded Dataframe in  %s seconds" % (time.time() - start_time))
    return df
    
def loadOHDF():
    start_time = time.time()
    df=pd.read_csv(Path("../Data/OHdf.csv"), index_col=0)
    print("Loaded One-Hot Dataframe in  %s seconds" % (time.time() - start_time))
    return df

def loadDropDF():
    start_time = time.time()
    df=pd.read_csv(Path("../Data/BIKED_reduced.csv"), index_col=0) 
    print("Loaded Reduced Parameter Space Dataframe in  %s seconds" % (time.time() - start_time))
    return df

def loadClassDF(): 
    start_time = time.time()
    df=pd.read_csv(Path("../Data/classdf.csv"), index_col=0)
    print("Loaded Class  Dataframe in  %s seconds" % (time.time() - start_time))
    return df

def loadImpDF():
    start_time = time.time()
    df=pd.read_csv(Path("../Data/BIKED_processed.csv"), index_col=0)
    print("Loaded Imputed Dataframe in  %s seconds" % (time.time() - start_time))
    return df

def processDF(dropdf, intermediates=0):
    dropdf=dropClasses(dropdf) #Remove Class Labels from processed DF
    
    OHdf=convertOneHot(dropdf, save=intermediates)
    
    
    # Uncomment the following 4 lines to select only SHAP features
    # shapdf=pd.read_csv(Path("SHAPdf.csv"), index_col=0)
    # shapcols=shapdf["col_name"][:50]
    # OHdf=OHdf[shapcols.values]
    # OHdf.to_csv(Path("../Data/OHdf.csv"))
    
    imputeNan(OHdf)
    normalizeDF()
    # getDataCounts(seldf)
    if intermediates==1:
        getclassdf()
    
    # getDataCounts(OHdf)
    print("Dataframe Successfully exported to CSV")


def dropClasses(df):
    #Drop any parameters not desired in the application
    df=df.drop("BIKESTYLE", axis=1)
    return df

def dropData(df):
    # df=df.drop(list(range(101,4801)), errors='ignore')
    df.dropna(axis=0,how='all',inplace=True)
    df.dropna(axis=1,how='all',inplace=True)
    df=df.loc[:, ~(df == df.iloc[0]).all()]
    return df
 


def getclassdf():
    df=loadScaledDF()
    df=df.astype('float64')
    dropdf=loadDropDF()  
    df["BIKESTYLE"]=dropdf["BIKESTYLE"]    
    classdf=df.groupby("BIKESTYLE").median()
    indices=[]
    for style in classdf.index:
        styledf=df[df["BIKESTYLE"] ==style]
        styledf=styledf.drop("BIKESTYLE", axis=1)
        kdb=KDTree(styledf.values)
        num=kdb.query(classdf.loc[style],k=1)[-1]
        indices.append(styledf.index[num])
    classdf.to_csv(Path("../Data/classdf.csv"))    
    meddf=pd.DataFrame(index=classdf.index, columns=["medidx"],data=indices)
    meddf.to_csv(Path("../Data/meddf.csv"))

   
def imputeNan(df): #Impute missing Values and remove outliers

    start_time = time.time()
    flag=0#flag==0 for simple imputer, flag==1 for KNN imputer
    
    
    #The Following lines can be used to remove outliers past a certain SD
    
    # print(np.abs(stats.zscore(df.astype(float)) > 3).all(axis=1))
    # df=df[(np.abs(stats.zscore(df)) < 3).all(axis=1)] #Uncomment this for outlier cutoff
    # df=df[(np.abs(stats.zscore(df)) < 3).all(axis=1)] #Uncomment this for outlier cutoff
    
    
    #Remove values with magnitude higher then preset cutoff 
    cutoff=10000
    nan_value = float("NaN")
    df=df.apply(lambda x: [y if -cutoff<=y <= cutoff else nan_value for y in x])

    #Impute     
    if flag==0:
        imp = SimpleImputer(missing_values=np.nan,strategy='mean')
        imp=imp.fit_transform(df)
        impdf=pd.DataFrame(data=imp, index=df.index.values, columns=df.columns)
        # impdf = df.fillna(df.mean())
    else:
        imp = KNNImputer(n_neighbors=2)
        imp=imp.fit_transform(df)
        impdf=pd.DataFrame(data=imp, index=df.index.values, columns=df.columns)
    dtypedf=pd.read_csv(Path("../Data/BIKED_datatypes.csv"), index_col=0).T
    for column in impdf.columns:
        if ' OHCLASS: ' in column:
            front,back=column.split(' OHCLASS: ')
        else:
            front=column
        if dtypedf.at["type",front]=="int64":
            impdf[column] = impdf[column].round().astype('int64')
    impdf.to_csv(Path('../Data/BIKED_processed.csv')) 
    print("Finished imputing Nan values in  %s seconds" % (time.time() - start_time))



def convertOneHot(df, save=1):
    start_time=time.time()
    colstoOH=[]
    count=0
    colstoOH=[]
    dtypedf=pd.read_csv(Path("../data/BIKED_datatypes.csv"), index_col=0).T
    for col in df.columns:
        if dtypedf.at["type",col] =="str" or dtypedf.at["type",col]=="object":
            colstoOH.append(col)
            count=count+1
    print("One-hot encoding " + str(count) + " features")
    for col in colstoOH:
        df=pd.get_dummies(df, prefix_sep=' OHCLASS: ', columns=[col], dtype=np.bool_)
        try: 
            del df[str(col)+' OHCLASS: ']
        except:
            pass
    OHdtypes=df.dtypes
    count=0
    for col in df.columns:
        if col in dtypedf.columns:
            count+=1
            OHdtypes[col]=dtypedf.at["type",col]
    OHdtypes.to_csv("../Data/BIKED_processed_datatypes.csv",header=["type"])
    if save==1:
        df.to_csv(Path('../Data/OHdf.csv'))
    
    countdf=df.dtypes.value_counts()
    print("Onehot Completed in %s seconds" % (time.time() - start_time))
    return df

def interpolate(df, idx1, idx2, steps):
#     empty=pd.Series([Nan]*20, index=df.columns)
    df=df.iloc[[idx1,idx2]]
    for i in range(steps):
        df=df.append(pd.Series(name="i"+str(i)))
    newindices=[idx1]+["i"+str(i) for i in range(steps)]+[idx2]
    df=df.loc[newindices,:]
    df=df.interpolate(axis=0)
    df.to_csv(Path("../Data/interpolatedf.csv"))
    return df

    

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
    
def getDataCounts(df):
    countdf=df["type"].value_counts()
        # countdf=df.dtypes.value_counts()
    print(countdf/countdf.sum()*100)
    print(countdf)