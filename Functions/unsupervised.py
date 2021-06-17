# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 23:33:22 2020

@author: Lyle
"""
import time
import dataFrameTools
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path


def catPCA():
    df=dataFrameTools.loadClassDF()
    pca=PCA(n_components=2)
    pca_result=pca.fit_transform(df)
    df['dim1'] = pca_result[:,0]
    df['dim2'] = pca_result[:,1] 
    df['class'] = df.index
    plotClass(df, title="PCA")
    
def cattSNE():
    df=dataFrameTools.loadClassDF()
    tsne = TSNE(n_components=2, verbose=1, perplexity=2, n_iter=5000)
    tsne_result = tsne.fit_transform(df)
    df['dim1'] = tsne_result[:,0]
    df['dim2'] = tsne_result[:,1] 
    df['class'] = df.index
    plotClass(df, title="PCA")
    
#Run Principle Component Analysis
def runPCA(plotclass=0):
    start_time=time.time()
    df=dataFrameTools.loadScaledDF()
    rawdf=dataFrameTools.loadDropDF()
    result=rPCA(df,rawdf) #Call the PCA run function
    if plotclass==1:
        plotClass(result,title="PCA")
    else:
        plotNum(result,title="PCA")
    print("PCA plotted in %s seconds" % (time.time() - start_time))
    return result
#Run T-Distribured Stochastic Neighbor Embedding
def runtSNE(plotclass=0, ppx=5):
    start_time=time.time()
    df=dataFrameTools.loadScaledDF()
    rawdf=dataFrameTools.loadDropDF()
    result=rtSNE(df,rawdf,ppx) #Call the tSNE run function
    if plotclass==1:
        plotClass(result,title="t-SNE")
    else:
        plotNum(result,title="t-SNE")
    print("tSNE plotted in %s seconds" % (time.time() - start_time))
    result.to_csv(Path("../Data/tsnedf.csv"))
    return result

# def runkmaeans(k=5):
#     start_time=time.time()
#     df=dataFrameTools.loadScaledDF()
#     rawdf=pd.read_csv("seldf.csv",index_col=0)
#     result=rkmeans(df,rawdf,k) #Call the tSNE run function
#     plotKmeans(result,title="t-SNE")
#     print("tSNE plotted in %s seconds" % (time.time() - start_time))
#     return result
    
#Run k-means and plot bar charts of the bike categories in each cluster
def plotClusterDist(numclusters=8):
    start_time=time.time()
    df=dataFrameTools.loadScaledDF()
    rawdf=dataFrameTools.loadDropDF()
    kmeans,inertia=rkMeans(df,rawdf,numclusters) #Call the k-means run function
    tallyKmeans(kmeans,numclusters)
    print("Distribution of Bike types in clusters printed in %s seconds" % (time.time() - start_time))
    
#Plot loss of K-means clustering for various cluster sizes
def plotKmeansLoss(numclusters):
    start_time=time.time()
    df=dataFrameTools.loadScaledDF()
    rawdf=dataFrameTools.loadDropDF()
    loss=np.zeros(numclusters)
    for i in range(1,numclusters+1):
        kmeans,inertia=rkMeans(df,rawdf,i)
        print("Iteration "+str(i)+" out of "+str(numclusters)+" complete")
        loss[i-1]=inertia
    plt.plot(range(1, numclusters+1), loss, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Loss')
    plt.show()
    print("Kmeans loss for various cluser counts plotted in %s seconds" % (time.time() - start_time))


def plotKmeans(x,y,resdf, title=""):
    plt.figure(figsize=(16,10))
    ax=sns.scatterplot(
        x="dim"+str(x), y="dim"+str(y),
        hue="class",
    #     palette=sns.color_palette("hls", 20),
        palette=["#9d6d00", "#903ee0", "#11dc79", "#f568ff", "#419500", "#013fb0", 
          "#f2b64c", "#007ae4", "#ff905a", "#33d3e3", "#9e003a", "#019085", 
          "#950065", "#afc98f", "#ff9bfa", "#83221d", "#01668a", "#ff7c7c", 
          "#643561", "#75608a"],
        data=resdf,
        legend="full",
        alpha=0.7
    )
    ax.set_title(title)
    
#Tally the numbers of each class of bike for a specific cluster
def tallyKmeans(kmeans,numclusters):

    
    tallydf=pd.DataFrame()
    categorylist=["ROAD","MTB","TRACK","OTHER","DIRT_JUMP","TOURING","CYCLOCROSS","POLO", "TIMETRIAL", "BMX", "CITY", "COMMUTER", "CRUISER", "HYBRID", "TRIALS", "CARGO", "GRAVEL", "TANDEM", "CHILDRENS", "FAT"]
    for category in categorylist:
        l=np.zeros(numclusters)
        for i in range(len(kmeans.values)):
            if kmeans.values[i,0]==category:
                l[kmeans.values[i,1]]+=1
        tallydf[category]=l
    tallydf.index=range(numclusters)
    tallydft=tallydf.T  
    numrows=2
    numcols=4
    fig,ax=plt.subplots(numrows,numcols,figsize=(20,10))
    for i in range(numclusters):
        axis=ax[i//numcols,i%numcols]
        sns.barplot(ax=axis,x=tallydft.index, y=i,data=tallydft)
        axis.set_xlabel(axis.get_xlabel(), fontsize=10)
        # axis.set_ylabel(axis.get_ylabel(), fontsize=30)
        axis.set_title("Cluster "+str(i), fontsize=10)
        axis.set_xticklabels(axis.get_xticklabels(),rotation=90, fontsize=10)
        # axis.set_yticklabels(axis.get_yticklabels(), fontsize=30)
    fig.suptitle("Bike Class Distribution by Cluster", fontsize=30)
    fig.tight_layout()
    fig.show()

    numrows=4
    numcols=5
    fig,ax=plt.subplots(numrows,numcols,figsize=(25,20))
    print(len(categorylist))
    for i in range(len(categorylist)):
        axis=ax[i//numcols,i%numcols]
        sns.barplot(ax=axis,x=tallydf.index, y=categorylist[i],data=tallydf)
        axis.set_xlabel(axis.get_xlabel(), fontsize=10)
        # axis.set_ylabel(axis.get_ylabel(), fontsize=30)
        axis.set_title(categorylist[i], fontsize=10)
        axis.set_xticklabels(axis.get_xticklabels(), fontsize=10)
    fig.suptitle("Cluster Distribution by Bike Class", fontsize=30)
    fig.tight_layout()
    fig.show()
    
    # fig,ax=plt.figure(figsize=(20,20))
    # ax = sns.countplot(x="predictions", hue="class",data=kmeans)
    # ax.tick_params(axis='both', which='major', labelsize=13)
    # ax.set_xlabel(ax.get_xlabel(), fontsize=30)
    # ax.set_ylabel(ax.get_ylabel(), fontsize=30)
    # plt.title("Bike Class Distribution by Cluster: " + str(cluster), fontsize=40)
    # plt.tight_layout()
    # plt.show()
    # fig,ax=plt.figure(figsize=(20,20))
    # ax = sns.countplot(x="class", hue="predictions",data=kmeans)
    # ax.tick_params(axis='both', which='major', labelsize=13)
    # ax.set_xlabel(ax.get_xlabel(), fontsize=30)
    # ax.set_ylabel(ax.get_ylabel(), fontsize=30)
    # plt.title("Cluster Distribution by Bike Class: " + str(cluster), fontsize=40)
    # plt.tight_layout()
    # plt.show()


def rPCA(df,rawdf):
    pca=PCA(n_components=2)
    pca_result=pca.fit_transform(df)
    PCAdf=pd.DataFrame()
    PCAdf['class']=rawdf.loc[df.index]["BIKESTYLE"]
    PCAdf['dim1'] = pca_result[:,0]
    PCAdf['dim2'] = pca_result[:,1] 
    
    return PCAdf
def rtSNE(df,rawdf,ppx):
    tsne = TSNE(n_components=2, verbose=1, perplexity=ppx, n_iter=5000)
    tsne_results = tsne.fit_transform(df)
    tsnedf=pd.DataFrame() 
    tsnedf['class']=rawdf.loc[df.index]["BIKESTYLE"]
    tsnedf['dim1'] = tsne_results[:,0]
    tsnedf['dim2'] = tsne_results[:,1]
    return tsnedf

#runs kmeans algorithm
def rkMeans(df,rawdf, numclusters):
    kmeans = KMeans(n_clusters=numclusters,n_init=50, max_iter=1000, random_state=0)
    kmeans.fit(df)
    inertia=kmeans.inertia_
    kmeansdf=pd.DataFrame() 
    kmeansdf['class']=rawdf.loc[df.index]["BIKESTYLE"]
    kmeansdf['predictions']=kmeans.labels_
    return kmeansdf,inertia
def plotClass(resdf, title=""):
    print(resdf["class"][1])
    plt.figure(figsize=(7.5,7.5))
    ax=sns.scatterplot(
        x="dim1", y="dim2",
        hue="class",
    #     palette=sns.color_palette("hls", 20),
        # palette=["#9d6d00", "#903ee0", "#11dc79", "#f568ff", "#419500", "#013fb0", 
        #   "#f2b64c", "#007ae4", "#ff905a", "#33d3e3", "#9e003a", "#019085", 
        #   "#950065", "#afc98f", "#ff9bfa", "#83221d", "#01668a", "#ff7c7c", 
        #   "#643561", "#75608a"],
        palette=['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#0000e5', '#a9a9a9', '#000000'],
        data=resdf,
        legend="full",
        alpha=1,
        s=10
    )
    # ax.set_title(title)
    ax.legend(labelspacing=0.1,handletextpad=0.1,markerscale=1.4,fontsize='xx-large',frameon=False)
    plt.setp(ax.get_legend().get_texts(), fontsize='10') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='12') # for legend title
    plt.show()
    
def plotNum(resdf, title=""):
    resdf["Model Number"]=resdf.index
    plt.figure(figsize=(7.5,7.5))
    ax=sns.scatterplot(
        x="dim1", y="dim2",
        hue="Model Number",
        palette=sns.color_palette("flare", as_cmap=True),
        # palette=["#9d6d00", "#903ee0", "#11dc79", "#f568ff", "#419500", "#013fb0", 
        #   "#f2b64c", "#007ae4", "#ff905a", "#33d3e3", "#9e003a", "#019085", 
        #   "#950065", "#afc98f", "#ff9bfa", "#83221d", "#01668a", "#ff7c7c", 
        #   "#643561", "#75608a"],
        # palette=['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#0000e5', '#a9a9a9', '#000000'],
        data=resdf,
        # legend="full",
        alpha=1,
        s=8
    )
    # ax.set_title(title)
    # ax.legend(labelspacing=0.1,handletextpad=0.1,markerscale=1.4,fontsize='xx-large',frameon=False)
    plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
    plt.show()
    
    