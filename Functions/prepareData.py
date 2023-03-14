# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 21:01:27 2020

@author: Lyle Regenwetter (regenwet@mit.edu)
"""

import dataFrameTools
import time
import paramRedux


def prepareData(dataset=""):
    start_time = time.time()
    # genxmldf.genxmldf() #This line to (re)generate the full raw dataframe from bcad files
    
    paramRedux.paramRedux(dataset)  #This line to (re)generate the reduced parameter space dataframe from raw dataframe
    dropdf=dataFrameTools.loadDropDF(dataset)
    #turn on intermediates to save a few intermediate dataframes from partway through the processing
    dataFrameTools.processDF(dropdf, intermediates=1, dataset=dataset)
    # dataFrameTools.normalizeDF()
    # dataFrameTools.exportCorrDF(0, "kendall", dataset)
    # dataFrameTools.exportCorrDF(0, "pearson", dataset)
    print("Successfully Prepared BIKED Data!")
    print("Total Execution time: %s seconds" % (time.time() - start_time))

if __name__ == '__main__':
    prepareData("micro")


