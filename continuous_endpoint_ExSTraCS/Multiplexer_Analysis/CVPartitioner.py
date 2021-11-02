#-------------------------------------------------------------------------------
# Name:        CVPartitioner.py
# Purpose:     Designed to Partition up a data set into X Parts for X-Fold Cross Validation : Note - only can handle balanced data sets of a size divisible by X.
# Author:      Ryan Urbanowicz
# Created:     08/16/2013
# Updated:     
# Status:      FUNCTIONAL * FINAL * v.1.0
# Notes:       Assume that a folder has been made to save these data files.
#              Filepath is the path up to the folder mentioned above plus a unique name up to but not including .txt
#              Assumes that the dataset is perfectly dividable by 20.
#-------------------------------------------------------------------------------
#!/usr/bin/env python
import os
import random

def CVPart(dataSetName, filePath, numPartitions):
    #Load Dataset--------------------------------------------------
    f = open(dataSetName, 'r')
    datasetList = []
    headerList = f.readline().rstrip('\n')  #strip off first row
    for line in f:
        lineList = line.strip('\n').split('\t')
        datasetList.append(lineList)
    f.close()
    #--------------------------------------------------------------
    dataLength = len(datasetList)
    
    #Characterize Phenotype----------------------------------------------------------------------------
    discretePhenotype = True
    phenotypeRef = len(datasetList[0])-1
    discreteAttributeLimit = 10
    inst = 0
    classDict = {}
    while len(list(classDict.keys())) <= discreteAttributeLimit and inst < dataLength:  #Checks which discriminate between discrete and continuous attribute
        target = datasetList[inst][phenotypeRef]
        if target in list(classDict.keys()):  #Check if we've seen this attribute state yet.
            classDict[target] += 1
        else: #New state observed
            classDict[target] = 1
        inst += 1
        
    if len(list(classDict.keys())) > discreteAttributeLimit:
        #print "Continuous Phenotype Detected"
        discretePhenotype = False
    else:
        #print "Discrete Phenotype Detected"
        pass
    #---------------------------------------------------------------------------------------------------

    CVList = [] #stores all partitions
    for x in range(numPartitions):
        CVList.append([])

    if discretePhenotype:
        masterList = []
        classKeys = list(classDict.keys())
        for i in range(len(classKeys)):
            masterList.append([])
        for i in datasetList:
            notfound = True
            j = 0
            while notfound:
                if i[phenotypeRef] == classKeys[j]:
                    masterList[j].append(i)
                    notfound = False
                j += 1
        
        #Randomize class instances before partitioning------------------
        from random import shuffle
        for i in range(len(classKeys)):
            shuffle(masterList[i])
        #---------------------------------------------------------------
            
        for currentClass in masterList:
            currPart = 0
            counter = 0
            for x in currentClass:
                CVList[currPart].append(x)
                counter += 1
                currPart = counter%numPartitions
                
        makePartitions(CVList,numPartitions,filePath,headerList)
            
    else: #Continuous Endpoint
        from random import shuffle
        shuffle(datasetList)  
        currPart = 0
        counter = 0
        for x in datasetList:
            CVList[currPart].append(x)
            counter += 1
            currPart = counter%numPartitions
        
        makePartitions(CVList,numPartitions,filePath,headerList)
        

def makePartitions(CVList,numPartitions,filePath,headerList):         
    for part in range(numPartitions): #Builds CV data files.
        if not os.path.exists(filePath+'_CV_'+str(part)+'_Train.txt') or not os.path.exists(filePath+'_CV_'+str(part)+'_Test.txt'):
            print("Making new CV files:  "+filePath+'_CV_'+str(part))
            trainFile = open(filePath+'_CV_'+str(part)+'_Train.txt','w')
            testFile = open(filePath+'_CV_'+str(part)+'_Test.txt','w')
            testFile.write(headerList + "\n")
            trainFile.write(headerList + "\n")  

            testList=CVList[part]
            trainList=[]
            tempList = []                 
            for x in range(numPartitions): 
                tempList.append(x)                            
            tempList.pop(part)

            for v in tempList: #for each training partition
                trainList.extend(CVList[v])    
        
            for i in testList: #Write to Test Datafile
                tempString = ''
                for point in range(len(i)):
                    if point < len(i)-1:
                        tempString = tempString + str(i[point])+"\t"
                    else:
                        tempString = tempString +str(i[point])+"\n"                        
                testFile.write(tempString)
                      
            for i in trainList: #Write to Train Datafile
                tempString = ''
                for point in range(len(i)):
                    if point < len(i)-1:
                        tempString = tempString + str(i[point])+"\t"
                    else:
                        tempString = tempString +str(i[point])+"\n"                        
                trainFile.write(tempString)
                                                
            trainFile.close()
            testFile.close()  