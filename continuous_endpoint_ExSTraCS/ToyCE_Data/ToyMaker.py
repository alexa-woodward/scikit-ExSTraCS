'''
Created on Mar 26, 2015
Designed to generate some simple test datasets based on an existing discrete SNP datasets.
@author: Ryan
'''
#Patterns:
    #-linear continuous relationship between an atttribute and an endpoint
    #-linear relationship continuous att related to two random enpoint groups
    #-linear relatinoship continuous att related to 4 random endpoint groups
    #-Same as above but with other atts being continuous not discrete.
    
def loadData(dataFile):
    """ Load the data file. """     
  
    datasetList = []
    f = open(dataFile,'rU')
    headerList = f.readline().rstrip('\n').split('\t')   #strip off first row

    for line in f:
        lineList = line.strip('\n').split('\t')
        datasetList.append(lineList)
    f.close()

    return [headerList,datasetList]  #[header,dataarray]


def CE_Transform1(datasetList, CA_Ref, endpointRef):
    """  Simple linear pattern """
    #Get rid of existing dataset signal by permuting class column
    shuffle(datasetList[endpointRef])
    
    starting_value = 2.7
    increment = 0.3
    starting_endpoint = 0.34
    endpoint_increment = 0.01
    for i in range(len(datasetList)):
        datasetList[i][CA_Ref] = starting_value
        starting_value += increment
        datasetList[i][endpointRef] = starting_endpoint
        starting_endpoint += endpoint_increment
    return datasetList


def CE_Transform2(datasetList, CA_Ref, endpointRef):
    """ Linear attribute with upper and lower continuous endpoint groups """
    #Get rid of existing dataset signal by permuting class column
    shuffle(datasetList[endpointRef])
    samples = len(datasetList)
    midpoint = int(samples/2)
    
    starting_value = 2.7
    increment = 0.3
    
    for i in range(len(datasetList)):
        datasetList[i][CA_Ref] = starting_value
        starting_value += increment
        
        if i < midpoint:
            datasetList[i][endpointRef] = random.uniform(0,50)
        else:
            datasetList[i][endpointRef] = random.uniform(50.01,100)
            
    return datasetList


def CE_Transform3(datasetList, CA_Ref, endpointRef):
    """ Linear attribute with upper and lower continuous endpoint groups """
    #Get rid of existing dataset signal by permuting class column
    shuffle(datasetList[endpointRef])
    endpointchunks = 4
    samples = len(datasetList)
    singlechunk = int(samples/endpointchunks)
    
    starting_value = 2.7
    increment = 0.3
    
    for i in range(len(datasetList)):
        datasetList[i][CA_Ref] = starting_value
        starting_value += increment
        
        if i < singlechunk:
            datasetList[i][endpointRef] = random.uniform(0,50)
        elif i < singlechunk * 2:
            datasetList[i][endpointRef] = random.uniform(50.01,100)
        elif i < singlechunk * 3:
            datasetList[i][endpointRef] = random.uniform(100.01,150)
        else:
            datasetList[i][endpointRef] = random.uniform(150.01,200)
            
    return datasetList


def CVPart(numPartitions, datasetList, headerList, endpointRef,fileName):
    """ Given a data set, CVPart randomly partitions it into X random balanced 
    partitions for cross validation which are individually saved in the specified file. 
    filePath - specifies the path and name of the new datasets. """
    # Open the specified data file.
    dataLength = len(datasetList)   
        
    CVList = [] #stores all partitions
    for x in range(numPartitions):
        CVList.append([])
    
    if discretePhenotype:
        inst = 0
        classDict = {}
        while inst < dataLength:  #Checks which discriminate between discrete and continuous attribute
            target = datasetList[inst][endpointRef]
            if target in list(classDict.keys()):  #Check if we've seen this attribute state yet.
                classDict[target] += 1
            else: #New state observed
                classDict[target] = 1
            inst += 1
        
        masterList = []
        classKeys = list(classDict.keys())
        for i in range(len(classKeys)):
            masterList.append([])
        for i in datasetList:
            notfound = True
            j = 0
            while notfound:
                if i[endpointRef] == classKeys[j]:
                    masterList[j].append(i)
                    notfound = False
                j += 1
        
        #Randomize class instances before partitioning------------------
        #from random import shuffle
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
                
        makePartitions(CVList,numPartitions,fileName,headerList)
        
    else: #Continuous Endpoint
        #from random import shuffle
        shuffle(datasetList)  
        currPart = 0
        counter = 0
        for x in datasetList:
            CVList[currPart].append(x)
            counter += 1
            currPart = counter%numPartitions
        
        makePartitions(CVList,numPartitions,fileName,headerList)
        
        
def makePartitions(CVList,numPartitions,fileName,headerList): 
    part = 1        
    if not os.path.exists(fileName+'_CV_'+str(part)+'_Train.txt') or not os.path.exists(fileName+'_CV_'+str(part)+'_Test.txt'):
        print("Making new CV files:  "+fileName+'_CV_'+str(part))
        trainFile = open(fileName+'_CV_'+str(part)+'_Train.txt','w')
        testFile = open(fileName+'_CV_'+str(part)+'_Test.txt','w')
        
        for i in range(len(headerList)):   
            if i < len(headerList)-1:
                testFile.write(headerList[i] + "\t")
                trainFile.write(headerList[i] + "\t")  
            else:
                testFile.write(headerList[i] + "\n")
                trainFile.write(headerList[i] + "\n") 

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


def WritePermutedDataset(headerList,datasetList,endpointRef, fileName):
    outputFile = open(fileName, 'w')
    outputFile.write('\t'.join(headerList) + '\n')

    permDataset = datasetList
    permColumn = headerList.index(classLabel)

    ### Shuffle the column to permute.
    from random import shuffle
    shuffle(permDataset[permColumn])

    for i in range(len(datasetList[0])):
        currList = []
        for j in range(len(headerList)):
            currList.append(permDataset[j][i])
            
        outputFile.write('\t'.join(currList) + '\n')
        
    outputFile.close()
        
def LoadToPermuteDataset(filename):
    #Have to load data transposed to perform endpoint permutation.
    datasetFile = open(filename, 'r')
    headerList = datasetFile.readline().rstrip('\n').split('\t')
    datasetList = []
    for i in range(len(headerList)):
        datasetList.append([])
        
    for line in datasetFile:
        lineList = line.strip('\n').split('\t')
        for i in range(len(headerList)):
            datasetList[i].append(lineList[i])
            
    datasetFile.close()
    return [headerList,datasetList]  

import random
import os
from random import shuffle
seedFileName = "ContinuousFullData" #"DiscreteFullData"
CA_Ref = 7
classLabel = 'Class'
numPartitions = 10
discretePhenotype = False

#open existing dataset 
#and load into list array
rawData = loadData(seedFileName+'.txt') #Load the raw data.
endpointRef = rawData[0].index(classLabel)
#Transform data as needed (add pattern)
tName = "Trans3"
#transData = CE_Transform1(rawData[1], CA_Ref, endpointRef)
#transData = CE_Transform2(rawData[1], CA_Ref, endpointRef)
transData = CE_Transform3(rawData[1], CA_Ref, endpointRef)

#Randomly select instances and generate a train and test dataset.  
CVPart(numPartitions, transData, rawData[0], endpointRef, seedFileName+'_'+tName)

print("Generating Permuted Datasets")
#Permuted Data file.
trainData = LoadToPermuteDataset(seedFileName+'_'+tName+'_CV_'+str(1)+'_Train.txt')
fileName = seedFileName+'_'+tName+'_CV_'+str(1)+'_Train_Permuted.txt'
WritePermutedDataset(trainData[0],trainData[1],endpointRef,fileName)

testData = LoadToPermuteDataset(seedFileName+'_'+tName+'_CV_'+str(1)+'_Test.txt')
fileName = seedFileName+'_'+tName+'_CV_'+str(1)+'_Test_Permuted.txt'
WritePermutedDataset(testData[0],testData[1],endpointRef,fileName)

print("Finished Generation")


