"""
Name: run_ExSTraCS_Sim_Main.py
Author: Ryan Urbanowicz
Date: 10/9/14
Description:  Run ExSTraCS implementation on all of the GH permuted data. All jobs are submitted as a set 'N' Cross-validation ExSTraCS runs to be run
        in series on a single node.
"""

def main(outputFolder, fileCheck, resubmitJobs):
    version = 'F2.3_ContEndPaper_AddFitness2Prediction6'
    makeCVDatasets = True  #Needs to be True the first time new simulated datasets are introduced. 
    permuted = True
    hours = 50 #Walltime limit (in hours)
    
    #Simulated Dataset Parameters#######################################################################################
    attNum = 20
    dataFolder = 'ToyBuilding' #'GAMETES_2.1__Datasets_2Het_Loc_2_Qnt_2_Pop_100000_CE'#  _CE, _CA, _CE_CA, _HH_CA 

    numPartitions = 10
    #####################################################################################################################

    onlyEKScores = 0 # 1=True, 0 = False
    onlyRC = 0 #1 is True, 0 is False

    #ExSTraCS Run Parameters#############################################################################################
    randomSeed = 1#'False' #Use a fixed random seed? If no put (False), if yes, set the random seed value as an integer.  
    popSize = 2000 #2000 #
    learningIterations = '10000.50000.100000.200000'#'10000.50000.100000.200000' # 
    #Attribute Tracking/Feedback ------------------
    doAttributeTracking = 1 #1 is True, 0 is False
    doAttributeFeedback = 1 #1 is True, 0 is False
    adjustedAT = 0 #1 is True, 0 is False
    #Expert Knowledge -----------------------------
    useExpertKnowledge = 1 #1 is True, 0 is False
    internalEK = 1   #NOTE I ADDED THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    filterAlgorithm = 'multisurf'# #ReliefF, SURF, SURFStar, MultiSURF
    #Rule Compaction ------------------------------
    doRuleCompaction = 1 #1 is True, 0 is False
    ruleCompactionMethod = 'QRF' 
    doPopulationReboot = 0 #1 is True, 0 is False 
    popRebootIteration = '200000'

    checkIteration = '200000'
    if doRuleCompaction == 1:
        checkIteration = 'RC_'+str(ruleCompactionMethod)+'_'+checkIteration
        
    seedFileName = "DiscreteFullData"#"ContinuousFullData" # "DiscreteFullData"#
    CA_Ref = 7
    classLabel = 'Class'
    numPartitions = 10
    discretePhenotype = False
    
    dataPath = '/idata/cgl/ryanu/datasets'   
    #open existing dataset 
    #and load into list array
    rawData = loadData(dataPath+'/'+dataFolder+'/'+seedFileName+'.txt') #Load the raw data.
    endpointRef = rawData[0].index(classLabel)
    #Transform data as needed (add pattern)
    tName = "Trans1"
    transData = CE_Transform1(rawData[1], CA_Ref, endpointRef)
    #transData = CE_Transform2(rawData[1], CA_Ref, endpointRef)
    #transData = CE_Transform3(rawData[1], CA_Ref, endpointRef)
                        
    #Save TransData 
    filename = dataPath+'/'+dataFolder+'/'+seedFileName+'_'+ tName+ '.txt'
    WriteOriginalDataset(rawData[0],transData,filename)
    dataIdentifier = seedFileName+'_'+ tName        
    pDataIdentifier = seedFileName+'_'+ tName + '_Permuted'
    #Generate Permuted Datasets
    fileNamePerm = dataPath+'/'+dataFolder+'/'+seedFileName+'_'+ tName+ '_Permuted.txt'
    rawPermuteData = LoadToPermuteDataset(filename)
    WritePermutedDataset(rawPermuteData[0],rawPermuteData[1],endpointRef,fileNamePerm)
        
    #####################################################################################################################################################
    if not fileCheck: #Run Simulation Data Analysis
        CVDatasetsFolder = dataFolder+'__CV_Data'
        if not os.path.exists(dataPath+'/'+CVDatasetsFolder):
            os.mkdir(dataPath+'/'+CVDatasetsFolder)  #top of heirarchy
            
        # Create folder Heirarchy in home directory
        if not os.path.exists(outputPath+outputFolder):
            os.mkdir(outputPath+outputFolder)  #top of heirarchy
                               
        print("output folder heirarchy made")
            
        jobCounter = 0 
        # For each data file under specified parameters run iterMDR.pl, and save output to txt file in associated folder.                

        #Make CV Datasets ####################################################################
        if makeCVDatasets:
            dataSetName = filename
            filePath = dataPath+'/'+CVDatasetsFolder+'/'+dataIdentifier
            CVPartitioner.CVPart(dataSetName, filePath, numPartitions)
        #####################################################################################
        
        #Make Configuration Files
        for part in range(int(numPartitions)):
            if useExpertKnowledge or onlyEKScores:
                if not os.path.exists(outputPath+filterAlgorithm+'_'+CVDatasetsFolder):
                    os.mkdir(outputPath+filterAlgorithm+'_'+CVDatasetsFolder)  #top of heirarchy
                if internalEK or onlyEKScores:
                    external_EK_Generation = 'None'
                    outEKFileName = outputPath+filterAlgorithm+'_'+CVDatasetsFolder+'/'
                else: 
                    external_EK_Generation = outputPath+filterAlgorithm+'_'+CVDatasetsFolder+'/'+dataIdentifier+'_CV_'+str(part)+'_Train_'+str(filterAlgorithm)+'_scores.txt'
                    outEKFileName = ''
            else:
                external_EK_Generation = 'None'
                outEKFileName = ''
                
            ############################################
            if permuted:
                trainDataName = dataPath+'/'+CVDatasetsFolder+'/'+dataIdentifier+'_Permuted_CV_'+str(part)+'_Train' #.txt not needed
                testDataName = dataPath+'/'+CVDatasetsFolder+'/'+dataIdentifier+'_Permuted_CV_'+str(part)+'_Test'   #.txt not needed
            else:   
                trainDataName = dataPath+'/'+CVDatasetsFolder+'/'+dataIdentifier+'_CV_'+str(part)+'_Train' #.txt not needed
                testDataName = dataPath+'/'+CVDatasetsFolder+'/'+dataIdentifier+'_CV_'+str(part)+'_Test'   #.txt not needed
            outputName = outputPath+outputFolder+'/'
            #Generate Configuration File ###############
            configFileName = outputPath+outputFolder+'/'+dataIdentifier+'_CV_'+str(part)+'_Train'+'_ExSTraCS_ConfigFile.txt'
            makeConfigFile(trainDataName, testDataName, outputName, outEKFileName, randomSeed, learningIterations, popSize, doAttributeTracking, doAttributeFeedback, adjustedAT, useExpertKnowledge, external_EK_Generation, doRuleCompaction, onlyRC, ruleCompactionMethod, doPopulationReboot, popRebootIteration, part, filterAlgorithm, onlyEKScores, configFileName)
            ############################################ 
        
        outputNameBase = outputPath+outputFolder+'/'+dataIdentifier
        configFileNameBase = outputPath+outputFolder+'/'+dataIdentifier+'_CV_'
        jobNameID = dataIdentifier
        submitJob(popRebootIteration, numPartitions, version, outputNameBase, configFileNameBase, jobNameID, hours)
        jobCounter +=1  
        
        print("Last job submitted, the end of the world is near")
        print(str(jobCounter) + ' jobs have been submitted.')

    #####################################################################################################################################################
    else: #run check to see if all runs have completed.
        CVDatasetsFolder = dataFolder+'__CV_Data'
        if not os.path.exists(dataPath+'/'+CVDatasetsFolder):
            os.mkdir(dataPath+'/'+CVDatasetsFolder)  #top of heirarchy
        counter = 0
        #Make CV Datasets ####################################################################
        if makeCVDatasets:
            dataSetName = filename
            filePath = dataPath+'/'+CVDatasetsFolder+'/'+dataIdentifier
            CVPartitioner.CVPart(dataSetName, filePath, numPartitions)

            dataSetName = fileNamePerm 
            filePath = dataPath+'/'+CVDatasetsFolder+'/'+pDataIdentifier
            CVPartitioner.CVPart(dataSetName, filePath, numPartitions)
            
         
        #####################################################################################        fileMissing = False
        for part in range(numPartitions):
            outputName = outputPath+outputFolder+'/'+dataIdentifier+'_CV_'+str(part)+'_Train_ExSTraCS_'+str(checkIteration)+'_PopStats.txt'
            if not os.path.exists(outputName):  
                print(outputName)
                fileMissing = True
                counter += 1
                
        if resubmitJobs and fileMissing:
            outputNameBase = outputPath+outputFolder+'/'+dataIdentifier
            configFileName = outputPath+outputFolder+'/'+dataIdentifier+'_CV_'+str(part)+'_Train'+'_ExSTraCS_ConfigFile.txt'
            configFileNameBase = outputPath+outputFolder+'/'+dataIdentifier+'_CV_'
            jobNameID = dataIdentifier
            submitJob(popRebootIteration, numPartitions, version, outputNameBase, configFileNameBase , jobNameID, hours)
                                          
        print(str(counter)+' Missing Files.')
                   
                         
def submitJob(popRebootIteration, numPartitions, version, outputNameBase, configFileNameBase , jobNameID, hours): 
    """ Submit Job to the cluster. """
    #MAKE CLUSTER JOBS###################################################################
    jobName = scratchPath+'_'+str(outputFolder)+'_'+jobNameID+'_run.pbs'                            
    pbsFile = open(jobName, 'w')
    pbsFile.write('#!/bin/bash -l\n') #NEW
    pbsFile.write('#PBS -A Williams\n') #NEW
    pbsFile.write('#PBS -q largeq\n')
    pbsFile.write('#PBS -N '+str(outputFolder)+'_'+jobNameID+'\n')
    pbsFile.write('#PBS -l walltime='+str(int(hours))+':00:00\n')
    pbsFile.write('#PBS -l nodes=1:ppn=1\n')
    pbsFile.write('#PBS -M Ryan.J.Urbanowicz\@dartmouth.edu\n\n')
    pbsFile.write('#PBS -o localhost:/idata/cgl/ryanu/logs\n\n')
    pbsFile.write('#PBS -e localhost:/idata/cgl/ryanu/logs\n\n')
    
    pbsFile.write('time python '+myHome+'/exstracs/'+str(version)+'/Run_Sum/run_ExSTraCS_Sim_Sub.py '+str(popRebootIteration)+' '+str(numPartitions)+' '+str(version)+' '+str(outputNameBase)+' '+str(configFileNameBase)+'\n')
    pbsFile.close()
    os.system('qsub '+jobName)    
    os.unlink(jobName)  #deletes the job submission file after it is submitted.
    #####################################################################################   
    
                   
def makeConfigFile(trainDataName, testDataName, outputName, outEKFileName, randomSeed, learningIterations, popSize, doAttributeTracking, doAttributeFeedback, adjustedAT, useExpertKnowledge, external_EK_Generation, doRuleCompaction, onlyRC, ruleCompactionMethod, doPopulationReboot, popRebootIteration, part, filterAlgorithm, onlyEKScores, configFileName):
    """ Construct Configuration File for CV Analysis """
    configFile = open(configFileName,'w')
    
    #Write to Config File#############################################################################################################
    configFile.write('offlineData=1# \n')
    configFile.write('trainFile='+str(trainDataName)+'# \n')
    configFile.write('testFile='+str(testDataName)+'# \n')
    configFile.write('internalCrossValidation=0# \n')
    
    configFile.write('outFileName='+str(outputName)+'# \n')
    configFile.write('randomSeed='+str(randomSeed)+'# \n')
    configFile.write('labelInstanceID=InstanceID# \n')
    configFile.write('labelPhenotype=Class# \n')
    configFile.write('discreteAttributeLimit=10# \n')
    configFile.write('labelMissingData=NA# \n')
    configFile.write('outputSummary=1# \n')
    configFile.write('outputPopulation=1# \n')
    configFile.write('outputAttCoOccur=1# \n')
    configFile.write('outputTestPredictions=1# \n')  
        
    configFile.write('trackingFrequency=0# \n')
    configFile.write('learningIterations='+str(learningIterations)+'# \n')
    
    configFile.write('N='+str(popSize)+'# \n') 
    configFile.write('nu=1# \n')
    configFile.write('chi=0.8# \n')
    configFile.write('upsilon=0.04# \n')
    configFile.write('theta_GA=25# \n')
    configFile.write('theta_del=20# \n')   
    configFile.write('theta_sub=20# \n')    
    configFile.write('acc_sub=0.99# \n')    
    configFile.write('beta=0.2# \n')         
    configFile.write('delta=0.1# \n')     
    configFile.write('init_fit=0.01# \n')
    configFile.write('fitnessReduction=0.1# \n')
    configFile.write('theta_sel=0.5# \n')
    configFile.write('RSL_Override=0# \n')

    configFile.write('doSubsumption=1# \n')
    configFile.write('selectionMethod=tournament# \n')
    
    configFile.write('doAttributeTracking='+str(doAttributeTracking)+'# \n')
    configFile.write('doAttributeFeedback='+str(doAttributeFeedback)+'# \n')
    configFile.write('adjustedAT='+str(adjustedAT)+'# \n')
    
    configFile.write('useExpertKnowledge='+str(useExpertKnowledge)+'# \n')
    configFile.write('external_EK_Generation='+str(external_EK_Generation)+'# \n')
    configFile.write('outEKFileName='+str(outEKFileName)+'# \n')
    
    configFile.write('filterAlgorithm='+str(filterAlgorithm)+'# \n')
    configFile.write('turfPercent=0.2# \n')
    configFile.write('reliefNeighbors=10# \n')
    configFile.write('reliefSampleFraction=1# \n')
    configFile.write('onlyEKScores='+str(onlyEKScores)+'# \n')
    
    configFile.write('doRuleCompaction='+str(doRuleCompaction)+'# \n')
    configFile.write('onlyRC='+str(onlyRC)+'# \n')
    configFile.write('ruleCompactionMethod='+str(ruleCompactionMethod)+'# \n')
    
    configFile.write('doPopulationReboot='+str(doPopulationReboot)+'# \n')
    configFile.write('popRebootIteration='+str(popRebootIteration)+'# \n')
    
    configFile.close()
    ########################################################################################################################                  

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
        datasetList[i][CA_Ref] = str(starting_value)
        starting_value += increment
        datasetList[i][endpointRef] = str(starting_endpoint)
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
        datasetList[i][CA_Ref] = str(starting_value)
        starting_value += increment
        
        if i < midpoint:
            datasetList[i][endpointRef] = str(random.uniform(0,50))
        else:
            datasetList[i][endpointRef] = str(random.uniform(50.01,100))
            
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
        datasetList[i][CA_Ref] = str(starting_value)
        starting_value += increment
        
        if i < singlechunk:
            datasetList[i][endpointRef] = str(random.uniform(0,50))
        elif i < singlechunk * 2:
            datasetList[i][endpointRef] = str(random.uniform(50.01,100))
        elif i < singlechunk * 3:
            datasetList[i][endpointRef] = str(random.uniform(100.01,150))
        else:
            datasetList[i][endpointRef] = str(random.uniform(150.01,200))
            
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


def WriteOriginalDataset(headerList,datasetList,fileName):
    outputFile = open(fileName, 'w')
    outputFile.write('\t'.join(headerList) + '\n')

    for i in range(len(datasetList)):
        currList = []
        for j in range(len(headerList)):
            currList.append(datasetList[i][j])
            
        outputFile.write('\t'.join(currList) + '\n')
        
    outputFile.close()
        

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
                
                        
if __name__=="__main__":
    import sys
    import os
    import CVPartitioner
    import random
    from random import shuffle
    fileCheck = False
    resubmitJobs = False
    
    classLabel = 'Class'
    discretePhenotype = False

    outputFolder = sys.argv[1] 
    if len(sys.argv) > 2:
        print("Running Simulation Analysis Completion Check.")
        fileCheck = True
    if len(sys.argv) > 3:
        print("Resubmitting Simulation Analysis Missing Jobs.")
        resubmitJobs = True #During Filecheck, resubmits jobs if they are missing.
        
    myHome = os.environ.get('HOME')
    outputPath = '/idata/cgl/ryanu/output/'
    dataPath = '/idata/cgl/ryanu/datasets'   
    scratchPath = '/global/scratch/ryanu/'      
    main(outputFolder, fileCheck, resubmitJobs)
    
# import random
# import os
# from random import shuffle
# seedFileName = "ContinuousFullData" #"DiscreteFullData"
# CA_Ref = 7
# classLabel = 'Class'
# numPartitions = 10
# discretePhenotype = False
# 
# dataPath = '/idata/cgl/ryanu/datasets'   
# #open existing dataset 
# #and load into list array
# rawData = loadData(seedFileName+'.txt') #Load the raw data.
# endpointRef = rawData[0].index(classLabel)
# #Transform data as needed (add pattern)
# tName = "Trans3"
# #transData = CE_Transform1(rawData[1], CA_Ref, endpointRef)
# #transData = CE_Transform2(rawData[1], CA_Ref, endpointRef)
# transData = CE_Transform3(rawData[1], CA_Ref, endpointRef)
# 
# #Randomly select instances and generate a train and test dataset.  
# CVPart(numPartitions, transData, rawData[0], endpointRef, seedFileName+'_'+tName)
# 
# print "Generating Permuted Datasets"
# #Permuted Data file.
# trainData = LoadToPermuteDataset(seedFileName+'_'+tName+'_CV_'+str(1)+'_Train.txt')
# fileName = seedFileName+'_'+tName+'_CV_'+str(1)+'_Train_Permuted.txt'
# WritePermutedDataset(trainData[0],trainData[1],endpointRef,fileName)
# 
# testData = LoadToPermuteDataset(seedFileName+'_'+tName+'_CV_'+str(1)+'_Test.txt')
# fileName = seedFileName+'_'+tName+'_CV_'+str(1)+'_Test_Permuted.txt'
# WritePermutedDataset(testData[0],testData[1],endpointRef,fileName)
# 
# print "Finished Generation"


