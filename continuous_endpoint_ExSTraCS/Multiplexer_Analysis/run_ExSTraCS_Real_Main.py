"""
Name: run_ExSTraCS_Real_Main.py
Author: Ryan Urbanowicz
Date: 10/10/14
Description: Organizes and submits a real dataset analysis using ExSTraCS to the Discovery cluster.
"""


def main(outputFolder, fileCheck, resubmitJobs):
    #Full analysis 1.) OrigData = True, 2.) OrigData = False & permutedData = False, 3.) OrigData = False & permutedData = True
    analysisFolder = 'Multiplexer' #Run_Sum
    version = 'F2.2_Paper'

    #Data Parameters##################################################################################################################
    hours = 200 #Walltime limit (in hours)
    permutationAnalysis = False
    permutations = 1000 #1000
    justTrain = True
    
    dataName = '70Multiplexer_Data_20000'#'135Multiplexer_Data_20000'#'37Multiplexer_Data_2000'#'11Multiplexer_Data_Complete'  #file name without extension
    classLabel = 'Class'
    projectFolder = 'Multiplexer' 
    dataFolder = 'OriginalData'
    permuteFolder = 'PermutedData'
    numPartitions = 10

    onlyEKScores = 0 # 1=True, 0 = False
    onlyRC = 0 #1 is True, 0 is False

    #ExSTraCS Run Parameters#############################################################################################
    offlineData = 1 #Switch on/off multiplexer
    trackingFrequency = 1000
    randomSeed = 1#'False' #Use a fixed random seed? If no put (False), if yes, set the random seed value as an integer.  
    popSize = 10000 #2000 #
    nu = 10
    learningIterations = '10000.50000.100000.200000.500000'#'10000.50000.100000.200000' # 
    #Attribute Tracking/Feedback ------------------
    doAttributeTracking = 1 #1 is True, 0 is False
    doAttributeFeedback = 1 #1 is True, 0 is False
    adjustedAT = 0 #1 is True, 0 is False
    #Expert Knowledge -----------------------------
    useExpertKnowledge = 1 #1 is True, 0 is False
    internalEK = 1
    filterAlgorithm = 'multisurf'# #ReliefF, SURF, SURFStar, MultiSURF
    externalEKManualPath = None #Probably won't need but here incase I want to specify EK for a one time training run.
    
    #Rule Compaction ------------------------------
    doRuleCompaction = 1 #1 is True, 0 is False
    ruleCompactionMethod = 'QRF' 
    
    doPopulationReboot = 0 #1 is True, 0 is False 
    popRebootIteration = '10000'

    checkIteration = '10000'
    if doRuleCompaction == 1:
        checkIteration = 'RC_'+str(ruleCompactionMethod)+'_'+checkIteration
        
     
    if not permutationAnalysis: #CV Analysis of True Data
        if justTrain:
            #Output File Preparation------------------------------------------
            if not os.path.exists(outputPath + outputFolder):
                os.mkdir(outputPath + outputFolder)  #top of heirarchy
            if not os.path.exists(outputPath + outputFolder+'/'+dataFolder):
                os.mkdir(outputPath + outputFolder+'/'+dataFolder)  #top of heirarchy
                
            trainDataName = dataPath+projectFolder+'/'+dataFolder+'/'+dataName
            testDataName = 'None'
            outputName = outputPath + outputFolder+'/'+dataFolder+'/'
                
            if useExpertKnowledge or onlyEKScores:
                if not os.path.exists(outputPath+filterAlgorithm+'_'+dataName):
                    os.mkdir(outputPath+filterAlgorithm+'_'+dataName)  #top of heirarchy
                if internalEK or onlyEKScores:
                    external_EK_Generation = 'None'
                    outEKFileName = outputPath+filterAlgorithm+'_'+dataName+'/'
                else: 
                    external_EK_Generation = externalEKManualPath
                    outEKFileName = ''
            if not fileCheck:
                #Generate Configuration File ###############
                configFileName = outputPath+outputFolder+'/'+dataFolder+'/'+dataName+'_ExSTraCS_ConfigFile.txt'
                makeConfigFile(trainDataName, testDataName, outputName, outEKFileName, randomSeed, learningIterations, popSize, doAttributeTracking, doAttributeFeedback, adjustedAT, useExpertKnowledge, external_EK_Generation, doRuleCompaction, onlyRC, ruleCompactionMethod, doPopulationReboot, popRebootIteration, filterAlgorithm, onlyEKScores, configFileName,classLabel,nu,offlineData,trackingFrequency)
                ############################################                 
                submitSingleJob(version,dataName,hours,configFileName)

            if fileCheck:
                missing = 0
                if not os.path.exists(outputName+dataName+'_ExSTraCS_'+str(checkIteration)+'_PopStats.txt'):
                    print(outputName+dataName+'_ExSTraCS_'+str(checkIteration)+'_PopStats.txt')
                    missing += 1
                    if resubmitJobs:
                        submitSingleJob(version,dataName,hours,configFileName)
                print(str(missing)+' Missing File.')  
                            
        else:
            realData = dataPath+projectFolder+'/'+dataFolder+'/'+dataName+'.txt'
            
            #Make CV Datasets-------------------------------------------------
            if not os.path.exists(dataPath+projectFolder+'/'+'CV_Data'):
                os.mkdir(dataPath+projectFolder+'/'+'CV_Data')  
            CVPath = dataPath+projectFolder+'/'+'CV_Data/'+dataName
            CVPartitioner.CVPart(realData, CVPath, numPartitions) 
            #-----------------------------------------------------------------
            
            #Output File Preparation------------------------------------------
            if not os.path.exists(outputPath + outputFolder):
                os.mkdir(outputPath + outputFolder)  #top of heirarchy
            if not os.path.exists(outputPath + outputFolder+'/'+dataFolder):
                os.mkdir(outputPath + outputFolder+'/'+dataFolder)  #top of heirarchy
            
            jobCounter = 0
            missingJobs = 0
            for part in range(numPartitions):
                trainDataName = CVPath +'_CV_'+str(part)+'_Train'#.txt'
                testDataName = CVPath +'_CV_'+str(part)+'_Test'#.txt'
                outputName = outputPath + outputFolder+'/'+dataFolder+'/'
                
                if useExpertKnowledge or onlyEKScores:
                    if not os.path.exists(outputPath+filterAlgorithm+'_'+dataName):
                        os.mkdir(outputPath+filterAlgorithm+'_'+dataName)  #top of heirarchy
                    if internalEK or onlyEKScores:
                        external_EK_Generation = 'None'
                        outEKFileName = outputPath+filterAlgorithm+'_'+dataName+'/'
                    else: 
                        external_EK_Generation = outputPath+filterAlgorithm+'_'+dataName+'/'+dataName+'_CV_'+str(part)+'_Train_'+str(filterAlgorithm)+'_scores.txt'
                        outEKFileName = ''
            
                if not fileCheck:
                    #Generate Configuration File ###############
                    configFileName = outputPath+outputFolder+'/'+dataFolder+'/'+dataName+'_CV_'+str(part)+'_Train_ExSTraCS_ConfigFile.txt'
                    makeConfigFile(trainDataName, testDataName, outputName, outEKFileName, randomSeed, learningIterations, popSize, doAttributeTracking, doAttributeFeedback, adjustedAT, useExpertKnowledge, external_EK_Generation, doRuleCompaction, onlyRC, ruleCompactionMethod, doPopulationReboot, popRebootIteration, filterAlgorithm, onlyEKScores, configFileName,classLabel,nu, offlineData,trackingFrequency)
                    ############################################                 
                    submitJob(version,dataName,part,hours,configFileName)
                    jobCounter += 1
                if fileCheck:
                    if not os.path.exists(outputName+dataName+'_CV_'+str(part)+'_Train'+'_ExSTraCS_'+str(checkIteration)+'_PopStats.txt'):
                        print(outputName+dataName+'_CV_'+str(part)+'_Train'+'_ExSTraCS_'+str(checkIteration)+'_PopStats.txt')
                        missingJobs += 1
                        if resubmitJobs:
                            submitJob(version,dataName,part,hours,configFileName)
                            jobCounter += 1
                            
            if not fileCheck or resubmitJobs:
                print(str(jobCounter) + ' jobs have been submitted.')
            if fileCheck:
                print(str(missingJobs)+' Missing Files.')               
                   
    else: #Permutation Test Analysis
        #Generate Permuted Datasets
        #######################################################################################################
        realData = dataPath+projectFolder+'/'+dataFolder+'/'+dataName+'.txt'

        if not os.path.exists(dataPath+projectFolder+'/'+permuteFolder):
            os.mkdir(dataPath+projectFolder+'/'+permuteFolder)
        if not os.path.exists(dataPath+projectFolder+'/'+permuteFolder+'/'+'CV_Data'):
            os.mkdir(dataPath+projectFolder+'/'+permuteFolder+'/'+'CV_Data')
        LoadDataset(realData)
            
        #Output File Preparation------------------------------------------
        if not os.path.exists(outputPath + outputFolder):
            os.mkdir(outputPath + outputFolder)  #top of heirarchy
        if not os.path.exists(outputPath + outputFolder+'/'+permuteFolder):
            os.mkdir(outputPath + outputFolder+'/'+permuteFolder)  #top of heirarchy
        jobCounter = 0
        missingJobs = 0
        for i in range(permutations):
            dataFilePath = dataPath+projectFolder+'/'+permuteFolder+'/'+dataName+'_Permuted_'+str(i)+'.txt'
            WritePermutedDataset(dataFilePath,classLabel)
            #Make CV files-------------------------------------------------------------
            CVPath = dataPath+projectFolder+'/'+permuteFolder+'/'+'CV_Data/'+dataName+'_Permuted_'+str(i)
            CVPartitioner.CVPart(dataFilePath, CVPath, numPartitions) 
        
            for part in range(numPartitions):
                configFileName = outputPath+outputFolder+'/'+permuteFolder+'/'+dataName+'_Permuted_'+str(i)+'_CV_'+str(part)+'_ExSTraCS_ConfigFile.txt'
                trainDataName = dataPath+projectFolder+'/'+permuteFolder+'/CV_Data/'+dataName+'_Permuted_'+str(i)+'_CV_'+str(part)+'_Train'
                testDataName = dataPath+projectFolder+'/'+permuteFolder+'/CV_Data/'+dataName+'_Permuted_'+str(i)+'_CV_'+str(part)+'_Test'
                outputName = outputPath + outputFolder+'/'+permuteFolder+'/'
                
                if useExpertKnowledge or onlyEKScores:
                    if not os.path.exists(outputPath+filterAlgorithm+'_'+dataName):
                        os.mkdir(outputPath+filterAlgorithm+'_'+dataName)  #top of heirarchy
                    if not os.path.exists(outputPath+filterAlgorithm+'_'+dataName+'/'+permuteFolder):
                        os.mkdir(outputPath+filterAlgorithm+'_'+dataName+'/'+permuteFolder)  #top of heirarchy
                        
                    if internalEK or onlyEKScores:
                        external_EK_Generation = 'None'
                        outEKFileName = outputPath+filterAlgorithm+'_'+dataName+'/'+permuteFolder+'/'
                    else: 
                        external_EK_Generation = outputPath+filterAlgorithm+'_'+dataName+'/'+permuteFolder+'/'+dataName+'_Permuted_'+str(i)+'_CV_'+str(part)+'_Train_'+str(filterAlgorithm)+'_scores.txt'
                        outEKFileName = ''
                else:
                    external_EK_Generation = 'None'
                    outEKFileName = ''
                makeConfigFile(trainDataName, testDataName, outputName, outEKFileName, randomSeed, learningIterations, popSize, doAttributeTracking, doAttributeFeedback, adjustedAT, useExpertKnowledge, external_EK_Generation, doRuleCompaction, onlyRC, ruleCompactionMethod, doPopulationReboot, popRebootIteration, filterAlgorithm, onlyEKScores, configFileName,classLabel,nu, offlineData,trackingFrequency)

            configBaseName =  outputPath+outputFolder+'/'+permuteFolder+'/'+dataName+'_Permuted_'+str(i)+'_CV_'

            if not fileCheck:
                submitPermuteJob(configBaseName,outputFolder,permuteFolder,version,checkIteration,numPartitions,dataName,i,hours,analysisFolder)
                jobCounter += 1
                
            if fileCheck:
                resubmit = False
                for part in range(numPartitions):
                    outputName = outputPath + outputFolder+'/'+permuteFolder+'/'+dataName+'_Permuted_'+str(i)+'_CV_'+str(part)
                    outName = outputName+'_Train_ExSTraCS_'+checkIteration+'_PopStats.txt'
                    if not os.path.exists(outName):
                        print(outName)
                        missingJobs += 1
                        resubmit = True
                if resubmitJobs and resubmit:
                    submitPermuteJob(configBaseName,outputFolder,permuteFolder,version,checkIteration,numPartitions,dataName,i,hours,analysisFolder)
                    jobCounter += 1
                
        if not fileCheck or resubmitJobs:
            print(str(jobCounter) + ' jobs have been submitted.')        
        if fileCheck:
            print(str(missingJobs)+' Missing Files.') 
          
            
def submitSingleJob(version,dataName,hours,configFileName): 
    """ Submit Job to the cluster. """
    #MAKE CLUSTER JOBS###################################################################
    jobName = scratchPath+dataName+'_run.pbs'                                                  
    pbsFile = open(jobName, 'w')
    pbsFile.write('#!/bin/bash -l\n') #NEW
    pbsFile.write('#PBS -A Moore\n') #NEW
    pbsFile.write('#PBS -q largeq\n')
    pbsFile.write('#PBS -N '+dataName+'\n')
    pbsFile.write('#PBS -l walltime='+str(int(hours))+':00:00\n')
    pbsFile.write('#PBS -l nodes=1:ppn=1\n')
    pbsFile.write('#PBS -M Ryan.J.Urbanowicz\@dartmouth.edu\n\n')
    pbsFile.write('#PBS -o localhost:/idata/cgl/ryanu/logs\n\n')
    pbsFile.write('#PBS -e localhost:/idata/cgl/ryanu/logs\n\n')
    pbsFile.write('time python '+myHome+'/exstracs/'+str(version)+'/'+'ExSTraCS_Main.py '+str(configFileName)+'\n')
    pbsFile.close()
    os.system('qsub '+jobName)    
    os.unlink(jobName)  #deletes the job submission file after it is submitted. 
    #####################################################################################  
                
                
def submitJob(version,dataName,part,hours,configFileName): 
    """ Submit Job to the cluster. """
    #MAKE CLUSTER JOBS###################################################################
    jobName = scratchPath+dataName+'_'+str(part)+'_run.pbs'                                                  
    pbsFile = open(jobName, 'w')
    pbsFile.write('#!/bin/bash -l\n') #NEW
    pbsFile.write('#PBS -A Moore\n') #NEW
    pbsFile.write('#PBS -q largeq\n')
    pbsFile.write('#PBS -N '+dataName+'_'+str(part)+'\n')
    pbsFile.write('#PBS -l walltime='+str(int(hours))+':00:00\n')
    pbsFile.write('#PBS -l nodes=1:ppn=1\n')
    pbsFile.write('#PBS -M Ryan.J.Urbanowicz\@dartmouth.edu\n\n')
    pbsFile.write('#PBS -o localhost:/idata/cgl/ryanu/logs\n\n')
    pbsFile.write('#PBS -e localhost:/idata/cgl/ryanu/logs\n\n')
    pbsFile.write('time python '+myHome+'/exstracs/'+str(version)+'/'+'ExSTraCS_Main.py '+str(configFileName)+'\n')
    pbsFile.close()
    os.system('qsub '+jobName)    
    os.unlink(jobName)  #deletes the job submission file after it is submitted. 
    #####################################################################################         
    
    
def submitPermuteJob(configBaseName,outputFolder,permuteFolder,version,checkIteration,numPartitions,dataName,i,hours,analysisFolder): 
    """ Submit Job to the cluster. """
    #MAKE CLUSTER JOBS###################################################################
    jobName = scratchPath+dataName+'_'+str(i)+'_run.pbs'                                                  
    pbsFile = open(jobName, 'w')
    pbsFile.write('#!/bin/bash -l\n') #NEW
    pbsFile.write('#PBS -A Moore\n') #NEW
    pbsFile.write('#PBS -q largeq\n')
    pbsFile.write('#PBS -N '+dataName+'_'+str(i)+'\n')
    pbsFile.write('#PBS -l walltime='+str(int(hours*10))+':00:00\n')
    pbsFile.write('#PBS -l nodes=1:ppn=1\n')
    pbsFile.write('#PBS -M Ryan.J.Urbanowicz\@dartmouth.edu\n\n')
    pbsFile.write('#PBS -o localhost:/idata/cgl/ryanu/logs\n\n')
    pbsFile.write('#PBS -e localhost:/idata/cgl/ryanu/logs\n\n')
    pbsFile.write('time python '+myHome+'/exstracs/'+str(version)+'/'+analysisFolder+'/'+'run_ExSTraCS_Real_Sub.py '+str(configBaseName)+' '+str(outputFolder)+' '+str(permuteFolder)+' '+str(version)+' '+str(checkIteration)+' '+str(numPartitions)+' '+str(dataName)+' '+str(i)+'\n')
    pbsFile.close()
    os.system('qsub '+jobName)    
    os.unlink(jobName)  #deletes the job submission file after it is submitted. 
    #####################################################################################  
    
    
def LoadDataset(filename):
    global headerList
    global datasetList
    datasetFile = open(filename, 'r')
    headerList = datasetFile.readline().rstrip('\n').split('\t')
    print(headerList)
    for i in range(len(headerList)):
        datasetList.append([])
        
    for line in datasetFile:
        lineList = line.strip('\n').split('\t')
        for i in range(len(headerList)):
            datasetList[i].append(lineList[i])
            
    datasetFile.close()
                
                
def WritePermutedDataset(filename,classLabel):
    outputFile = open(filename, 'w')
    outputFile.write('\t'.join(headerList) + '\n')
    permDataset = ReturnPermutedDataset(classLabel)
    for i in range(len(datasetList[0])):
        currList = []
        for j in range(len(headerList)):
            currList.append(permDataset[j][i])
            
        outputFile.write('\t'.join(currList) + '\n')
        
    outputFile.close()
        
        
def ReturnPermutedDataset(classLabel):
    permDataset = datasetList
    #print headerList
    #print classLabel
    permColumn = headerList.index(classLabel)
    ### Shuffle the column to permute.
    #from random import shuffle
    shuffle(permDataset[permColumn])
    return permDataset    


def makeConfigFile(trainDataName, testDataName, outputName, outEKFileName, randomSeed, learningIterations, popSize, doAttributeTracking, doAttributeFeedback, adjustedAT, useExpertKnowledge, external_EK_Generation, doRuleCompaction, onlyRC, ruleCompactionMethod, doPopulationReboot, popRebootIteration, filterAlgorithm, onlyEKScores, configFileName,classLabel,nu,offlineData,trackingFrequency):
    """ Construct Configuration File for CV Analysis """
    configFile = open(configFileName,'w')
    
    #Write to Config File#############################################################################################################
    configFile.write('offlineData='+str(offlineData)+'# \n')
    configFile.write('trainFile='+str(trainDataName)+'# \n')
    configFile.write('testFile='+str(testDataName)+'# \n')
    configFile.write('internalCrossValidation=0# \n')
    
    configFile.write('outFileName='+str(outputName)+'# \n')
    configFile.write('randomSeed='+str(randomSeed)+'# \n')
    configFile.write('labelInstanceID=InstanceID# \n')
    configFile.write('labelPhenotype='+str(classLabel)+'# \n')
    configFile.write('discreteAttributeLimit=10# \n')
    configFile.write('labelMissingData=NA# \n')
    configFile.write('outputSummary=1# \n')
    configFile.write('outputPopulation=1# \n')
    configFile.write('outputAttCoOccur=1# \n')
    configFile.write('outputTestPredictions=1# \n')  
        
    configFile.write('trackingFrequency='+str(trackingFrequency)+'# \n')
    configFile.write('learningIterations='+str(learningIterations)+'# \n')
    
    configFile.write('N='+str(popSize)+'# \n') 
    configFile.write('nu='+str(nu)+'# \n')# 1
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

if __name__=="__main__":
    import sys
    import os
    import time
    import CVPartitioner
    from random import shuffle
    
    datasetList = []
    headerList = []
    classheader = 'Class'
    fileCheck = False
    resubmitJobs = False
    
    outputFolder = sys.argv[1] 
    if len(sys.argv) > 2:
        print("Running Real Analysis Permutation Completion Check.")
        fileCheck = True
    if len(sys.argv) > 3:
        print("Resubmitting Real Analysis Missing Jobs.")
        resubmitJobs = True #During Filecheck, resubmits jobs if they are missing.
    
    myHome = os.environ.get('HOME')
    outputPath = '/idata/cgl/ryanu/output/'
    dataPath = '/idata/cgl/ryanu/datasets/'   
    scratchPath = '/global/scratch/ryanu/'      
    main(outputFolder, fileCheck, resubmitJobs) 
