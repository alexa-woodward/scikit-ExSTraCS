"""
Name: run_ExSTraCS_Multiplexer_Testing.py
Author: Ryan Urbanowicz
Date: 11/14/14
Description: Organizes and submits a real dataset analysis using ExSTraCS to the Discovery cluster.
"""


def main(outputFolder, fileCheck, resubmitJobs):
    #ppn= 2 for 70 multiplexer and ppn=3 for 135 multiplexer.
    #Multiplexer Data Simulation---------------------------------
    multiplexer = 135 #6,11,20,37,70,135
    instances = [10000,20000,40000] #[500,1000,2000,5000,10000,10000,20000,40000]
    replications = 30
    hours = 10 #Walltime limit (in hours)
    learningCheckpoints = ['10000','50000','100000','200000','500000','1000000','1500000','RC_QRF_1500000']#['10000','50000','100000','200000','500000','RC_QRF_500000']#['10000','50000','100000','200000','RC_QRF_200000']#[10000,50000,100000,200000,500000,1000000,1500000]
    
    doPopulationReboot = 1 #1 is True, 0 is False 
    #popRebootIteration = '10000'
    
    version = 'F2.2_Paper_Testing'
    trackingFrequency = 1000
    learningIterations = '10000.50000.100000.200000.500000.1000000.1500000'#'10000.50000.100000.200000' # 
    popSize = 10000 #2000 #
    
    projectFolder = 'Multiplexer' 
    dataFolder = 'OriginalData'
    nu = 10
    randomSeed = 'False' #Use a fixed random seed? If no put (False), if yes, set the random seed value as an integer.  
    
    #Data Parameters####################################################################################################
    onlyEKScores = 0 # 1=True, 0 = False
    onlyRC = 0 #1 is True, 0 is False
    #ExSTraCS Run Parameters############################################################################################
    offlineData = 1 #Switch on/off multiplexer
    classLabel = 'Class'
    numPartitions = 10

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

    checkIteration = '10000'

    testLocator = [] #This will shuffle the random multiplexer datasets that are used for testing vs. training.  Basically any of the other replicate training samples could be used.
    for i in range(1,replications):
        testLocator.append(i)
    testLocator.append(0)
    
    jobCounter = 0
    missingJobs = 0
    for w in learningCheckpoints:
        for i in instances:
            for j in range(replications):
                
                outputName = outputPath + outputFolder+'/'+dataFolder+'/'
                shortName = str(multiplexer)+"Multiplexer_Data_"+str(i)+"_"+str(j)
                if not os.path.exists(outputName+shortName+'_ExSTraCS_'+str(w)+'_PopStats_Testing.txt'):
                    dataName = dataPath+projectFolder+'/'+dataFolder+'/'+str(multiplexer)+"Multiplexer_Data_"+str(i)+"_"+str(j)+".txt"
                    testDataName = dataPath+projectFolder+'/'+dataFolder+'/'+str(multiplexer)+"Multiplexer_Data_"+str(i)+"_"+str(testLocator[j])+".txt"
                    #Output File Preparation------------------------------------------
                    if not os.path.exists(outputPath + outputFolder):
                        os.mkdir(outputPath + outputFolder)  #top of heirarchy
                    if not os.path.exists(outputPath + outputFolder+'/'+dataFolder):
                        os.mkdir(outputPath + outputFolder+'/'+dataFolder)  #top of heirarchy
                        
                    trainDataName = dataName
                    testDataName = testDataName


                    shorterName = str(multiplexer)+"Multiplexer_Data_"+str(i)
                    
                    if useExpertKnowledge or onlyEKScores:
                        if not os.path.exists(outputPath+filterAlgorithm+'_'+shorterName):
                            os.mkdir(outputPath+filterAlgorithm+'_'+shorterName)  #top of heirarchy
                        if internalEK or onlyEKScores:
                            external_EK_Generation = 'None'
                            outEKFileName = outputPath+filterAlgorithm+'_'+shorterName+'/'
                        else: 
                            external_EK_Generation = externalEKManualPath
                            outEKFileName = ''
                    if not fileCheck:
                        #Generate Configuration File ###############
                        configFileName = outputPath+outputFolder+'/'+dataFolder+'/'+shortName+'_ExSTraCS_ConfigFile.txt'
                        makeConfigFile(trainDataName, testDataName, outputName, outEKFileName, randomSeed, learningIterations, popSize, doAttributeTracking, doAttributeFeedback, adjustedAT, useExpertKnowledge, external_EK_Generation, doRuleCompaction, onlyRC, ruleCompactionMethod, doPopulationReboot, w, filterAlgorithm, onlyEKScores, configFileName,classLabel,nu,offlineData,trackingFrequency)
                        ############################################                 
                        submitSingleJob(version,shortName,hours,configFileName)
                        jobCounter += 1
            
                    if fileCheck:
                        missing = 0
                        if not os.path.exists(outputName+shortName+'_ExSTraCS_'+str(checkIteration)+'_PopStats.txt'):
                            print(outputName+shortName+'_ExSTraCS_'+str(checkIteration)+'_PopStats.txt')
                            missing += 1
                            if resubmitJobs:
                                submitSingleJob(version,shortName,hours,configFileName)
                        print(str(missing)+' Missing File.')  
              
                        
    if not fileCheck or resubmitJobs:
        print(str(jobCounter) + ' jobs have been submitted.')            
   
            
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
    pbsFile.write('#PBS -l nodes=1:ppn=3\n')
    #pbsFile.write('#PBS -l nodes=1:ppn=1\n')
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
    pbsFile.write('#PBS -l nodes=1:ppn=2\n')
    pbsFile.write('#PBS -M Ryan.J.Urbanowicz\@dartmouth.edu\n\n')
    pbsFile.write('#PBS -o localhost:/idata/cgl/ryanu/logs\n\n')
    pbsFile.write('#PBS -e localhost:/idata/cgl/ryanu/logs\n\n')
    pbsFile.write('time python '+myHome+'/exstracs/'+str(version)+'/'+'ExSTraCS_Main.py '+str(configFileName)+'\n')
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
    configFile.write('onlyTest=1# \n')  
      
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
    import Problem_Multiplexer
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
