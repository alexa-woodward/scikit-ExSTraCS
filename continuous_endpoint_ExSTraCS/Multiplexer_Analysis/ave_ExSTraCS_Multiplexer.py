"""
Name: run_ExSTraCS_Real_Main.py
Author: Ryan Urbanowicz
Date: 10/10/14
Description: Organizes and submits a real dataset analysis using ExSTraCS to the Discovery cluster.
"""


def main():
    outputFolder = 'ExSTraCS_V2.2_135Multiplexer_10000Pop_FINAL_NEW'#'ExSTraCS_V2.2_6Multiplexer_500Pop_FINAL'
    #Full analysis 1.) OrigData = True, 2.) OrigData = False & permutedData = False, 3.) OrigData = False & permutedData = True
    #Multiplexer Data Simulation---------------------------------
    multiplexer = 135 #11,20,37,70,135
    instances = [10000,20000,40000] #[10000,20000,40000]
    replications = 30
    trackingFrequency = 1000
    iterations = 1500000
    datapoints = int(iterations / trackingFrequency)
    dataFolder = 'OriginalData'
    
    for i in instances:
        print(i)
        trackArray = [0]*datapoints
        counter = 0
        for j in range(replications):
            readFile = outputPath+outputFolder+'/'+dataFolder+'/'+str(multiplexer)+"Multiplexer_Data_"+str(i)+"_"+str(j)+"_ExSTraCS_LearnTrack.txt"
            try:
                fileObject = open(readFile, 'r')  # opens each datafile to read.
                counter += 1
                tempLine = fileObject.readline()
                local = 0
                for line in fileObject:
                    tempList = line.strip().split('\t')
                    trackArray[local] += float(tempList[3])
                    local += 1
                fileObject.close()
            except:
                print("Data-set Not Found! Progressing without it.")
                print(readFile)
        


        for k in range(len(trackArray)):
            trackArray[k] = trackArray[k] / float(counter)

        summaryFile = open(outputPath+outputFolder+'/'+str(multiplexer)+"Multiplexer_Data_"+str(i)+'_Summary.txt', 'w')  
        summaryFile.write('Iteration\tAveTrainAccuracy\tReplications\n')   
        summaryFile.write('0\t0.5\t'+str(replications)+'\n')  
        iterationCount = 0
        for each in trackArray:
            iterationCount += trackingFrequency
            summaryFile.write(str(iterationCount)+'\t'+str(each)+'\t'+str(counter)+'\n')
         
        summaryFile.close()      


    ######################################################################################################################## 

if __name__=="__main__":
    import sys
    import os
    import time

    myHome = os.environ.get('HOME')
    outputPath = '/idata/cgl/ryanu/output/'
    dataPath = '/idata/cgl/ryanu/datasets/'   
    scratchPath = '/global/scratch/ryanu/'      
    main() 
