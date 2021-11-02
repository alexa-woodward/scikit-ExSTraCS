"""
Name: run_ExSTraCS_Real_Main.py
Author: Ryan Urbanowicz
Date: 10/10/14
Description: Organizes and submits a real dataset analysis using ExSTraCS to the Discovery cluster.
"""


def main():
    outputFolder = 'ExSTraCS_V2.2_135Multiplexer_10000Pop_FINAL_NEW'#'ExSTraCS_V2.2_6Multiplexer_500Pop_FINAL'
    #open all stats files and get testing accuracy

    #Multiplexer Data Simulation---------------------------------
    multiplexer = 135 #11,20,37,70,135
    instances = [10000,20000,40000] #[500,1000,2000,5000,10000,10000,20000,40000]
    replications = 30
    trackingFrequency = 1000
    learningCheckpoints = ['10000','50000','100000','200000','500000','1000000','1500000','RC_QRF_1500000']#['10000','50000','100000','200000','RC_QRF_2000000']#[10000,50000,100000,200000,500000,1000000,1500000]
    

    dataFolder = 'OriginalData'

    for i in instances:
        print(i)
        trackArray = [0]*len(learningCheckpoints)
        local = 0
        summaryFile = open(outputPath+outputFolder+'/'+str(multiplexer)+"Multiplexer_Data_"+str(i)+'_TestingAcc_Summary.txt', 'w')  
        summaryFile.write('Iteration\tAveTestingAccuracy\tReplications\n')   
        for w in learningCheckpoints:
            counter = 0

            for j in range(replications):
                readFile = outputPath+outputFolder+'/'+dataFolder+'/'+str(multiplexer)+"Multiplexer_Data_"+str(i)+"_"+str(j)+"_ExSTraCS_"+str(w)+"_PopStats_Testing.txt"
                try:
                    fileObject = open(readFile, 'r')  # opens each datafile to read.
                    counter += 1
                    tempLine = None
                    for v in range(3):
                        tempLine = fileObject.readline()
                    tempList = tempLine.strip().split('\t')
                    trackArray[local] += float(tempList[1])
                    
                    testAcc = float(tempList[1])
                    fileObject.close()
                except:
                    print("Data-set Not Found! Progressing without it.")
                    print(readFile)
            


                
            trackArray[local] = trackArray[local] / float(counter)
            summaryFile.write(str(w)+'\t'+str(trackArray[local])+'\t'+str(counter)+'\n')
            local += 1

         
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
