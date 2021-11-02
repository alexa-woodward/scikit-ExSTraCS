"""
Name: run_ExSTraCS_Real_Sub.py
Author: Ryan Urbanowicz
Date: 10/10/14
Description: 
"""

def main(configBaseName,outputFolder,permuteFolder, version, restartFrom, numPartitions, dataName, i):

    for part in range(int(numPartitions)):
        if not os.path.exists(outputPath + outputFolder+'/'+permuteFolder+'/'+dataName+'_Permuted_'+str(i)+'_CV_'+str(part)+'_ExSTraCS_'+str(restartFrom)+'_PopStats.txt'):   
            configFileName = configBaseName+str(part)+'_ExSTraCS_ConfigFile.txt'                                                                             
            os.system('time python '+myHome+'/exstracs/'+str(version)+'/ExSTraCS_Main.py '+str(configFileName)+'\n')
        else:
            print(outputPath + outputFolder+'/'+permuteFolder+'/'+dataName+'_Permuted_'+str(i)+'_CV_'+str(part)+'_ExSTraCS_'+str(restartFrom)+'_PopStats.txt  EXISTS ALREADY.')  
    

if __name__=="__main__":
    import sys
    import os
    import time
    import random
    import CVPartitioner
    
    configBaseName = sys.argv[1]
    outputFolder = sys.argv[2]
    permuteFolder = sys.argv[3]
    version = sys.argv[4]
    restartFrom = sys.argv[5]
    numPartitions = sys.argv[6]
    dataName = sys.argv[7]
    i = sys.argv[8]
    
    myHome = os.environ.get('HOME')
    outputPath = '/idata/cgl/ryanu/output/'
    dataPath = '/idata/cgl/ryanu/datasets'  
  
    main(configBaseName,outputFolder,permuteFolder, version, restartFrom, numPartitions, dataName, i)
