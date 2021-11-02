"""
Name: run_ExSTraCS_Real_Summary_Main.py
Author: Ryan Urbanowicz
Date: 8/28/14
Description: 
"""

def main():
    analysisFolder = 'GlaucomaAnalysis'
    outputFolder = 'GlaucTest'
    dataName = 'glaucoma_imputed_19snp_cglformat' 
    
    numTrials = ['5000','10000','RC_QRF_10000'] #['10000','50000','100000','200000''RC_QRF_200000']  #,'400000','RC_QRF_400000']
    version = 'F2_Develop'
    
    permutationTesting = 0  #0 for False, and 1 for True
    doAttTrack = 1 #0 for False, and 1 for True
    doCOTrack = 1 #0 for False, and 1 for True
    doCOGephi = 1 #0 for False, and 1 for True
    instanceLabel = 'InstanceID'

    CVPartitions = 10
    permutations = 1000
    
    jobCounter = 0
    for num in numTrials:
        if not os.path.exists(outputPath+outputFolder+'/'+outputFolder+'_'+num+'_Summary.txt'):   
            jobName = scratchPath+num+'_run.pbs'                            
            pbsFile = open(jobName, 'w')
            #pbsFile.write('#PBS -q testq\n')
            pbsFile.write('#PBS -q default\n')
            pbsFile.write('#PBS -N '+num+'\n')
            #pbsFile.write('#PBS -l walltime=00:15:00\n') #pbsFile.write('#PBS -l walltime=00:15:00\n')
            pbsFile.write('#PBS -l walltime=02:00:00\n') #pbsFile.write('#PBS -l walltime=00:15:00\n')
            pbsFile.write('#PBS -l nodes=1:ppn=1\n')
            pbsFile.write('#PBS -M Ryan.J.Urbanowicz\@dartmouth.edu\n\n')
            pbsFile.write('#PBS -o localhost:/idata/cgl/ryanu/logs\n\n')
            pbsFile.write('#PBS -e localhost:/idata/cgl/ryanu/logs\n\n')
            pbsFile.write('time python '+myHome+'/exstracs/'+str(version)+'/'+str(analysisFolder)+'/sum_ExSTraCS_Real_Sub.py '+outputFolder+ ' '+str(num)+ ' '+str(dataName)+' '+str(version)+' '+str(permutations)+ ' '+str(CVPartitions)+' '+str(permutationTesting)+' '+str(doAttTrack)+' '+str(doCOTrack)+' '+str(instanceLabel)+' '+str(doCOGephi)+'\n')
            pbsFile.close()
            os.system('qsub '+jobName)    
            os.unlink(jobName)  #deletes the job submission file after it is submitted.
            jobCounter +=1  
        else:
            print(outputPath+outputFolder+'/'+outputFolder+'_'+num+'_Summary.txt' + " EXISTS ALREADY.")    
    print(str(jobCounter) + ' jobs have been submitted.')
            
            
if __name__=="__main__":
    import sys
    import os
    import time

    outputPath = '/idata/cgl/ryanu/output/' 
    myHome = os.environ.get('HOME')
    scratchPath = '/global/scratch/ryanu/'      
    
    main()