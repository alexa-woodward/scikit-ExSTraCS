"""
Name: run_ExSTraCS_Real_Summary_Sub.py
Author: Ryan Urbanowicz
Date: 8/28/14
Description: 
"""

def main(outputFolder,num,dataName,version,permutations,CVPartitions,permutationTesting,doAttTrack,doCOTrack,instanceLabel,doCOGephi):
    dataFolder = 'OriginalData'
    permuteFolder = 'PermutedData'

    summaryFile = open(outputPath+outputFolder+'/'+outputFolder+'_'+num+'_Summary.txt', 'w')  
    summaryFile.write('Config.ID\tIteration\tAveTrainAccuracy\tAveTestAccuracy\tAveTrainCover\tAveTestCover\tAveMacroPopSize\tAveMicroPopSize\tAveGenerality\tAveGlobalTime\tValidCVs\n')     

    counter = 10
    AveTrainAccuracy = 0
    AveTestAccuracy = 0
    AveTrainCover = 0
    AveTestCover = 0
    AveMacroPopSize = 0
    AveMicroPopSize = 0
    AveGenerality = 0  
    AveGlobalTime = 0

    AttNameList = []
    AttCountList = []
    AttAccList = []
    AttTrackSum = []
    AveCoList = []
    
    #Real Data Analysis -----------------------------------------------------------------------------------------------
    for part in range(0,CVPartitions):  
        print(part)
        readFile = outputPath + outputFolder+'/'+dataFolder+'/'+str(dataName)+'_CV_'+str(part)+'_Train_ExSTraCS'+'_'+num+'_PopStats.txt'          
        #*********************************************************************        
        try:
            fileObject = open(readFile, 'r')  # opens each datafile to read.

        except:
            print("Data-set Not Found! Progressing without it.")
            print(readFile)
            counter -=1
            continue
        #*********************************************************************  
        tempLine = None
        for v in range(3):
            tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t')

        AveTrainAccuracy += float(tempList[0])
        AveTestAccuracy += float(tempList[1])
        AveTrainCover += float(tempList[2])
        AveTestCover += float(tempList[3])
                        
        for v in range(4):
            tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t')
                     
        AveMacroPopSize += float(tempList[0])
        AveMicroPopSize += float(tempList[1])
        AveGenerality += float(tempList[2])   
                      
        for v in range(3):
            tempLine = fileObject.readline()
        AttNameList = tempLine.strip().split('\t')   
           
        tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t')
        if part == 0:
            AttCountList = tempList
            for j in range(len(AttCountList)): #make values floats
                AttCountList[j] = float(AttCountList[j])
        else:
            for i in range(len(tempList)):
                AttCountList[i] += float(tempList[i])

        for v in range(4):
            tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t')
        if part == 0:
            AttAccList = tempList
            for j in range(len(AttAccList)):
                AttAccList[j] = float(AttAccList[j])
        else:
            for i in range(len(tempList)):
                AttAccList[i] += float(tempList[i])
        
        for v in range(4):
            tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t')
        
        if doAttTrack:
            if part == 0:
                AttTrackSum = tempList
                for j in range(len(AttTrackSum)):
                    AttTrackSum[j] = float(AttTrackSum[j])
            else:
                for i in range(len(tempList)):
                    AttTrackSum[i] += float(tempList[i])
        
        for v in range(3):
            tempLine = fileObject.readline()
        tempList = tempLine.strip().split('\t')  
        AveGlobalTime += float(tempList[1])
                            
        fileObject.close()
        
        if doCOTrack:    
            #Determine Co-occurence Stats
            readFile = outputPath + outputFolder+'/'+dataFolder+'/'+str(dataName)+'_CV_'+str(part)+'_Train_ExSTraCS'+'_'+num+'_CO.txt'           
            #*********************************************************************        
            try:
                fileObject = open(readFile, 'r')  # opens each datafile to read.

            except:
                print("Data-set Not Found! Progressing without it.")
                print(readFile)
                continue
            #*********************************************************************  
            if part == 0:
                for line in fileObject:
                    tempList = line.strip().split('\t')  
                    AveCoList.append(tempList)

            else:
                for line in fileObject:
                    tempList = line.strip().split('\t')  
                    comboNotFound = True
                    comboNum = 0
                    while comboNotFound:
                        if AveCoList[comboNum][0] == tempList[0] and AveCoList[comboNum][1] == tempList[1]:
                            AveCoList[comboNum][2] = float(AveCoList[comboNum][2]) + float(tempList[2])
                            comboNotFound = False
                        comboNum += 1
            fileObject.close()
            
            if doCOGephi:                  
                #make adjacency matrix
                adj_mat = []
                for i in range(len(AttNameList)):
                    lineList = []
                    for j in range(len(AttNameList)):
                        if AttNameList[i] == AttNameList[j]: #same attribute
                            lineList.append(AttCountList[j]) #use SPEC SUM
                            #lineList.append(SNPAccList[j]) #use AWSPEC SUM
                        else:
                            attList = [AttNameList[i],AttNameList[j]]
                            coId = 0
                            notFound = True
                            while notFound:
                                if (AveCoList[coId][0] in attList) and (AveCoList[coId][1] in attList):
                                    lineList.append(AveCoList[coId][2])
                                    notFound = False
                                else:
                                    coId += 1
                    adj_mat.append(lineList)
                    
                f = open(outputPath+outputFolder+'/'+outputFolder+'_'+num+'_Gephi.csv', 'w')
                for n in AttNameList:
                    f.write(';'+n)
                f.write('\n')
                for i in range(len(adj_mat)):
                    f.write(AttNameList[i])
                    for j in range(len(adj_mat[i])):
                        a = adj_mat[i][j]
                        f.write(';')
                        f.write(str(a))
                    f.write('\n')
                f.close()

    #Calculate final Values
    if counter > 0:
        AveTrainAccuracy = AveTrainAccuracy / float(counter)
        AveTestAccuracy = AveTestAccuracy / float(counter)
        AveTrainCover = AveTrainCover / float(counter)
        AveTestCover = AveTestCover / float(counter)
        AveMacroPopSize = AveMacroPopSize / float(counter)
        AveMicroPopSize = AveMicroPopSize / float(counter)
        AveGenerality = AveGenerality / float(counter)
        AveGlobalTime = AveGlobalTime / float(counter)

        for each in (AttCountList):
            each = each / float(counter)
        for each in (AttAccList):
            each = each / float(counter)
        for each in (AttTrackSum):
            each = each / float(counter)
        
    else:
        AveTrainAccuracy = 'NA'
        AveTestAccuracy = 'NA'
        AveTrainNoMatch = 'NA'
        AveTestNoMatch = 'NA'
        AveMacroPopSize = 'NA'
        AveMicroPopSize = 'NA'
        AveGenerality = 'NA'
        AveGlobalTime = 'NA'

    summaryFile.write(str(outputFolder)+'\t'+str(num)+'\t'+str(AveTrainAccuracy)+'\t'+str(AveTestAccuracy)+'\t'+str(AveTrainCover)+'\t'+str(AveTestCover)+'\t'+str(AveMacroPopSize)+'\t'+str(AveMicroPopSize)+'\t'+str(AveGenerality)+'\t'+str(AveGlobalTime)+'\t'+str(counter)+'\n')                                                                                                      


    #Permuted Dataset Analysis------------------------------------------------------------------------------------------------------------------------------------------
    if permutationTesting:
        trainAccuracyList = []
        testAccuracyList = []
        trainCoverList = []
        testCoverList = []
        macroPopSizeList = []
        microPopSizeList = []
        generalityList = []
        globalTimeList = []
        
        PermAttCountList = []
        PermAttAccList = []
        PermAttTrackSum = []
        PermAveCoList = []
    
        for perm in range(0,permutations):
            print(perm)
            counter = 10

            permTrainAccuracy = 0
            permTestAccuracy = 0
            permTrainCover = 0
            permTestCover = 0
            permMacroPop = 0
            permMicroPop = 0
            permGenerality = 0
            permGlobalTime = 0

            tempPermAttCountList = []
            tempPermAttAccList = []
            tempPermAttTrackSum = []
            PermAveCoList.append([])
            
            for part in range(0,CVPartitions):   
                permFile = outputPath + outputFolder+'/'+permuteFolder+'/'+str(dataName)+'_Permuted_'+str(perm)+'_CV_'+str(part)+'_'+'ExSTraCS'+'_'+num+'_PopStats.txt'            
                #*********************************************************************        
                try:
                    fileObject = open(permFile, 'r')  # opens each datafile to read.
    
                except:
                    print("Data-set Not Found! Progressing without it.")
                    print(permFile)
                    counter -=1
                    continue
                #*********************************************************************  
                tempLine = None
                for v in range(3):
                    tempLine = fileObject.readline()
                tempList = tempLine.strip().split('\t')
                permTrainAccuracy += float(tempList[0])
                permTestAccuracy += float(tempList[1])
                permTrainCover += float(tempList[2])
                permTestCover += float(tempList[3])
            
                for v in range(4):
                    tempLine = fileObject.readline()
                tempList = tempLine.strip().split('\t')
                permMacroPop += float(tempList[0])
                permMicroPop += float(tempList[1])
                permGenerality += float(tempList[2])
                
                for v in range(4):
                    tempLine = fileObject.readline()
                tempList = tempLine.strip().split('\t')
                if part == 0:
                    tempPermAttCountList = tempList
                    for j in range(len(tempPermAttCountList)):
                        tempPermAttCountList[j] = float(tempPermAttCountList[j])
                else:
                    for i in range(len(tempList)):
                        tempPermAttCountList[i] += float(tempList[i])
                
                for v in range(4):
                    tempLine = fileObject.readline()
                tempList = tempLine.strip().split('\t')
                if part == 0:
                    tempPermAttAccList = tempList
                    for j in range(len(tempPermAttAccList)):
                        tempPermAttAccList[j] = float(tempPermAttAccList[j])
                else:
                    for i in range(len(tempList)):
                        tempPermAttAccList[i] += float(tempList[i])
                        
                for v in range(4):
                    tempLine = fileObject.readline()
                tempList = tempLine.strip().split('\t')
                if doAttTrack:
                    if part == 0:
                        tempPermAttTrackSum = tempList
                        for j in range(len(tempPermAttTrackSum)):
                            tempPermAttTrackSum[j] = float(tempPermAttTrackSum[j])
                    else:
                        for i in range(len(tempList)):
                            tempPermAttTrackSum[i] += float(tempList[i])          
                        
                for v in range(3):
                    tempLine = fileObject.readline()
                tempList = tempLine.strip().split('\t')
                permGlobalTime += float(tempList[0]) 
                        
                fileObject.close() 

                if doCOTrack:    
                    #Determine Co-occurence Stats
                    readFile = outputPath + outputFolder+'/'+dataFolder+'/'+str(dataName)+'_Permuted_'+str(perm)+'_CV_'+str(part)+'_'+'ExSTraCS'+'_'+num+'_CO.txt'           
                    #*********************************************************************        
                    try:
                        fileObject = open(readFile, 'r')  # opens each datafile to read.
        
                    except:
                        print("Data-set Not Found! Progressing without it.")
                        print(readFile)
                        continue
                    #*********************************************************************  
                    if part == 0:
                        for line in fileObject:
                            tempList = line.strip().split('\t')  
                            PermAveCoList[perm].append(tempList)
        
                    else:
                        for line in fileObject:
                            tempList = line.strip().split('\t')  
                            comboNotFound = True
                            comboNum = 0
                            while comboNotFound:
                                if PermAveCoList[perm][comboNum][0] == tempList[0] and PermAveCoList[perm][comboNum][1] == tempList[1]:
                                    PermAveCoList[perm][comboNum][2] = float(PermAveCoList[perm][comboNum][2]) + float(tempList[2])
                                    comboNotFound = False
                                comboNum += 1
                    fileObject.close()
            
            if counter > 0:
                trainAccuracyList.append(permTrainAccuracy/ float(counter))
                testAccuracyList.append(permTestAccuracy/ float(counter))
                trainCoverList.append(permTrainCover/ float(counter))
                testCoverList.append(permTestCover/ float(counter))
                macroPopSizeList.append(permMacroPop/ float(counter))
                microPopSizeList.append(permMicroPop/ float(counter))
                generalityList.append(permGenerality/ float(counter))
                globalTimeList.append(permGlobalTime/ float(counter))
                
                for each in (tempPermAttCountList):
                    each = each / float(counter)
                for each in (tempPermAttAccList):
                    each = each / float(counter)
                for each in (tempPermAttTrackSum):
                    each = each / float(counter)

            else:
                trainAccuracyList.append('NA')
                testAccuracyList.append('NA')
                trainCoverList.append('NA')
                testCoverList.append('NA')
                macroPopSizeList.append('NA')
                microPopSizeList.append('NA')
                generalityList.append('NA')
                globalTimeList.append('NA')
            
            PermAttCountList.append(tempPermAttCountList)
            PermAttAccList.append(tempPermAttAccList)
            PermAttTrackSum.append(tempPermAttTrackSum)
 
        #Determine P-Values-----------------------------------------------------------------------------------------------
        checkAllPValues = True
        symbolList = []
        
        summaryFile.write('\t'+'P-Values:')
        
        trainAccuracyList = sorted(trainAccuracyList)
        pvalResult = PValCheck(AveTrainAccuracy,trainAccuracyList,permutations)
        summaryFile.write('\t'+str(pvalResult[0]))
        symbolList.append(pvalResult[1])  #0 = TrainAcc
                          
        testAccuracyList = sorted(testAccuracyList)
        pvalResult = PValCheck(AveTestAccuracy,testAccuracyList,permutations)
        summaryFile.write('\t'+str(pvalResult[0]))
        symbolList.append(pvalResult[1])  #1 = TestAcc
        if pvalResult > 0.05:  #Don't waste time evaluating attribute significance values if testing accuracy difference is not significantly bigger.
            checkAllPValues = False
        
        trainCoverList = sorted(trainCoverList)
        pvalResult = PValCheck(AveTrainCover,trainCoverList,permutations)
        summaryFile.write('\t'+str(pvalResult[0]))
        symbolList.append(pvalResult[1])  #2 = TrainCover
        
        testCoverList = sorted(testCoverList)
        pvalResult = PValCheck(AveTestCover,testCoverList,permutations)
        summaryFile.write('\t'+str(pvalResult[0]))
        symbolList.append(pvalResult[1])  #3 = TestCover
        
        macroPopSizeList = sorted(macroPopSizeList)
        pvalResult = PValCheck(AveMacroPopSize,macroPopSizeList,permutations)
        summaryFile.write('\t'+str(pvalResult[0]))
        symbolList.append(pvalResult[1])  #4 = MacroPop
        
        microPopSizeList = sorted(microPopSizeList)
        pvalResult = PValCheck(AveMicroPopSize,microPopSizeList,permutations)
        summaryFile.write('\t'+str(pvalResult[0]))
        symbolList.append(pvalResult[1])  #5 = MicroPop
        
        generalityList = sorted(generalityList)
        pvalResult = PValCheck(AveGenerality,generalityList,permutations)
        summaryFile.write('\t'+str(pvalResult[0]))
        symbolList.append(pvalResult[1])  #6 = Generality
                                     
        globalTimeList = sorted(globalTimeList)
        pvalResult = PValCheck(AveGlobalTime,globalTimeList,permutations)
        summaryFile.write('\t'+str(pvalResult[0]))
        symbolList.append(pvalResult[1])  #7 = GlobalTime
                                                                                                  
        summaryFile.write('\n'+'\t'+'Quick:')
        for each in symbolList:
            if each[0] < 0.05:
                if each[1]: #Bigger?
                    summaryFile.write('\t'+'Up_'+str(each[0]))
                else: #Smaller
                    summaryFile.write('\t'+'Down_'+str(each[0]))
        
        summaryFile.write('\n'+'\n'+'SpecificitySum:\n')
        for i in range(len(AttNameList)):
            if i < len(AttNameList)-1:
                summaryFile.write(str(AttNameList[i])+'\t')
            else:
                summaryFile.write(str(AttNameList[i])+'\n')

        for i in range(len(AttCountList)):
            if i < len(AttCountList)-1:
                summaryFile.write(str(AttCountList[i])+'\t')
            else:
                summaryFile.write(str(AttCountList[i])+'\n')
        #STATS for Specificity Sum ---------------------------------------------------------- 
        symbolList = []
        for i in range(len(AttNameList)):
            newAttList = []
            for x in range(len(PermAttCountList)): #for each permutation
                newAttList.append(PermAttCountList[x][i]) #Grabs att value across all permuations
                newAttList = sorted(newAttList)
                pvalResult = PValCheck(AttCountList[i],newAttList,permutations)
                symbolList.append(pvalResult[1])  #7 = GlobalTime
            if i < len(AttNameList)-1:
                summaryFile.write(str(pvalResult[0])+'\t')
            else: 
                summaryFile.write(str(pvalResult[0])+'\n')
        for x in range(len(symbolList)):
            if symbolList[x][0] < 0.05:
                if x < len(symbolList)-1:
                    if each[1]: #Bigger?
                        summaryFile.write('* Up'+'\t')
                    else: #Smaller
                        summaryFile.write('* Down'+'\t')
                else:
                    if each[1]: #Bigger?
                        summaryFile.write('* Up'+'\n')
                    else: #Smaller
                        summaryFile.write('* Down'+'\n')
        
        summaryFile.write('\n'+'\n'+'AccuracySum:\n')
        for i in range(len(AttNameList)):
            if i < len(AttNameList)-1:
                summaryFile.write(str(AttNameList[i])+'\t')
            else:
                summaryFile.write(str(AttNameList[i])+'\n')

        for i in range(len(AttAccList)):
            if i < len(AttAccList)-1:
                summaryFile.write(str(AttAccList[i])+'\t')
            else:
                summaryFile.write(str(AttAccList[i])+'\n')
        #STATS for Accuracy Sum ---------------------------------------------------------- 
        symbolList = []
        for i in range(len(AttNameList)):
            newAttList = []
            for x in range(len(PermAttAccList)): #for each permutation
                newAttList.append(PermAttAccList[x][i]) #Grabs att value across all permuations
                newAttList = sorted(newAttList)
                pvalResult = PValCheck(AttAccList[i],newAttList,permutations)
                symbolList.append(pvalResult[1])  #7 = GlobalTime
            if i < len(AttNameList)-1:
                summaryFile.write(str(pvalResult[0])+'\t')
            else: 
                summaryFile.write(str(pvalResult[0])+'\n')
        for x in range(len(symbolList)):
            if symbolList[x][0] < 0.05:
                if x < len(symbolList)-1:
                    if each[1]: #Bigger?
                        summaryFile.write('* Up'+'\t')
                    else: #Smaller
                        summaryFile.write('* Down'+'\t')
                else:
                    if each[1]: #Bigger?
                        summaryFile.write('* Up'+'\n')
                    else: #Smaller
                        summaryFile.write('* Down'+'\n')

        summaryFile.write('\n'+'\n'+'AttributeTrackingGlobalSum:\n')
        for i in range(len(AttNameList)):
            if i < len(AttNameList)-1:
                summaryFile.write(str(AttNameList[i])+'\t')
            else:
                summaryFile.write(str(AttNameList[i])+'\n')

        for i in range(len(AttTrackSum)):
            if i < len(AttTrackSum)-1:
                summaryFile.write(str(AttTrackSum[i])+'\t')
            else:
                summaryFile.write(str(AttTrackSum[i])+'\n')
        #STATS for Accuracy Sum ---------------------------------------------------------- 
        symbolList = []
        for i in range(len(AttNameList)):
            newAttList = []
            for x in range(len(PermAttTrackSum)): #for each permutation
                newAttList.append(PermAttTrackSum[x][i]) #Grabs att value across all permuations
                newAttList = sorted(newAttList)
                pvalResult = PValCheck(AttTrackSum[i],newAttList,permutations)
                symbolList.append(pvalResult[1])  #7 = GlobalTime
            if i < len(AttNameList)-1:
                summaryFile.write(str(pvalResult[0])+'\t')
            else: 
                summaryFile.write(str(pvalResult[0])+'\n')
        for x in range(len(symbolList)):
            if symbolList[x][0] < 0.05:
                if x < len(symbolList)-1:
                    if each[1]: #Bigger?
                        summaryFile.write(+'* Up'+'\t')
                    else: #Smaller
                        summaryFile.write('* Down'+'\t')
                else:
                    if each[1]: #Bigger?
                        summaryFile.write(+'* Up'+'\n')
                    else: #Smaller
                        summaryFile.write('* Down'+'\n')

        if doCOTrack:
            #permutation Testing on CO values
            SigCovals = []
            sigCount = 0
            for v in range(len(PermAveCoList[0])): #all CO's
                count = 0
                tmpList = []
                endNotHere = True
                for w in range(permutations):
                    tmpList.append(PermAveCoList[w][v][2])
    
                newsortedCO = sorted(tmpList)
                #make list which we can sort
                while endNotHere and AveCoList[v][2] > newsortedCO[count]:
                    count += 1
                    if count == permutations:
                        endNotHere = False
                Pval = None
                if count == permutations:
                    Pval = 1/float(permutations)
                else:
                    Pval = (1-(count/float(permutations)))  
                if Pval < 0.05:
                    SigCovals.append(AveCoList[v])
                    SigCovals[sigCount].append(Pval)
                    sigCount += 1

            tupleList = []
            for i in SigCovals:
                tupleList.append((i[0],i[1],i[2],i[3]))
                    
            sortedComboList = sorted(tupleList,key=lambda test: test[2], reverse=True)

            summaryFile.write('\n'+'COSigResults:\n') 
            for i in range(len(sortedComboList)):
                for j in range(len(sortedComboList[i])):
                    if j < len(sortedComboList[i])-1:
                        summaryFile.write(str(sortedComboList[i][j])+'\t')
                    else:
                        summaryFile.write(str(sortedComboList[i][j])+'\n')
                        
    else:  #No Permutation Testing
        summaryFile.write('\n'+'\n'+'SpecificitySum:\n')
        for i in range(len(AttNameList)):
            if i < len(AttNameList)-1:
                summaryFile.write(str(AttNameList[i])+'\t')
            else:
                summaryFile.write(str(AttNameList[i])+'\n')
    
        for i in range(len(AttCountList)):
            if i < len(AttCountList)-1:
                summaryFile.write(str(AttCountList[i])+'\t')
            else:
                summaryFile.write(str(AttCountList[i])+'\n')
        
        summaryFile.write('\n'+'AccuracySum:\n')
        for i in range(len(AttNameList)):
            if i < len(AttNameList)-1:
                summaryFile.write(str(AttNameList[i])+'\t')
            else:
                summaryFile.write(str(AttNameList[i])+'\n')

        for i in range(len(AttAccList)):
            if i < len(AttAccList)-1:
                summaryFile.write(str(AttAccList[i])+'\t')
            else:
                summaryFile.write(str(AttAccList[i])+'\n')
                
        summaryFile.write('\n'+'AttributeTrackingGlobalSum:\n')
        for i in range(len(AttNameList)):
            if i < len(AttNameList)-1:
                summaryFile.write(str(AttNameList[i])+'\t')
            else:
                summaryFile.write(str(AttNameList[i])+'\n')

        for i in range(len(AttTrackSum)):
            if i < len(AttTrackSum)-1:
                summaryFile.write(str(AttTrackSum[i])+'\t')
            else:
                summaryFile.write(str(AttTrackSum[i])+'\n')   
        
    summaryFile.close()

        
def PValCheck(real,permuteList,permutations):
    count = 0
    bigger = True
    permuteAve = sum(permuteList)/float(len(permuteList)) #Determine direction of change
    if permuteAve < real: #Check for Bigger
        for i in range(permutations):
            if real > permuteList[i]:
                count += 1
    else: #Check for Smaller
        bigger = False
        for i in range(permutations):
            if real < permuteList[i]:
                count += 1
        
    if count == permutations:
        pVal = 1/float(permutations)
    else:
        pVal = 1.0 - (count / float(permutations))
    return [pVal,bigger]


if __name__=="__main__":
    import sys
    import os
    import time
    import copy

    outputFolder = sys.argv[1] 
    num = sys.argv[2]
    dataName = sys.argv[3]
    version = sys.argv[4]
    permutations = int(sys.argv[5])
    CVPartitions = int(sys.argv[6])
    permutationTesting = bool(int(sys.argv[7]))
    doAttTrack = bool(int(sys.argv[8]))
    doCOTrack = bool(int(sys.argv[9]))
    instanceLabel = sys.argv[10]
    doCOGephi = bool(int(sys.argv[11]))
    
    myHome = os.environ.get('HOME')
    outputPath = '/idata/cgl/ryanu/output/'      
    main(outputFolder,num,dataName,version,permutations,CVPartitions,permutationTesting,doAttTrack,doCOTrack,instanceLabel,doCOGephi)   