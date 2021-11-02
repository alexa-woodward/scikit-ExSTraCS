import os
from random import shuffle


def LoadDataset(filename):
    global headerList
    global datasetList
    datasetList = []
    datasetFile = open(filename, 'r')
    headerList = datasetFile.readline().rstrip('\n').split('\t')
    #print headerList
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



datasetList = []
headerList = []
dataPath = '/idata/cgl/ryanu/datasets/'
originalFolder = 'GAMETES_2.1__Datasets_2Het_Loc_2_Qnt_2_Pop_100000_CE__CV_Data'
newFolder = 'GAMETES_2.1__Datasets_2Het_Loc_2_Qnt_2_Pop_100000_CE_Permuted__CV_Data'
#outputPath = '/cgl/data/niranjan/output/v2Final/CE_permuted/' #Target folder
#dataPath = '/idata/cgl/ryanu/datasets/GAMETES_2.1__Datasets_2Het_Loc_2_Qnt_2_Pop_100000_CE__CV_Data/'

if not os.path.exists(dataPath+'/'+newFolder):
    os.mkdir(dataPath+'/'+newFolder)  #top of heirarchy

fileList = os.listdir(dataPath+'/'+originalFolder+'/')
for file in fileList:
    print('Permuted ' + file)
    LoadDataset(dataPath+'/'+originalFolder+'/'+file)
    WritePermutedDataset(dataPath+'/'+newFolder+'/'+file.replace('.txt','') + '_Permuted.txt','Class')
    

