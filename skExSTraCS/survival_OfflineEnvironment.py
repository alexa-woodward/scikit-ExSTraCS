from skExSTraCS.survival_DataManagement import DataManagement

class OfflineEnvironment:
    def __init__(self,dataFeatures,dataEventTimes,dataEventStatus,model):
        """Initialize Offline Environment"""
        self.dataRef = 0
        self.formatData = DataManagement(dataFeatures,dataEventTimes,dataEventStatus,model)

        self.currentTrainState = self.formatData.trainFormatted[0][self.dataRef]
        self.currentTrainEvent = self.formatData.trainFormatted[1][self.dataRef]
        self.currentTrainStatus = self.formatData.trainFormatted[2][self.dataRef]

    def getTrainInstance(self):
        return (self.currentTrainState,self.currentTrainEvent,self.currentTrainStatus)

    def newInstance(self):
        if self.dataRef < self.formatData.numTrainInstances-1:
            self.dataRef+=1
            self.currentTrainState = self.formatData.trainFormatted[0][self.dataRef]
            self.currentTrainEvent = self.formatData.trainFormatted[1][self.dataRef]
            self.currentTrainStatus = self.formatData.trainFormatted[2][self.dataRef]
        else:
            self.resetDataRef()

    def resetDataRef(self):
        self.dataRef = 0
        self.currentTrainState = self.formatData.trainFormatted[0][self.dataRef]
        self.currentTrainEvent = self.formatData.trainFormatted[1][self.dataRef]
        self.currentTrainStatus = self.formatData.trainFormatted[2][self.dataRef]
