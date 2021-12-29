# Import required moduldes------------------------------------
import numpy as np
import math
from survival_pareto import *
#-------------------------------------------------------------

class DataManagement:
    def __init__(self,dataFeatures,dataEventTimes,dataEventStatus,model):
        self.savedRawTrainingData = [dataFeatures,dataEventTimes,dataEventStatus]
        self.numAttributes = dataFeatures.shape[1]  # The number of attributes in the input file.
        self.attributeInfoType = [0] * self.numAttributes  #stores false (d) or true (c) depending on its type, which points to parallel reference in one of the below 2 arrays
        self.attributeInfoContinuous = [[np.inf,-np.inf] for _ in range(self.numAttributes)] #stores continuous ranges and NaN otherwise
        self.attributeInfoDiscrete = [0] * self.numAttributes  # stores arrays of discrete values or NaN otherwise.
        for i in range(0, self.numAttributes):
            self.attributeInfoDiscrete[i] = AttributeInfoDiscreteElement() #list of distinct values (see the last function in this script)

        # About Event times, events or censoring 
        self.discreteEvent = False  # Is the Class/Phenotype Discrete? (False = Continuous)
        self.eventList = [0,0]  # Stores maximum and minimum event times SHOULD THE MIN ALWAYS JUST BE ZERO?
        self.eventTypes = [] #should end up being just zero and 1, if not maybe should print error??
        #self.eventDict = {}
        self.eventRange = None  # Stores the difference between the maximum and minimum values for a continuous phenotype
        self.eventStatus = [] # Will store the event status (had the event = 1, censored = 0 for each instance)
        #self.calcSD = None #do we need this? 
        #self.missingEndpointList = [] #??
        self.isDefault = True  # Is discrete attribute limit an int or string
        try:
            int(model.discrete_attribute_limit)
        except:
            self.isDefault = False

        #Initialize some variables
        self.continuousCount = 0
        self.classPredictionWeights = {} #what are these for 
        self.averageStateCount = 0
        
        ##NEW
        self.eventRanked = [list(i) for i in sorted(zip(dataEventTimes,dataEventStatus))]#used in setEventProb, the probability of the event occuring within a rule's event interval 

        # About Dataset
        self.numTrainInstances = dataFeatures.shape[0]  # The number of instances in the training data
        self.discriminateEventTimes(dataEventTimes)
        self.discriminateEventStatus(dataEventStatus)

        self.discriminateAttributes(dataFeatures, model)
        self.characterizeAttributes(dataFeatures, model)

        #Rule Specificity Limit
        if model.rule_specificity_limit == None:
            i = 1
            uniqueCombinations = math.pow(self.averageStateCount,i)
            while uniqueCombinations < self.numTrainInstances:
                i += 1
                uniqueCombinations = math.pow(self.averageStateCount,i)
            model.rule_specificity_limit = min(i,self.numAttributes)

        self.trainFormatted = self.formatData(dataFeatures, dataEventTimes, dataEventStatus, model)  # The only np array
        
        #Initialize pareto front
        self.ecFront = Pareto() #'ECFront', epoch complete
        self.necFront = Pareto() #'NECFront NOT epoch complete
        
        
        #For speedy matching
        self.matchKey = {}
        for i in range(self.numTrainInstances):
            self.matchKey[i] = [] #list of matching classifierIDs + covering flag (doCovering)
            
#----------------------------------------------------------------------------------------------------------------------------
# Function discriminateEventStatus: counts how many of each event/censored are in the dataset? 
#---------------------------------------------------------------------------------------------------------------------------- 
    def discriminateEventStatus(self,dataEventStatus): 
        currentEventIndex = 0
        classCount = {} #dictionary containing the key (1 or 0) and value (count of each).
        while (currentEventIndex < self.numTrainInstances):
            target = dataEventStatus[currentEventIndex]
            if target in self.eventTypes:
                classCount[target]+=1
                self.classPredictionWeights[target] += 1
            else:
                self.eventTypes.append(target)
                classCount[target] = 1
                self.classPredictionWeights[target] = 1
            currentEventIndex+=1

        total = 0
        for eachClass in list(classCount.keys()): #(1 or 0)
            total += classCount[eachClass] #total number of classes, which will always be 2 in our case...
        for eachClass in list(classCount.keys()): #(1 or 0)
            self.classPredictionWeights[eachClass] = 1 - (self.classPredictionWeights[eachClass]/total) #standardize the class prediciton weights?
            
            
#----------------------------------------------------------------------------------------------------------------------------
# Function discriminateEventTimes: counts how many of each event time are in the dataset (key = eventTime, value = number of occurences). Is it more appropriate to use the prediction weights here??
#---------------------------------------------------------------------------------------------------------------------------- 
    def discriminateEventTimes(self,dataEventTimes): 
        currentEventIndex = 0
        classCount = {}
        while (currentEventIndex < self.numTrainInstances):
            target = dataEventTimes[currentEventIndex]
            if target in self.eventList:
                classCount[target]+=1
                self.classPredictionWeights[target] += 1
            else:
                self.eventList.append(target)
                classCount[target] = 1
                self.classPredictionWeights[target] = 1
            currentEventIndex+=1

        total = 0
        for eachClass in list(classCount.keys()):
            total += classCount[eachClass]
        for eachClass in list(classCount.keys()):
            self.classPredictionWeights[eachClass] = 1 - (self.classPredictionWeights[eachClass]/total)
            
#----------------------------------------------------------------------------------------------------------------------------
# Function discriminateAttributes: create a dictionary with key = state and value = # of times it appears (I think)
#---------------------------------------------------------------------------------------------------------------------------- 
    def discriminateAttributes(self,dataFeatures,model): 
        for att in range(self.numAttributes): #for eaach attribute 
            attIsDiscrete = True #set is discrete to true (what if it isn't?)
            if self.isDefault: # if the discrete atttribute limit is an integer
                currentInstanceIndex = 0
                stateDict = {}
                while attIsDiscrete and len(list(stateDict.keys())) <= model.discrete_attribute_limit and currentInstanceIndex < self.numTrainInstances:
                    target = dataFeatures[currentInstanceIndex,att] #retrieve the attribute value (features is an np array)
                    if target in list(stateDict.keys()): #if the attribute is present in the stateDict key (of a key value pair), add 1 (to the value)
                        stateDict[target] += 1
                    elif np.isnan(target): #if it is missing, pass
                        pass
                    else:
                        stateDict[target] = 1 #if it isn't already present, add it to stateDict.key and add 1 to the value. 
                    currentInstanceIndex+=1 #jump to the next index

                if len(list(stateDict.keys())) > model.discrete_attribute_limit: #if the length of the list of keys in stateDict is larger than the discrete attribute limit, set attisdisrete to false
                    attIsDiscrete = False
            elif model.discrete_attribute_limit == "c": #if the discrete attribute limit is "c" (see user guide:Multipurpose param. If it is a nonnegative integer, discrete_attribute_limit determines the threshold that determines if an attribute will be treated as a continuous or discrete attribute. For example, if discrete_attribute_limit == 10, if an attribute has more than 10 unique values in the dataset, the attribute will be continuous. If the attribute has 10 or less unique values, it will be discrete. Alternatively, discrete_attribute_limit can take the value of "c" or "d". See next param for this.)
                if att in model.specified_attributes: #and if att is in the specified attibutes from the model (If discrete_attribute_limit == "c", attributes specified by index in this param will be continuous and the rest will be discrete. If "d", attributes specified by index in this param will be discrete and the rest will be continuous)
                    attIsDiscrete = False #set attIsDiscrete to False
                else: #if it is not in the specified attributes, set it to true.
                    attIsDiscrete = True
            elif model.discrete_attribute_limit == "d": #if the discrete attribute limit is "d" 
                if att in model.specified_attributes: 
                    attIsDiscrete = True
                else:
                    attIsDiscrete = False

            if attIsDiscrete:
                self.attributeInfoType[att] = False
            else:
                self.attributeInfoType[att] = True
                self.continuousCount += 1
#----------------------------------------------------------------------------------------------------------------------------
# Function characterizeAttributes: identifies features as continuous or discrete and returns lists of the values present in each attribute (separately for continuous or discrete features) 
#---------------------------------------------------------------------------------------------------------------------------- 
    def characterizeAttributes(self,dataFeatures,model):
        for currentFeatureIndexInAttributeInfo in range(self.numAttributes): #for each feature index in attribute info
            for currentInstanceIndex in range(self.numTrainInstances): #for each instance in the environment 
                target = dataFeatures[currentInstanceIndex,currentFeatureIndexInAttributeInfo] #set the target as the attribute info (value)
                if not self.attributeInfoType[currentFeatureIndexInAttributeInfo]:#if attribute is discrete (recall that false = discrete and true = continuous for attributeInfoType)
                    if target in self.attributeInfoDiscrete[currentFeatureIndexInAttributeInfo].distinctValues or np.isnan(target): #if the value is missing or is already in the list of distict values, the pass
                        pass
                    else:
                        self.attributeInfoDiscrete[currentFeatureIndexInAttributeInfo].distinctValues.append(target) #otherwise append that value to the distinct values list 
                        self.averageStateCount += 1 #increase the state count by 1
                else: #if attribute is continuous, aka if "true"
                    if np.isnan(target): #and the value is missing, pass
                        pass #FOR THE NEXT LINES 115-118: Note these happen for each attribute across EACH instance....effectively updating the RANGE of the continuous attribute as it encounters all the instances
                    elif float(target) > self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][1]: #if the target value is greater than -inf
                        self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][1] = float(target) #set the SECOND list item of attributeInfoContinuous to "target" (the value of that attribute)
                    elif float(target) < self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][0]: #if the target value is less than inf
                        self.attributeInfoContinuous[currentFeatureIndexInAttributeInfo][0] = float(target) #set the FIRST list item of attributeInfoContinuous to  "target" (the value of that attribute)
                    else:
                        pass
            if self.attributeInfoType[currentFeatureIndexInAttributeInfo]: #if attribute is continuous
                self.averageStateCount += 2
        self.averageStateCount = self.averageStateCount/self.numAttributes
        
#----------------------------------------------------------------------------------------------------------------------------
# Function characterizeEventTimes: Determine the range of event times 
#----------------------------------------------------------------------------------------------------------------------------   
    def characterizeEventTimes(self,dataEventTimes,model): 
        timeeventList = [] #create an empty list 
        for currentInstanceIndex in range(len(dataEventTimes)): 
            target = dataEventTimes[currentInstanceIndex] 
            timeeventList.append(target)
            #Find Minimum and Maximum values for the continuous phenotype so we know the range.
            if np.isnan(target): #if it is missing, pass
                pass
            elif float(target) > self.eventList[1]:  
                self.eventList[1] = float(target)
            elif float(target) < self.eventList[0]:
                self.eventList[0] = float(target)
            else:
                pass
        self.eventSD = self.calcSD(timeeventList)#still need to fix this
        self.eventRange = self.eventList[1] - self.eventList[0]
        

#----------------------------------------------------------------------------------------------------------------------------
# Function calcSD: this seems to be only used for RBAs (in Ryan's continuous ExSTraCS - not sure if it's necessary here, but leaving it for now
#---------------------------------------------------------------------------------------------------------------------------- 
    def calcSD(self, timeeventList):
        """  Calculate the standard deviation of the continuous phenotype scores. """
        for i in range(len(timeeventList)):
            timeeventList[i] = float(timeeventList[i])

        avg = float(sum(timeeventList)/len(timeeventList))
        dev = []
        for x in timeeventList:
            dev.append(x-avg)
            sqr = []
        for x in dev:
            sqr.append(x*x)
            
        return math.sqrt(sum(sqr)/(len(sqr)-1))    
#----------------------------------------------------------------------------------------------------------------------------
# Function formatData:
#---------------------------------------------------------------------------------------------------------------------------- 
    def formatData(self,dataFeatures,dataEventTimes, dataEventStatus, model):
        formatted = np.insert(dataFeatures,self.numAttributes,dataEventTimes,dataEventStatus, 1) #Combines features and phenotypes into one array

        self.shuffleOrder = np.random.choice(self.numTrainInstances,self.numTrainInstances,replace=False) #e.g. first element in this list is where the first element of the original list will go
        shuffled = []
        for i in range(self.numTrainInstances):
            shuffled.append(None)
        for instanceIndex in range(self.numTrainInstances):
            shuffled[self.shuffleOrder[instanceIndex]] = formatted[instanceIndex]
        formatted = np.array(shuffled)

        shuffledFeatures = formatted[:,:-1].tolist()
        shuffledLabels = formatted[:,self.numAttributes].tolist() #might need to update this, because now the labels are two values 
        for i in range(len(shuffledFeatures)):
            for j in range(len(shuffledFeatures[i])):
                if np.isnan(shuffledFeatures[i][j]):
                    shuffledFeatures[i][j] = None
            if np.isnan(shuffledLabels[i]):
                shuffledLabels[i] = None
        return [shuffledFeatures,shuffledLabels]


class AttributeInfoDiscreteElement():
    def __init__(self):
        self.distinctValues = []
