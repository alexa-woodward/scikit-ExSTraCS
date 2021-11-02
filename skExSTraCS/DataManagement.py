import numpy as np
import math

class DataManagement:
    def __init__(self,dataFeatures,dataPhenotypes,model):
        self.savedRawTrainingData = [dataFeatures,dataPhenotypes]
        self.numAttributes = dataFeatures.shape[1]  # The number of attributes in the input file.
        self.attributeInfoType = [0] * self.numAttributes  # stores false (d) or true (c) depending on its type, which points to parallel reference in one of the below 2 arrays
        self.attributeInfoContinuous = [[np.inf,-np.inf] for _ in range(self.numAttributes)] #stores continuous ranges and NaN otherwise
        self.attributeInfoDiscrete = [0] * self.numAttributes  # stores arrays of discrete values or NaN otherwise.
        for i in range(0, self.numAttributes):
            self.attributeInfoDiscrete[i] = AttributeInfoDiscreteElement() #list of distinct values (see the last function in this script)

        # About Phenotypes
        self.discretePhenotype = True  # Is the Class/Phenotype Discrete? (False = Continuous)
        self.phenotypeList = []  # Stores all possible discrete phenotype states/classes or maximum and minimum values for a continuous phenotype
        self.phenotypeRange = None  # Stores the difference between the maximum and minimum values for a continuous phenotype
        self.isDefault = True  # Is discrete attribute limit an int or string
        try:
            int(model.discrete_attribute_limit)
        except:
            self.isDefault = False

        #Initialize some variables
        self.continuousCount = 0
        self.classPredictionWeights = {}
        self.averageStateCount = 0

        # About Dataset
        self.numTrainInstances = dataFeatures.shape[0]  # The number of instances in the training data
        self.discriminateClasses(dataPhenotypes)

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

        self.trainFormatted = self.formatData(dataFeatures, dataPhenotypes, model)  # The only np array

    def discriminateClasses(self,phenotypes): #counts how many of each class are in the dataset?
        currentPhenotypeIndex = 0
        classCount = {}
        while (currentPhenotypeIndex < self.numTrainInstances):
            target = phenotypes[currentPhenotypeIndex]
            if target in self.phenotypeList:
                classCount[target]+=1
                self.classPredictionWeights[target] += 1
            else:
                self.phenotypeList.append(target)
                classCount[target] = 1
                self.classPredictionWeights[target] = 1
            currentPhenotypeIndex+=1

        total = 0
        for eachClass in list(classCount.keys()):
            total += classCount[eachClass]
        for eachClass in list(classCount.keys()):
            self.classPredictionWeights[eachClass] = 1 - (self.classPredictionWeights[eachClass]/total)

    def discriminateAttributes(self,features,model): #create a dictionary with key = state and value = # of times it appears. 
        for att in range(self.numAttributes): #for eaach attribute 
            attIsDiscrete = True #set is discrete to true (what if it isn't?)
            if self.isDefault: # if the discrete atttribute limit is an integer
                currentInstanceIndex = 0
                stateDict = {}
                while attIsDiscrete and len(list(stateDict.keys())) <= model.discrete_attribute_limit and currentInstanceIndex < self.numTrainInstances:
                    target = features[currentInstanceIndex,att] #retrieve the attribute value (features is an np array)
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

    def characterizeAttributes(self,features,model):
        for currentFeatureIndexInAttributeInfo in range(self.numAttributes): #for each feature index in attribute info
            for currentInstanceIndex in range(self.numTrainInstances): #for each instance in the environment 
                target = features[currentInstanceIndex,currentFeatureIndexInAttributeInfo] #set the target as the attribute info (value)
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

    def formatData(self,features,phenotypes,model):
        formatted = np.insert(features,self.numAttributes,phenotypes,1) #Combines features and phenotypes into one array

        self.shuffleOrder = np.random.choice(self.numTrainInstances,self.numTrainInstances,replace=False) #e.g. first element in this list is where the first element of the original list will go
        shuffled = []
        for i in range(self.numTrainInstances):
            shuffled.append(None)
        for instanceIndex in range(self.numTrainInstances):
            shuffled[self.shuffleOrder[instanceIndex]] = formatted[instanceIndex]
        formatted = np.array(shuffled)

        shuffledFeatures = formatted[:,:-1].tolist()
        shuffledLabels = formatted[:,self.numAttributes].tolist()
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
