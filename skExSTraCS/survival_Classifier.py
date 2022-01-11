import random
import copy
import numpy as np
from survival_DataManagement import *

class Classifier: #this script is for an INDIVIDUAL CLASSIFIER
    def __init__(self,model):
        #Major Parameters --------------------------------------------------
        self.specifiedAttList = []
        self.condition = []
        self.eventStatus = None
        self.eventTime = None    
        self.event_RP = None        #NEW - probability of this event time occurring by chance.
    
        self.fitness = model.init_fitness #cant remember what its set at 
        self.relativeIndFitness = None
        self.accuracy = 0.0
        self.accuracyComponent = 0.0
        self.numerosity = 1
        self.coverDiff = 1 #Number of instances correctly covered by rule beyond what would be expected by chance.
        self.aveMatchSetSize = None
        self.deletionProb = None
        
        #Individual Survival Prediction -----------------------------------------------
        self.coverTimes = [] #for each rule, a list of eventTimes from instances that were correctly covered 
        
        #Experience Management ---------------------------------------------
        self.timeStampGA = None
        self.initTimeStamp = None
        self.epochComplete = False
        
        #Classifier Accuracy Tracking --------------------------------------
        self.matchCount = 0
        self.correctCount = 0
        self.matchCover = 0 #what are these? 
        self.correctCover = 0
        
        #Continuous Endpoint
        self.errorSum = 0
        self.errorCount = 0
        

#----------------------------------------------------------------------------------------------------------------------------
# initializeByCopy: XXX What is this doing? 
#----------------------------------------------------------------------------------------------------------------------------  
    def initializeByCopy(self,toCopy,iterationCount): #idk what this means 
        self.specifiedAttList = copy.deepcopy(toCopy.specifiedAttList)
        self.condition = copy.deepcopy(toCopy.condition)
        self.phenotype = copy.deepcopy(toCopy.phenotype)
        self.timeStampGA = iterationCount
        self.initTimeStamp = iterationCount
        self.aveMatchSetSize = copy.deepcopy(toCopy.aveMatchSetSize)
        self.fitness = toCopy.fitness
        self.accuracy = toCopy.accuracy
                    
### THE FUNCTION BELOW COPIED FROM [here](https://github.com/alexa-woodward/scikit-ExSTraCS/blob/master/continuous_endpoint_ExSTraCS/exstracs_classifier.py) on 11/15, will delete later...provides strategy for covering with continuous endpoint. Will need to update how this is done based on whether the event status is 1 or 0.                
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CLASSIFIER CONSTRUCTION METHODS
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     

#--------------------------------------------------------------------------------------------- 
# New covering function for survival outcomes - updated 11/15
#---------------------------------------------------------------------------------------------
                    
    def initializeByCovering(self,model,setSize,state,eventTime,eventStatus): #will need to add a way to do this for the continuous outcome!
        self.timeStampGA = model.iterationCount #the timestamp is set to what iteration we're on
        self.initTimeStamp = model.iterationCount #same
        self.aveMatchSetSize = setSize #zero to start?
        #self.event = event #this will have to change
        self.eventTime = eventTime #time of event or censoring 
        self.eventStatus = eventStatus #event status, failed = 1, censored = 0
        self.discreteEvent = False #may or may not need this - will update data_management.py file "discreteEvent" being "discretePhenotype"...set to false 
    
        toSpecify = random.randint(1, model.rule_specificity_limit) #RSL gets set in the data_management.py...draws a random integer within the range 1 to RSL (i.e., how many attributes can be specified within a given rule).
        if model.doExpertKnowledge: #if the model uses expert knowledge, do the following:
            i = 0
            while len(self.specifiedAttList) < toSpecify and i < model.env.formatData.numAttributes - 1:
                target = model.EK.EKRank[i]
                if state[target] != None:
                    self.specifiedAttList.append(target)
                    self.condition.append(self.buildMatch(model,target,state))
                i += 1
        else: #if not, then:
            potentialSpec = random.sample(range(model.env.formatData.numAttributes),toSpecify) #randomly sample "toSpecify" values from the range = the number of attributes
            for attRef in potentialSpec: #for each attribute specified
                if state[attRef] != None: #if the state of that attribute is not none
                    self.specifiedAttList.append(attRef) #append the attribute (position?) to the specific attribute list
                    self.condition.append(self.buildMatch(model,attRef,state)) #also append the condition of that attribute
                
### Creating a continuous event range (endpoint):           
        if self.eventStatus == 1: #if the event occured
            eventRange = model.env.formatData.eventList[1] - model.env.formatData.eventList[0] #basically this should be equal Tmax 
            rangeRadius = random.randint(25,75)*0.01*eventRange / 2.0 #Continuous initialization domain radius.
            Low = float(eventTime) - rangeRadius
            High = float(eventTime) + rangeRadius
            self.eventInterval = [Low,High]  
            self.setEventProb(model,model.env.formatData.eventRanked) #this might need to have eventStatus as a parameter...see 
        else: #if the instance was censored
            eventRange = model.env.formatData.eventList[1] - model.env.formatData.eventList[0] #again, this should be the same at Tmax
            rangeRadius = random.randint(25,75)*0.01*eventRange / 2.0 #Continuous initialization domain radius, same as above
            adjEvent = random.randrange(eventTime, model.env.formatData.eventList[1]+1,1) #create an adjusted event time - randomly choose a value greater than the censoring time and below Tmax, form the range around that
            Low = float(adjEvent) - rangeRadius #build the range around the new adjusted event time 
            High = float(adjEvent) + rangeRadius
            self.eventInterval = [Low,High]

#--------------------------------------------------------------------------------------------- 
# evaluateAccuracyAndInitialFitness: going to need to add the updateFront function in here I think
#---------------------------------------------------------------------------------------------                                                 
    def evaluateAccuracyAndInitialFitness(self,model,nextID): #need to add event status here too #This method should only be called once it is CERTAIN a classifier will be added to the population.
        training_data = model.env.formatData.trainFormatted
        num_instances = model.env.formatData.numTrainInstances
        match_count = 0
        correct_count = 0

        for instance_index in range(num_instances): #for each instance in the environment 
            state = training_data[0][instance_index]
            eventTime = training_data[1][instance_index]
            eventStatus = training_data[2][instance_index]
            if self.match(model, state): #apply match function 
                match_count += 1 #call updateExperience?
                model.env.formatData.matchKey[instance_index].append(nextID) #Add that this rule matches with this training instance
                if eventStatus == 1:
                    if float(eventTime) <= float(self.eventInterval[1]) and float(eventTime) >= float(self.eventInterval[0]):
                        correct_count += 1
                        self.updateCorrect()
                        self.updateError(model,eventTime,eventStatus)
                        self.updateCorrectTimes(eventTime) #appends the eventTime to a list of "correctTimes" for each correctly matched training instance
                        self.updateCorrectCoverage()
                    else: 
                        self.updateIncorrectError()
                else: #if the instance was censored, append to the correct set IF the interval includes the censoring time or the interval is BEYOND the censoring time
                    if (float(eventTime) <= float(self.eventInterval[1]) and float(eventTime) >= float(self.eventInterval[0])) or (float(eventTime) < float(self.eventInterval[0])):
                        correct_count += 1
                        self.updateError(model,eventTime,eventStatus)
                    else:    
                        self.updateIncorrectError()
        try:
            self.accuracy = updateAccuracy(model) #updateError has now been called above
        except:
            self.accuracy = (correct_count / match_count)   #keeping this here just in case      
#        self.ID = nextID #I dont think we need this
        self.updateFitness(model) #this calls the pareto fitness function  
        self.epochComplete = True #set epochComplete (For this rule) equal to TRUE..later could remove all references to epochComplete
#----------------------------------------------------------------------------------------------------------------------------
# setEventProb: Calculate the relative probability that an event time of an instance in the training data will fall withing the event range specified by this rule. 
#----------------------------------------------------------------------------------------------------------------------------                              
    def setEventProb(self,model,eventRanked): #only considers instances with eventStatus = 1
        count = 0
        ref = 0
#         print self.eventRanked
#         print self.eventList
        while ref < len(eventRanked) and eventRanked[ref][0] <= self.eventInterval[1]:
            if eventRanked[ref][0] >= self.eventInterval[0] and eventRanked[ref][1] == 1:
                count += 1
            ref += 1

        self.event_RP = count/float(model.env.formatData.numTrainInstances)
            
                
#----------------------------------------------------------------------------------------------------------------------------
# Build match function: create a condition that matches the attributes in an instance, called in the above function initalizebyCovering 
#----------------------------------------------------------------------------------------------------------------------------    

    def buildMatch(self,model,attRef,state): 
        attributeInfoType = model.env.formatData.attributeInfoType[attRef] #set the type of attribute (discrete/continuous) (see lines #96-100 of data_mangement.py)
        if not (attributeInfoType):  # Discrete
            attributeInfoValue = model.env.formatData.attributeInfoDiscrete[attRef] #if not "true", set the attributeInfoValue to...
        else: #continuous
            attributeInfoValue = model.env.formatData.attributeInfoContinuous[attRef] #pulls from attributeInfoContinous (a list of lists I think, or a np array)...see data_management.py

        if attributeInfoType: #Continuous Attribute (true = continuous) (i.e., if TRUE:...)
            attRange = attributeInfoValue[1] - attributeInfoValue[0] #okkkk figured this out. the infovalues form from attributeInfoContinuous which is (a np array I think) of max and min values of each attribute across all the instances. So index 1 minus index 0 of "attributeInfoValue" gives the range 
            rangeRadius = random.randint(25, 75) * 0.01 * attRange / 2.0  # initialize a continuous domain radius. draw a random integer between 25 and 75 (why these numbers?), multiply by 0.01, by the attRange, and divide by 2
            Low = state[attRef] - rangeRadius 
            High = state[attRef] + rangeRadius
            condList = [Low, High] #Condition list is the range of values from low to high
        else:
            condList = state[attRef] #for a discrete attribute, the condition list is simply its state
        return condList

#----------------------------------------------------------------------------------------------------------------------------
# updateEpochStatus: determines whether or not the classifier has seen all of the instances. If true, set epochComplete = True. WON'T NEED THIS LATER
#----------------------------------------------------------------------------------------------------------------------------  

    def updateEpochStatus(self,model):  
        if not self.epochComplete and (model.iterationCount - self.initTimeStamp - 1) >= model.env.formatData.numTrainInstances:
            self.epochComplete = True
#----------------------------------------------------------------------------------------------------------------------------
# match: #this funtion matches attributes (from instances) to the conditions (from a rule) - this will likely stay the same, only deals with features, not the outcome
#---------------------------------------------------------------------------------------------------------------------------- 
    def match(self, model, state): #this funtion matches attributes (from instances) to the conditions (from a rule)
        for i in range(len(self.condition)): #for each attribute in the condition:
            specifiedIndex = self.specifiedAttList[i] #get the index of that attribute 
            attributeInfoType = model.env.formatData.attributeInfoType[specifiedIndex] #get whether it is discrete or continuous
            # Continuous
            if attributeInfoType:
                instanceValue = state[specifiedIndex]
                if instanceValue == None:
                    return False
                elif self.condition[i][0] < instanceValue < self.condition[i][1]: #making sure the instance value falls within the range?
                    pass
                else:
                    return False

            # Discrete
            else: #if attribute is discrete
                stateRep = state[specifiedIndex] #set its state (value) at the specified index
                if stateRep == self.condition[i]: #if the state (value) is the same as the condition at position i, pass and return true
                    pass
                elif stateRep == None: #if the state is none or is not the same as condition [i], return false
                    return False
                else:
                    return False
        return True 
    
#----------------------------------------------------------------------------------------------------------------------------
# equals: checks to see if there are any duplicate rules (can we look at intervals here? or need to do mid range of intervals?)
#---------------------------------------------------------------------------------------------------------------------------- 
    def equals(self,cl): 
        if cl.eventInterval == self.eventInterval and len(cl.specifiedAttList) == len(self.specifiedAttList): #if the event ranges are the same and the list of attributes are the same length, check the following...
            clRefs = sorted(cl.specifiedAttList) #sort the attribute indexes for the classifier
            selfRefs = sorted(self.specifiedAttList) #sort the attribute indexes
            if clRefs == selfRefs: #if they are the same, then...
                for i in range(len(cl.specifiedAttList)): #for each attribute in the classifier 
                    tempIndex = self.specifiedAttList.index(cl.specifiedAttList[i]) #the following checks to see if the conditions also match 
                    if not (cl.condition[i] == self.condition[tempIndex]):
                        return False #if the don't all match, return false
                return True #if they do all match, return true (I assume somewhere this would update the numerosity)
        return False #if the phenotypes and lengths of the specified attribute lists of two classifiers don't match, return false
    
    ## All the rest of this stuff would probably stay very similar
#----------------------------------------------------------------------------------------------------------------------------
# updateExperience: updates how many instances the classifier (rule) has seen. If it has seen all the instances, pass. WON'T NEED THIS TO UPDATE MATCHCOUNT  (since it only updates if epochComplete = False
#---------------------------------------------------------------------------------------------------------------------------- 
    def updateExperience(self): #add 1 to either the matchcount or the matchcover, depending on what was needed 
        self.matchCount += 1
        if self.epochComplete:  # Once epoch Completed, number of matches for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.matchCover += 1
#----------------------------------------------------------------------------------------------------------------------------
# updateMatchSetSize: #update the match set size. "beta" is set at 0.2, a learning parameter used in calculating the average correct set size
#---------------------------------------------------------------------------------------------------------------------------- 
    def updateMatchSetSize(self, model,matchSetSize):  
        if self.matchCount < 1.0 / model.beta: # if the match count is less than 5
            self.aveMatchSetSize = (self.aveMatchSetSize * (self.matchCount-1)+ matchSetSize) / float(self.matchCount) #update the average to...
        else: #if its not less than 5,
            self.aveMatchSetSize = self.aveMatchSetSize + model.beta * (matchSetSize - self.aveMatchSetSize)
#----------------------------------------------------------------------------------------------------------------------------
# updateCorrect: updates the correct count for a classifier until all instances have been seen, once epochComplete add to "correctCover" instead. WON'T NEED THIS TO UPDATE CORREXT COUNT (since it only updates if epochComplete = False)
#---------------------------------------------------------------------------------------------------------------------------- 
    def updateCorrect(self):
        self.correctCount += 1
        if self.epochComplete: #Once epoch Completed, number of correct for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.correctCover += 1
            
#----------------------------------------------------------------------------------------------------------------------------
# updateCorrectTimes: updates a list of eventTimes from correctly matched rules. Called in evaluateAccuracyandInitalFitness 
#---------------------------------------------------------------------------------------------------------------------------- 
    def updateCorrectTimes(self,eventTime):
        self.coverTimes.append(eventTime)

#----------------------------------------------------------------------------------------------------------------------------
# updateCorrectCoverage: updates the correct "coverage" for a rule, the difference between the true number of covered instances and the expected coverage
#----------------------------------------------------------------------------------------------------------------------------                   
    def updateCorrectCoverage(self):
        """ """
        self.coverDiff = self.correctCover - self.event_RP*self.matchCover
#         print self.coverDiff
#         expectedCoverage = self.numTrainInstances*self.totalFreq
#         self.coverDiff = self.correctCover - expectedCoverage  

#----------------------------------------------------------------------------------------------------------------------------
# updateError: updates the error for a classifier until all instances have been seen
#----------------------------------------------------------------------------------------------------------------------------             
    def updateError(self,model,eventTime,eventStatus):
#        if not self.epochComplete: Removed this 
        if eventStatus == 1:
            high = self.eventInterval[1]
            low = self.eventInterval[0]
            if self.eventInterval[1] > model.env.formatData.eventList[1]:
                high = model.env.formatData.eventList[1]
            if self.eventInterval[0] < model.env.formatData.eventList[0]:
                low = model.env.formatData.eventList[0]
                
            rangeCentroid = (high + low) / 2.0
            error = abs(rangeCentroid - eventTime)  #this or self.eventTime?
            adjustedError = error / (model.env.formatData.eventList[1] - model.env.formatData.eventList[0]) #Error is fraction of total phenotype range (i.e. maximum possible error)
        else: #if the eventStatus is 0 (censored instance)
            adjustedError = 0 #do not update the error, i.e., adjusted error for this instance is 0
        self.errorSum += adjustedError  
        self.errorCount += 1        
            
#----------------------------------------------------------------------------------------------------------------------------
# updateIncorrectError: adds the max error (1) to errorSum if the instance falls in [I] 
#----------------------------------------------------------------------------------------------------------------------------                 
    def updateIncorrectError(self):        
#        if not self.epochComplete:
        self.errorSum += 1.0
        self.errorCount += 1        
#----------------------------------------------------------------------------------------------------------------------------
# updateAccuracy: update the accuracy for a classifier. Going to change this so it's related to the ERROR. Might update to MFF
#---------------------------------------------------------------------------------------------------------------------------- 

#New as of 12/10, copied over from pareto version: for explanation, see [here](https://github.com/alexa-woodward/surviving-heterogeneity/blob/master/docs/misc/Retooling%20Fitenss%20for%20Noisy%20Problems%20in%20an%20LCS.pdf)             
    def updateAccuracy(self,model):
        """ Update the accuracy tracker """
        nonUsefulDiscount = 0.001 #ohhhh this is just 1/1000, or 1/cover opportunity 
        coverOpportunity = 1000 #number of times a rule has seen instances...
        adjAccuracy = 0
        #-----------------------------------------------------------------------------------
        # CALCULATE ACCURACY
        #-----------------------------------------------------------------------------------
        try:
            self.accuracy = 1 - (self.errorSum/self.matchCover) # 1- average error based on range centroid.  Should be natural pressure to achieve narrow endpoint range.
        except:
            print("CorrectCover: " + str(self.correctCover))
            print("MatchCover: " + str(self.matchCover))
            print("MatchCount: " + str(self.matchCount))
            print("InitTime: " + str(self.initTimeStamp))
            print("EpochComplete: " + str(self.epochComplete))
            raise NameError("Problem with updating accuracy")
        

        #-----------------------------------------------------------------------------------
        # CALCULATE ADJUSTED ACCURACY
        #-----------------------------------------------------------------------------------
        if self.accuracy > self.event_RP: #if the accuracy is greater than the probability that the event time will fall in the event range of the rule
            adjAccuracy = self.accuracy - self.event_RP #adjust the accuracy by subtracting that probability 
        elif self.matchCover == 2 and self.correctCover == 1 and not self.epochComplete and (model.interationCount - self.timeStampGA) < coverOpportunity: #else, if the rule has matched two instances but only been correct once, and the rules is NOT epoch complete and the difference between the number of iterations and the time stamp is less than 1000, 
            adjAccuracy = self.event_RP / 2.0 #set the accuracy to HALF the probability
        else:
            adjAccuracy = self.accuracy * nonUsefulDiscount #?? #else, multiple the accuracy by the nonUsefulDiscount (1/cover opportunity ) 
        #-----------------------------------------------------------------------------------
        # CALCULATE ACCURACY COMPONENT
        #-----------------------------------------------------------------------------------
        maxAccuracy = 1-self.event_RP #set max accuracy to 1 - the probability that the event time will fall in the event range of the rule
        if maxAccuracy == 0: #if max accuracy is zero
            self.accuracyComponent = 0 #set the accuracy component to zero
        else: 
            self.accuracyComponent = adjAccuracy / float(maxAccuracy) #Accuracy contribution scaled between 0 and 1 allowing for different maximum accuracies
        self.accuracyComponent = 2*((1/float(1+math.exp(-5*self.accuracyComponent)))-0.5)/float(0.98661429815) #what is all this
        self.accuracyComponent = math.pow(self.accuracyComponent,1)
        
#----------------------------------------------------------------------------------------------------------------------------
# updateFitness: Calculates the fitness of an individual rule based on it's accuracy and correct coverage relative to the 'Pareto' front
#---------------------------------------------------------------------------------------------------------------------------- 
#old, prior to Pareto front being used to calculate fitness.
#     def updateFitness(self):
#        """ Update the fitness parameter. """ 
#         if (self.eventInterval[1]-self.eventInterval[0]) >= self.eventRange:
#             self.fitness = pow(0.0001, 5)
#         else:
#             if self.matchCover < 2 and self.epochComplete:
#                 self.fitness = pow(0.0001, 5)
#             else:
#                 self.fitness = pow(self.accuracy, model.nu) #- (self.phenotype[1]-self.phenotype[0])/cons.env.formatData.phenotypeRange)
                    
                                      
    def updateFitness(self,model): 
        if self.coverDiff > 0: 
            self.fitness = model.env.formatData.ecFront.getParetoFitness([self.accuracyComponent,self.coverDiff])
#Got rid of all the stuff here that was if: epochComplete = False
        else: #if coverDiff is not greater than 0, set fitness to a really small number
#             print 'poor'
#             print self.accuracyComponent
            self.fitness = self.accuracyComponent / float(1000)

        if self.fitness < 0:
            print("negative fitness error")
        if round(self.fitness,5) > 1: #rounding added to handle odd division error, where 1.0 was being treated as a very small decimal just above 1.0
            print("big fitness error")

        self.lastFitness = copy.deepcopy(self.fitness) #what is this?    

                    
#----------------------------------------------------------------------------------------------------------------------------
# updateNumerosity: THIS WILL PROBABLY STAY THE SAME
#---------------------------------------------------------------------------------------------------------------------------- 
    def updateNumerosity(self, num):
        """ Alters the numerosity of the classifier.  Notice that num can be negative! """
        self.numerosity += num #but where does num come from??
#----------------------------------------------------------------------------------------------------------------------------
# isSubsumer: Returns if the classifier (self) is a possible subsumer. A classifier must have sufficient experience (one epoch) and it must also be as or more accurate than the classifier it is trying to subsume.  """
#---------------------------------------------------------------------------------------------------------------------------- 
    def isSubsumer(self, model): #is the match count and accuracy of a more general rule just as good as the more specific one? If so,  return true
        if self.matchCount > model.theta_sub and self.accuracy > model.acc_sub: #if the match count is greater than the theta_sub (subsumption experience threshold, default = 20) and the accuracy is greater than the acc_sub (default = 0.99), return true
            return True
        return False
   
#----------------------------------------------------------------------------------------------------------------------------
# subsumes: Returns if the classifier (self) subsumes cl - updated 11/29
#---------------------------------------------------------------------------------------------------------------------------- 
    def subsumes(self,model,cl):
        #FOR SURVIVAL DATA
        if self.event[0] >= cl.event[0] and self.event[1] <= cl.event[1]:
                if self.isSubsumer() and self.isMoreGeneral(cl):
                    return True
        return False
#----------------------------------------------------------------------------------------------------------------------------
# isMoreGeneral: Returns if the classifier (self) is more general than cl. Check that all attributes specified in self are also specified in cl. Should remain the same
#---------------------------------------------------------------------------------------------------------------------------- 
    def isMoreGeneral(self,model, cl): #
        if len(self.specifiedAttList) >= len(cl.specifiedAttList): #if the length of the specified attribute list for one classifier is greater than or equal to the that of the other, return false (classifier is more specific)
            return False
        for i in range(len(self.specifiedAttList)): #check if there are attributes specified in one that are not specified in the other, return false
            attributeInfoType = model.env.formatData.attributeInfoType[self.specifiedAttList[i]]
            if self.specifiedAttList[i] not in cl.specifiedAttList:
                return False

            # Continuous
            if attributeInfoType:
                otherRef = cl.specifiedAttList.index(self.specifiedAttList[i])
                if self.condition[i][0] < cl.condition[otherRef][0]:
                    return False
                if self.condition[i][1] > cl.condition[otherRef][1]:
                    return False
        return True #otherwise, return true 
    
    
   
#----------------------------------------------------------------------------------------------------------------------------
# updateTimeStamp: Sets the time stamp of the classifier.
#---------------------------------------------------------------------------------------------------------------------------- 
    def updateTimeStamp(self, ts): #where does ts come from?
        """ Sets the time stamp of the classifier. """
        self.timeStampGA = ts
#----------------------------------------------------------------------------------------------------------------------------
# uniformCrossover: Started updating 11/29. see code_notes for questions. calls "eventCrossover" below, 50% of the time 
#---------------------------------------------------------------------------------------------------------------------------- 
    def uniformCrossover(self,model,cl): 
        if random.random() < 0.5: #50% of the time crossover the condition, 50% crossover the eventrange 
            p_self_specifiedAttList = copy.deepcopy(self.specifiedAttList) #A deep copy constructs a new compound object and then, recursively, inserts copies into it of the objects found in the original.
            p_cl_specifiedAttList = copy.deepcopy(cl.specifiedAttList) #deep copy the attribute list of two parent rules 

            useAT = model.do_attribute_feedback and random.random() < model.AT.percent #do_attribute_feedback defaults to true - not sure where AT.percent comes from

            comboAttList = [] #create a combined attribute list 
            for i in p_self_specifiedAttList: #for each in the specified attribute list
                comboAttList.append(i) #append each to the combo list 
            for i in p_cl_specifiedAttList: #for each in the other list
                if i not in comboAttList: #if its not already in the combo list
                    comboAttList.append(i) #append it to the combo list
                elif not model.env.formatData.attributeInfoType[i]:  # Attribute specified in both parents, and the attribute is discrete (then no reason to cross over)
                    comboAttList.remove(i)
            comboAttList.sort() #sort the combined list 

            changed = False
            for attRef in comboAttList: #for each attribute in the combo list
                attributeInfoType = model.env.formatData.attributeInfoType[attRef] #set the type (discrete/continuous)
                if useAT: #if AT is true (default is, ATTRIBUTE CROSSOVER PROBAILITY - ATTRIBUTE FEEDBACK)
                    probability = model.AT.getTrackProb()[attRef] #set the probability (see attribute_tracking.py)
                else: #ATTRIBUTE CROSSOVER PROBAILITY - NORMAL CROSSOVER
                    probability = 0.5

                ref = 0
                if attRef in p_self_specifiedAttList:
                    ref += 1
                if attRef in p_cl_specifiedAttList:
                    ref += 1

                if ref == 0:
                    pass
                elif ref == 1:
                    if attRef in p_self_specifiedAttList and random.random() > probability:
                        i = self.specifiedAttList.index(attRef)
                        cl.condition.append(self.condition.pop(i))

                        cl.specifiedAttList.append(attRef)
                        self.specifiedAttList.remove(attRef)
                        changed = True

                    if attRef in p_cl_specifiedAttList and random.random() < probability:
                        i = cl.specifiedAttList.index(attRef)
                        self.condition.append(cl.condition.pop(i))

                        self.specifiedAttList.append(attRef)
                        cl.specifiedAttList.remove(attRef)
                        changed = True
                else:
                    # Continuous Attribute
                    if attributeInfoType:
                        i_cl1 = self.specifiedAttList.index(attRef)
                        i_cl2 = cl.specifiedAttList.index(attRef)
                        tempKey = random.randint(0, 3)
                        if tempKey == 0:
                            temp = self.condition[i_cl1][0]
                            self.condition[i_cl1][0] = cl.condition[i_cl2][0]
                            cl.condition[i_cl2][0] = temp
                        elif tempKey == 1:
                            temp = self.condition[i_cl1][1]
                            self.condition[i_cl1][1] = cl.condition[i_cl2][1]
                            cl.condition[i_cl2][1] = temp
                        else:
                            allList = self.condition[i_cl1] + cl.condition[i_cl2]
                            newMin = min(allList)
                            newMax = max(allList)
                            if tempKey == 2:
                                self.condition[i_cl1] = [newMin, newMax]
                                cl.condition.pop(i_cl2)

                                cl.specifiedAttList.remove(attRef)
                            else:
                                cl.condition[i_cl2] = [newMin, newMax]
                                self.condition.pop(i_cl1)

                                self.specifiedAttList.remove(attRef)

                    # Discrete Attribute
                    else:
                        pass

            #Specification Limit Check
            if len(self.specifiedAttList) > model.rule_specificity_limit:
                self.specLimitFix(model,self)
            if len(cl.specifiedAttList) > model.rule_specificity_limit:
                self.specLimitFix(model,cl)

            tempList1 = copy.deepcopy(p_self_specifiedAttList)
            tempList2 = copy.deepcopy(cl.specifiedAttList)
            tempList1.sort()
            tempList2.sort()
            if changed and (tempList1 == tempList2):
                changed = False
            return changed
        else: 
            return self.eventCrossover(cl, eventTime)
        
#----------------------------------------------------------------------------------------------------------------------------
# eventCrossover: Crossover the continuous event interval
#----------------------------------------------------------------------------------------------------------------------------      
        
    def eventCrossover(self, cl, eventTime):
        changed = False
        if self.eventInterval[0] == cl.eventInterval[0] and self.eventInterval[1] == cl.eventInterval[1]:
            return changed
        else:
            tempKey = random.random() < 0.5 #Make random choice between 4 scenarios, Swap minimums, Swap maximums, Children preserve parent phenotypes.
            if tempKey: #Swap minimum
                temp = self.eventInterval[0]
                self.eventInterval[0] = cl.eventInterval[0]
                cl.eventInterval[0] = temp
                changed = True
            elif tempKey:  #Swap maximum
                temp = self.eventInterval[1]
                self.eventInterval[1] = cl.eventInterval[1]
                cl.eventInterval[1] = temp
                changed = True
            if not self.eventInterval[0] < eventTime or not self.eventInterval[1] > eventTime: #for these next two statements, print "event interval crossover range error" if eventTime does not fall within the two intervals
                print('event interval crossover range error')
            if not cl.eventInterval[0] < eventTime or not cl.eventInterval[1] > eventTime:
                print('eventInterval crossover range error')
        return changed        
#----------------------------------------------------------------------------------------------------------------------------
# specLimitFix: Lowers classifier specificity to specificity limit.
#---------------------------------------------------------------------------------------------------------------------------- 
    def specLimitFix(self, model, cl):
        """ Lowers classifier specificity to specificity limit. """
        if model.do_attribute_feedback:
            # Identify 'toRemove' attributes with lowest AT scores
            while len(cl.specifiedAttList) > model.rule_specificity_limit:
                minVal = model.AT.getTrackProb()[cl.specifiedAttList[0]]
                minAtt = cl.specifiedAttList[0]
                for j in cl.specifiedAttList:
                    if model.AT.getTrackProb()[j] < minVal:
                        minVal = model.AT.getTrackProb()[j]
                        minAtt = j
                i = cl.specifiedAttList.index(minAtt)  # reference to the position of the attribute in the rule representation
                cl.specifiedAttList.remove(minAtt)
                cl.condition.pop(i)  # buildMatch handles both discrete and continuous attributes

        else:
            # Randomly pick 'toRemove'attributes to be generalized
            toRemove = len(cl.specifiedAttList) - model.rule_specificity_limit
            genTarget = random.sample(cl.specifiedAttList, toRemove)
            for j in genTarget:
                i = cl.specifiedAttList.index(j)  # reference to the position of the attribute in the rule representation
                cl.specifiedAttList.remove(j)
                cl.condition.pop(i)  # buildMatch handles both discrete and continuous attributes
#----------------------------------------------------------------------------------------------------------------------------
# setAccuracy: Sets the accuracy of the classifier
#---------------------------------------------------------------------------------------------------------------------------- 
    def setAccuracy(self, acc):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc
#----------------------------------------------------------------------------------------------------------------------------
# setFitness: Sets the fitness of the classifier
#---------------------------------------------------------------------------------------------------------------------------- 
    def setFitness(self, fit):
        """  Sets the fitness of the classifier. """
        self.fitness = fit
#----------------------------------------------------------------------------------------------------------------------------
# mutation:  Mutates the condition of the classifier. Also handles phenotype mutation. This is a niche mutation, which means that the resulting classifier will still match the current instance. 
#---------------------------------------------------------------------------------------------------------------------------- 
    def mutation(self,model,state):
        """ Mutates the condition of the classifier. Also handles phenotype mutation. This is a niche mutation, which means that the resulting classifier will still match the current instance.  """
        pressureProb = 0.5  # Probability that if EK is activated, it will be applied.
        useAT = model.do_attribute_feedback and random.random() < model.AT.percent
        changed = False

        steps = 0
        keepGoing = True
        while keepGoing:
            if random.random() < model.mu:
                steps += 1
            else:
                keepGoing = False

        # Define Spec Limits
        if (len(self.specifiedAttList) - steps) <= 1:
            lowLim = 1
        else:
            lowLim = len(self.specifiedAttList) - steps
        if (len(self.specifiedAttList) + steps) >= model.rule_specificity_limit:
            highLim = model.rule_specificity_limit
        else:
            highLim = len(self.specifiedAttList) + steps
        if len(self.specifiedAttList) == 0:
            highLim = 1

        # Get new rule specificity.
        newRuleSpec = random.randint(lowLim, highLim) #are we going to keep the same number of attributes, increase (increase specificity) or decrease?

        # MAINTAIN SPECIFICITY
        if newRuleSpec == len(self.specifiedAttList) and random.random() < (1 - model.mu):
            #Remove random condition element
            if not model.doExpertKnowledge or random.random() > pressureProb:
                genTarget = random.sample(self.specifiedAttList,1)
            else:
                genTarget = self.selectGeneralizeRW(model,1)

            attributeInfoType = model.env.formatData.attributeInfoType[genTarget[0]]
            if not attributeInfoType or random.random() > 0.5:
                if not useAT or random.random() > model.AT.getTrackProb()[genTarget[0]]:
                    # Generalize Target
                    i = self.specifiedAttList.index(genTarget[0])  # reference to the position of the attribute in the rule representation
                    self.specifiedAttList.remove(genTarget[0])
                    self.condition.pop(i)  # buildMatch handles both discrete and continuous attributes
                    changed = True
            else:
                self.mutateContinuousAttributes(model,useAT, genTarget[0])

            #Add random condition element
            if len(self.specifiedAttList) >= len(state):
                pass
            else:
                if not model.doExpertKnowledge or random.random() > pressureProb:
                    pickList = list(range(model.env.formatData.numAttributes))
                    for i in self.specifiedAttList:
                        pickList.remove(i)
                    specTarget = random.sample(pickList,1)
                else:
                    specTarget = self.selectSpecifyRW(model,1)

                if state[specTarget[0]] != None and (not useAT or random.random() < model.AT.getTrackProb()[specTarget[0]]):
                    self.specifiedAttList.append(specTarget[0])
                    self.condition.append(self.buildMatch(model,specTarget[0],state))  # buildMatch handles both discrete and continuous attributes
                    changed = True
                if len(self.specifiedAttList) > model.rule_specificity_limit:
                    self.specLimitFix(model,self)

        #Increase Specificity
        elif newRuleSpec > len(self.specifiedAttList): #Specify more attributes
            change = newRuleSpec - len(self.specifiedAttList)
            if not model.doExpertKnowledge or random.random() > pressureProb:
                pickList = list(range(model.env.formatData.numAttributes))
                for i in self.specifiedAttList: # Make list with all non-specified attributes
                    pickList.remove(i)
                specTarget = random.sample(pickList,change)
            else:
                specTarget = self.selectSpecifyRW(model,change)
            for j in specTarget:
                if state[j] != None and (not useAT or random.random() < model.AT.getTrackProb()[j]):
                    #Specify Target
                    self.specifiedAttList.append(j)
                    self.condition.append(self.buildMatch(model,j, state)) #buildMatch handles both discrete and continuous attributes
                    changed = True

        #Decrease Specificity
        elif newRuleSpec < len(self.specifiedAttList): # Generalize more attributes.
            change = len(self.specifiedAttList) - newRuleSpec
            if not model.doExpertKnowledge or random.random() > pressureProb:
                genTarget = random.sample(self.specifiedAttList,change)
            else:
                genTarget = self.selectGeneralizeRW(model,change)

            #-------------------------------------------------------
            # DISCRETE OR CONTINUOUS ATTRIBUTE - remove attribute specification with 50% chance if we have continuous attribute, or 100% if discrete attribute.
            #-------------------------------------------------------
            for j in genTarget:
                attributeInfoType = model.env.formatData.attributeInfoType[j]
                if not attributeInfoType or random.random() > 0.5: #GEN/SPEC OPTION
                    if not useAT or random.random() > model.AT.getTrackProb()[j]:
                        i = self.specifiedAttList.index(j) #reference to the position of the attribute in the rule representation
                        self.specifiedAttList.remove(j)
                        self.condition.pop(i) #buildMatch handles both discrete and continuous attributes
                        changed = True
                else:
                    self.mutateContinuousAttributes(model,useAT,j)
                    
        #-------------------------------------------------------
        # MUTATE PHENOTYPE
        #-------------------------------------------------------
        nowChanged = self.continuousEventMutation(eventTime) #NOTE: Must mutate to still include true current value.
        if changed:# or nowChanged:
            return True

    
#----------------------------------------------------------------------------------------------------------------------------
# continuousEventMutation: Mutate this rule's continuous phenotype
#----------------------------------------------------------------------------------------------------------------------------     
    
    def continuousEventMutation(self, eventTime):
        """ Mutate this rule's continuous eventTime. """
        #Continuous Phenotype Crossover------------------------------------
        changed = False
        if random.random() < cons.upsilon: #Mutate continuous phenotype
            eventRange = self.eventInterval[1] - self.phenotype[0]
            mutateRange = random.random()*0.5*eventRange
            tempKey = random.randint(0,2) #Make random choice between 3 scenarios, mutate minimums, mutate maximums, mutate both
            if tempKey == 0: #Mutate minimum 
                if random.random() > 0.5 or self.eventInterval[0] + mutateRange <= eventTime: #Checks that mutated range still contains current phenotype
                    self.eventInterval[0] += mutateRange
                else: #Subtract
                    self.eventInterval[0] -= mutateRange
                changed = True
            elif tempKey == 1: #Mutate maximum
                if random.random() > 0.5 or self.eventInterval[1] - mutateRange >= eventTime: #Checks that mutated range still contains current phenotype
                    self.eventInterval[1] -= mutateRange
                else: #Subtract
                    self.eventInterval[1] += mutateRange
                changed = True
            else: #mutate both
                if random.random() > 0.5 or self.eventInterval[0] + mutateRange <= eventTime: #Checks that mutated range still contains current phenotype
                    self.eventInterval[0] += mutateRange
                else: #Subtract
                    self.eventInterval[0] -= mutateRange
                if random.random() > 0.5 or self.eventInterval[1] - mutateRange >= eventTime: #Checks that mutated range still contains current phenotype
                    self.eventInterval[1] -= mutateRange
                else: #Subtract
                    self.eventInterval[1] += mutateRange
                changed = True
            
            #Repair range - such that min specified first, and max second.
            self.eventInterval.sort()
            #---------------------------------------------------------------------
        return changed 
    
    
#----------------------------------------------------------------------------------------------------------------------------
# selectGeneralizeRW: EK applied to the selection of an attribute to generalize for mutation.
#---------------------------------------------------------------------------------------------------------------------------- 
    def selectGeneralizeRW(self,model,count):
        probList = []
        for attribute in self.specifiedAttList:
            probList.append(1/model.EK.scores[attribute])
        if sum(probList) == 0:
            probList = (np.array(probList) + 1).tolist()

        probList = np.array(probList)/sum(probList) #normalize
        return np.random.choice(self.specifiedAttList,count,replace=False,p=probList).tolist()

    # def selectGeneralizeRW(self,model,count):
    #     EKScoreSum = 0
    #     selectList = []
    #     currentCount = 0
    #     specAttList = copy.deepcopy(self.specifiedAttList)
    #     for i in self.specifiedAttList:
    #         # When generalizing, EK is inversely proportional to selection probability
    #         EKScoreSum += 1 / float(model.EK.scores[i] + 1)
    #
    #     while currentCount < count:
    #         choicePoint = random.random() * EKScoreSum
    #         i = 0
    #         sumScore = 1 / float(model.EK.scores[specAttList[i]] + 1)
    #         while choicePoint > sumScore:
    #             i = i + 1
    #             sumScore += 1 / float(model.EK.scores[specAttList[i]] + 1)
    #         selectList.append(specAttList[i])
    #         EKScoreSum -= 1 / float(model.EK.scores[specAttList[i]] + 1)
    #         specAttList.pop(i)
    #         currentCount += 1
    #     return selectList
#----------------------------------------------------------------------------------------------------------------------------
# selectSpecifyRW: EK applied to the selection of an attribute to specify for mutation.
#---------------------------------------------------------------------------------------------------------------------------- 
    def selectSpecifyRW(self,model,count):
        pickList = list(range(model.env.formatData.numAttributes))
        for i in self.specifiedAttList:  # Make list with all non-specified attributes
            pickList.remove(i)

        probList = []
        for attribute in pickList:
            probList.append(model.EK.scores[attribute])
        if sum(probList) == 0:
            probList = (np.array(probList) + 1).tolist()
        probList = np.array(probList) / sum(probList)  # normalize
        return np.random.choice(pickList, count, replace=False, p=probList).tolist()

    # def selectSpecifyRW(self, model,count):
    #     """ EK applied to the selection of an attribute to specify for mutation. """
    #     pickList = list(range(model.env.formatData.numAttributes))
    #     for i in self.specifiedAttList:  # Make list with all non-specified attributes
    #         pickList.remove(i)
    #
    #     EKScoreSum = 0
    #     selectList = []
    #     currentCount = 0
    #
    #     for i in pickList:
    #         # When generalizing, EK is inversely proportional to selection probability
    #         EKScoreSum += model.EK.scores[i]
    #
    #     while currentCount < count:
    #         choicePoint = random.random() * EKScoreSum
    #         i = 0
    #         sumScore = model.EK.scores[pickList[i]]
    #         while choicePoint > sumScore:
    #             i = i + 1
    #             sumScore += model.EK.scores[pickList[i]]
    #         selectList.append(pickList[i])
    #         EKScoreSum -= model.EK.scores[pickList[i]]
    #         pickList.pop(i)
    #         currentCount += 1
    #     return selectList
#----------------------------------------------------------------------------------------------------------------------------
# mutateContinuousAttributes: XXX
#---------------------------------------------------------------------------------------------------------------------------- 
    def mutateContinuousAttributes(self, model,useAT, j):
        # -------------------------------------------------------
        # MUTATE CONTINUOUS ATTRIBUTES
        # -------------------------------------------------------
        if useAT:
            if random.random() < model.AT.getTrackProb()[j]:  # High AT probability leads to higher chance of mutation (Dives ExSTraCS to explore new continuous ranges for important attributes)
                # Mutate continuous range - based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, of size between 0 and 50% of att domain.
                attRange = float(model.env.formatData.attributeInfoContinuous[j][1]) - float(model.env.formatData.attributeInfoContinuous[j][0])
                i = self.specifiedAttList.index(j)  # reference to the position of the attribute in the rule representation
                mutateRange = random.random() * 0.5 * attRange
                if random.random() > 0.5:  # Mutate minimum
                    if random.random() > 0.5:  # Add
                        self.condition[i][0] += mutateRange
                    else:  # Subtract
                        self.condition[i][0] -= mutateRange
                else:  # Mutate maximum
                    if random.random() > 0.5:  # Add
                        self.condition[i][1] += mutateRange
                    else:  # Subtract
                        self.condition[i][1] -= mutateRange
                # Repair range - such that min specified first, and max second.
                self.condition[i].sort()
                changed = True
        elif random.random() > 0.5:
            # Mutate continuous range - based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, of size between 0 and 50% of att domain.
            attRange = float(model.env.formatData.attributeInfoContinuous[j][1]) - float(model.env.formatData.attributeInfoContinuous[j][0])
            i = self.specifiedAttList.index(j)  # reference to the position of the attribute in the rule representation
            mutateRange = random.random() * 0.5 * attRange
            if random.random() > 0.5:  # Mutate minimum
                if random.random() > 0.5:  # Add
                    self.condition[i][0] += mutateRange
                else:  # Subtract
                    self.condition[i][0] -= mutateRange
            else:  # Mutate maximum
                if random.random() > 0.5:  # Add
                    self.condition[i][1] += mutateRange
                else:  # Subtract
                    self.condition[i][1] -= mutateRange
            # Repair range - such that min specified first, and max second.
            self.condition[i].sort()
            changed = True
        else:
            pass

#----------------------------------------------------------------------------------------------------------------------------
# rangeCheck: Checks and prevents the scenario where a continuous attributes specified in a rule has a range that fully encloses the training set range for that attribute.
#---------------------------------------------------------------------------------------------------------------------------- 
    def rangeCheck(self,model):
        """ Checks and prevents the scenario where a continuous attributes specified in a rule has a range that fully encloses the training set range for that attribute."""
        for attRef in self.specifiedAttList:
            if model.env.formatData.attributeInfoType[attRef]: #Attribute is Continuous
                trueMin = model.env.formatData.attributeInfoContinuous[attRef][0]
                trueMax = model.env.formatData.attributeInfoContinuous[attRef][1]
                i = self.specifiedAttList.index(attRef)
                valBuffer = (trueMax-trueMin)*0.1
                if self.condition[i][0] <= trueMin and self.condition[i][1] >= trueMax: # Rule range encloses entire training range
                    self.specifiedAttList.remove(attRef)
                    self.condition.pop(i)
                    return
                elif self.condition[i][0]+valBuffer < trueMin:
                    self.condition[i][0] = trueMin - valBuffer
                elif self.condition[i][1]- valBuffer > trueMax:
                    self.condition[i][1] = trueMin + valBuffer
                else:
                    pass
#----------------------------------------------------------------------------------------------------------------------------
# getDelProp:  Returns the vote for deletion of the classifier.
#---------------------------------------------------------------------------------------------------------------------------- 
    def getDelProp(self, model, meanFitness):
        """  Returns the vote for deletion of the classifier. """
        if self.fitness / self.numerosity >= model.delta * meanFitness or self.matchCount < model.theta_del:
            deletionVote = self.aveMatchSetSize * self.numerosity
        elif self.fitness == 0.0:
            deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (model.init_fitness / self.numerosity)
        else:
            deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (self.fitness / self.numerosity)
        return deletionVote
