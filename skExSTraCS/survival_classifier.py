import random
import copy
import numpy as np

class Classifier: #this script is for an INDIVIDUAL CLASSIFIER
    def __init__(self,model):
        self.specifiedAttList = []
        self.condition = []
        self.phenotype = None
        
        ##addition for survival analysis
        self.EvalTime = 0 #what should the initial value be? When will it get re-assigned? When drawing an instance also need to choose a ET at which to evaluate the instance...this will become ET

        self.fitness = model.init_fitness
        self.accuracy = 0
        self.numerosity = 1
        self.aveMatchSetSize = None
        self.deletionProb = None

        self.timeStampGA = None
        self.initTimeStamp = None
        self.epochComplete = False

        self.matchCount = 0
        self.correctCount = 0
        self.matchCover = 0 #what are these? 
        self.correctCover = 0

    def initializeByCopy(self,toCopy,iterationCount): #idk what this means 
        self.specifiedAttList = copy.deepcopy(toCopy.specifiedAttList)
        self.condition = copy.deepcopy(toCopy.condition)
        self.phenotype = copy.deepcopy(toCopy.phenotype)
        self.timeStampGA = iterationCount
        self.initTimeStamp = iterationCount
        self.aveMatchSetSize = copy.deepcopy(toCopy.aveMatchSetSize)
        self.fitness = toCopy.fitness
        self.accuracy = toCopy.accuracy

    def initializeByCovering(self,model,setSize,state,phenotype): 
        self.timeStampGA = model.iterationCount #the timestamp is set to what iteration we're on
        self.initTimeStamp = model.iterationCount #same 
        self.aveMatchSetSize = setSize #zero to start?
        self.phenotype = phenotype #this will have to change 
        self.EvalTime = EvalTime #will need to set the evaluation time for covering , will need to choose this somewhere in the data_management script

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

    def buildMatch(self,model,attRef,state): 
        attributeInfoType = model.env.formatData.attributeInfoType[attRef] #set the type of attribute (discrete/continuous)
        if not (attributeInfoType):  # Discrete
            attributeInfoValue = model.env.formatData.attributeInfoDiscrete[attRef]
        else: #continuous
            attributeInfoValue = model.env.formatData.attributeInfoContinuous[attRef]

        if attributeInfoType: #Continuous Attribute
            attRange = attributeInfoValue[1] - attributeInfoValue[0]
            rangeRadius = random.randint(25, 75) * 0.01 * attRange / 2.0  # initialize a continuous domain radius.
            Low = state[attRef] - rangeRadius
            High = state[attRef] + rangeRadius
            condList = [Low, High] #Condition list is the range of values from low to high
        else:
            condList = state[attRef] #for a discrete attribute, the condition list is simply its state
        return condList

    def updateEpochStatus(self,model): #has the model iterated enough times? if not, keep going. If true, set epochComplete = True
        if not self.epochComplete and (model.iterationCount - self.initTimeStamp - 1) >= model.env.formatData.numTrainInstances:
            self.epochComplete = True

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

    def equals(self,cl): #for each rule, checks to see if the other rules are the sam
        if cl.phenotype == self.phenotype and len(cl.specifiedAttList) == len(self.specifiedAttList): #if the phenotypes are the same and the list of attributes are the same length, check the following...
            clRefs = sorted(cl.specifiedAttList) #sort the attribute indexes for the classifier
            selfRefs = sorted(self.specifiedAttList) #sort the attribute indexes
            if clRefs == selfRefs: #if they are the same, then...
                for i in range(len(cl.specifiedAttList)): #for each attribute in the classifier 
                    tempIndex = self.specifiedAttList.index(cl.specifiedAttList[i]) #the following checks to see if the conditions also match 
                    if not (cl.condition[i] == self.condition[tempIndex]):
                        return False #if the don't all match, return false
                return True #if they do all match, return true (I assume somewhere this would update the numerosity)
        return False #if the phenotypes and lengths of the specified attribute lists of two classifiers don't match, return false
    
    ## All the rest of this stuff would probably stay exactly the same 

    def updateExperience(self): #add 1 to either the matchcount or the matchcover, depending on what was needed 
        self.matchCount += 1
        if self.epochComplete:  # Once epoch Completed, number of matches for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.matchCover += 1

    def updateMatchSetSize(self, model,matchSetSize):  #update the match set size. "beta" is set at 0.2, a learning parameter used in calculating the average correct set size
        if self.matchCount < 1.0 / model.beta: # if the match count is less than 5
            self.aveMatchSetSize = (self.aveMatchSetSize * (self.matchCount-1)+ matchSetSize) / float(self.matchCount) #update the average to...
        else: #if its not less than 5,
            self.aveMatchSetSize = self.aveMatchSetSize + model.beta * (matchSetSize - self.aveMatchSetSize)

    def updateCorrect(self):
        self.correctCount += 1
        if self.epochComplete: #Once epoch Completed, number of correct for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.correctCover += 1

    def updateAccuracy(self):
        self.accuracy = self.correctCount / float(self.matchCount)

    def updateFitness(self,model):
        self.fitness = pow(self.accuracy, model.nu)

    def updateNumerosity(self, num):
        """ Alters the numerosity of the classifier.  Notice that num can be negative! """
        self.numerosity += num #but where does num come from??

    def isSubsumer(self, model): #is the match count and accuracy of a more general rule just as good as the more specific one? If so,  return true
        if self.matchCount > model.theta_sub and self.accuracy > model.acc_sub: #if the match count is greater than the theta_sub (subsumption experience threshold, default = 20) and the accuracy is greater than the acc_sub (default = 0.99), return true
            return True
        return False

    def subsumes(self,model,cl): 
        return cl.phenotype == self.phenotype and self.isSubsumer(model) and self.isMoreGeneral(model,cl)

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

    def updateTimeStamp(self, ts): #where does ts come from?
        """ Sets the time stamp of the classifier. """
        self.timeStampGA = ts

    def uniformCrossover(self,model,cl): #define the uniform crossover function 
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
            if useAT: #if AT is true (default is)
                probability = model.AT.getTrackProb()[attRef] #set the probability (see attribute_tracking.py)
            else:
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

    def setAccuracy(self, acc):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc

    def setFitness(self, fit):
        """  Sets the fitness of the classifier. """
        self.fitness = fit

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
        newRuleSpec = random.randint(lowLim, highLim)

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

        return changed

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

    def getDelProp(self, model, meanFitness):
        """  Returns the vote for deletion of the classifier. """
        if self.fitness / self.numerosity >= model.delta * meanFitness or self.matchCount < model.theta_del:
            deletionVote = self.aveMatchSetSize * self.numerosity
        elif self.fitness == 0.0:
            deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (model.init_fitness / self.numerosity)
        else:
            deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (self.fitness / self.numerosity)
        return deletionVote
