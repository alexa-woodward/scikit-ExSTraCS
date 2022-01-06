from skExSTraCS.survival_Classifier import Classifier
import copy
import random

class ClassifierSet:
    def __init__(self):
        self.popSet = []  # List of classifiers/rules
        self.matchSet = []  # List of references to rules in population that match
        self.correctSet = []  # List of references to rules in population that both match and specify correct phenotype
        self.microPopSize = 0
        self.aveEventRange = 0.0
        self.nextID = 0
        
#--------------------------------------------------------------------------------------------------------------------
# makeMatchSet: Constructs a match set from the population. Covering is initiated if the match set is empty or a rule with the current correct eventRange is absent.
#--------------------------------------------------------------------------------------------------------------------
    def makeMatchSet(self,model,state_event): 
        state = state_event[0]
        eventTime = state_event[1]
        doCovering = True
        setNumerositySum = 0

        model.timer.startTimeMatching()
        for i in range(len(self.popSet)): #for each rule in the population 
            cl = self.popSet[i] #cl is rule/classifer at index i
#            cl.updateEpochStatus(model) #DO WE NEED THIS HERE? update epochStatus, if not already epoch complete and the number of iterations (minus the rule's timestamp) is greater than the number of training instances, set epochComplete = True 

            if cl.match(model,state): #if the rule is a match, append it to the match set 
                self.matchSet.append(i)
                setNumerositySum += cl.numerosity #and increase its numerosity 

                if float(cl.eventInterval[0]) <= float(eventTime) <= float(cl.eventInterval[1]):   # Check that event time is within the rule event interval, if so, set covering = false
                        doCovering = False 

        model.timer.stopTimeMatching()

        model.timer.startTimeCovering()
        while doCovering: #if covering is still true...
            newCl = Classifier(model) #create a new classifier (call from survival_classifier.py script, runs the whole thing...which includes the "evaluateAccuracyandInitalFitness" function, called below before the classifier is added to popSet)
            newCl.initializeByCovering(model,setNumerositySum+1,state,eventTime)
            if len(newCl.specifiedAttList) > 0: #ADDED CHECK TO PREVENT FULLY GENERALIZED RULES
                self.addClassifierToPopulation(model,newCl,True)
                self.matchSet.append(len(self.popSet)-1)
                model.trackingObj.coveringCount += 1
                doCovering = False
        model.timer.stopTimeCovering()
        
#--------------------------------------------------------------------------------------------------------------------
# addClassifierToPopulation: Adds a classifier to the set and increases the numerositySum value accordingly.
#--------------------------------------------------------------------------------------------------------------------        
    def addClassifierToPopulation(self,model,cl,covering):
        model.timer.startTimeAdd()
        oldCl = None
        if not covering:
            oldCl = self.getIdenticalClassifier(cl)
        if oldCl != None:
            oldCl.updateNumerosity(1)
            self.microPopSize += 1
        else:
            cl.evaluateAccuracyAndInitialFitness(model,self.nextID) #this nextID thing might be an issue 
            self.popSet[self.nextID] = cl
            self.nextID += 1
            self.microPopSize += 1
            #self.popSet.append(cl) old, new @ line #60 above, to include param "nextID"
            self.microPopSize += 1
        model.timer.stopTimeAdd()
        
#--------------------------------------------------------------------------------------------------------------------
# getIdenticalClassifier: Looks for an identical classifier in the population.
#--------------------------------------------------------------------------------------------------------------------    
    def getIdenticalClassifier(self,newCl):
        for cl in self.popSet:
            if newCl.equals(cl):
                return cl
        return None

#--------------------------------------------------------------------------------------------------------------------
# makeCorrectSet: Constructs a correct set out of the given match set. 
#--------------------------------------------------------------------------------------------------------------------    
    def makeCorrectSet(self,eventStatus,eventTime): #If the eventTime is within the rule eventInterval, append to the correct set.
        for i in range(len(self.matchSet)):
            ref = self.matchSet[i]
            if eventStatus = 1:
                if float(eventTime) <= float(self.popSet[ref].eventInterval[1]) and float(eventTime) >= float(self.popSet[ref].eventInterval[0]):
                        self.correctSet.append(ref)
            else: #if the instance was censored, append to the correct set IF the interval includes the censoring time or the interval is BEYOND the censoring time
                if (float(eventTime) <= float(self.popSet[ref].eventInterval[1]) and float(eventTime) >= float(self.popSet[ref].eventInterval[0]) or (float(eventTime) < float(self.popSet[ref].eventInterval[0]):
                        self.correctSet.append(ref)                                                                                                                              
#--------------------------------------------------------------------------------------------------------------------
# updateSets: Updates all relevant parameters in the current match and correct sets.
#--------------------------------------------------------------------------------------------------------------------    
    def updateSets(self,model,eventTime,eventStatus):
        matchSetNumerosity = 0
        for ref in self.matchSet:
            matchSetNumerosity += self.popSet[ref].numerosity

        for ref in self.matchSet:
            self.popSet[ref].updateExperience() #this can go away, I think, but need to preserve a way to increase the matchCover
            self.popSet[ref].updateMatchSetSize(model,matchSetNumerosity)  # Moved to match set to be like GHCS
            if ref in self.correctSet: #if the rule is in the correct set, update the correctCoverage. NOTE error and accuracy ARENT updated here, because the rules should have already seen all the instances (in evaluateAccuracyandInitialFitness)
               self.popSet[ref].updateCorrect()  
            self.popSet[ref].updateFitness(model) #update fitness
            
#--------------------------------------------------------------------------------------------------------------------
# do_correct_set_subsumption: XXX
#--------------------------------------------------------------------------------------------------------------------    
    def do_correct_set_subsumption(self,model):
        subsumer = None
        for ref in self.correctSet:
            cl = self.popSet[ref]
            if cl.isSubsumer(model):
                if subsumer == None or cl.isMoreGeneral(model,subsumer):
                    subsumer = cl

        if subsumer != None:
            i = 0
            while i < len(self.correctSet):
                ref = self.correctSet[i]
                if subsumer.isMoreGeneral(model,self.popSet[ref]):
                    model.trackingObj.subsumptionCount += 1
                    model.trackingObj.subsumptionCount += 1
                    subsumer.updateNumerosity(self.popSet[ref].numerosity)
                    self.removeMacroClassifier(ref)
                    self.deleteFromMatchSet(ref)
                    self.deleteFromCorrectSet(ref)
                    i -= 1
                i+=1
                
#--------------------------------------------------------------------------------------------------------------------
# removeMacroClassifier: Removes the specified (macro-) classifier from the population.
#--------------------------------------------------------------------------------------------------------------------    
    def removeMacroClassifier(self, ref):
        self.popSet.pop(ref)

#--------------------------------------------------------------------------------------------------------------------
# deleteFromMatchSet: Delete reference to classifier in population, contained in self.matchSet
#--------------------------------------------------------------------------------------------------------------------    
    def deleteFromMatchSet(self, deleteRef):
        if deleteRef in self.matchSet:
            self.matchSet.remove(deleteRef)
        # Update match set reference list--------
        for j in range(len(self.matchSet)):
            ref = self.matchSet[j]
            if ref > deleteRef:
                self.matchSet[j] -= 1
                
#--------------------------------------------------------------------------------------------------------------------
# deleteFromCorrectSet: Delete reference to classifier in population, contained in self.correctSet
#--------------------------------------------------------------------------------------------------------------------    
    def deleteFromCorrectSet(self, deleteRef):
        """ Delete reference to classifier in population, contained in self.matchSet."""
        if deleteRef in self.correctSet:
            self.correctSet.remove(deleteRef)
        # Update match set reference list--------
        for j in range(len(self.correctSet)):
            ref = self.correctSet[j]
            if ref > deleteRef:
                self.correctSet[j] -= 1

#--------------------------------------------------------------------------------------------------------------------
# runGA: The genetic discovery mechanism in ExSTraCS is controlled here. 
#--------------------------------------------------------------------------------------------------------------------    
    def runGA(self,model,state,eventTime):
        if model.iterationCount - self.getIterStampAverage() < model.theta_GA:
            return
        self.setIterStamps(model.iterationCount)

        changed = False

        #Select Parents
        model.timer.startTimeSelection()
        if model.selection_method == "roulette":
            selectList = self.selectClassifierRW()
            clP1 = selectList[0]
            clP2 = selectList[1]
        elif model.selection_method == "tournament":
            selectList = self.selectClassifierT(model)
            clP1 = selectList[0]
            clP2 = selectList[1]
        model.timer.stopTimeSelection()

        #-------------------------------------------------------
        # INITIALIZE OFFSPRING 
        #-------------------------------------------------------
        cl1 = Classifier(model)
        cl1.initializeByCopy(clP1,model.iterationCount)
        cl2 = Classifier(model)
        if clP2 == None:
            cl2.initializeByCopy(clP1,model.iterationCount)
        else:
            cl2.initializeByCopy(clP2,model.iterationCount)

        #-------------------------------------------------------
        # CROSSOVER OPERATOR - Uniform Crossover Implemented (i.e. all attributes have equal probability of crossing over between two parents)
        #-----------------------------------------------------
        if not cl1.equals(cl2) and random.random() < model.chi:
            model.timer.startTimeCrossover()
            changed = cl1.uniformCrossover(model,cl2)
            model.timer.stopTimeCrossover()

        if changed:
            cl1.setAccuracy((cl1.accuracy + cl2.accuracy)/2.0)
            cl1.setFitness(model.fitness_reduction * (cl1.fitness + cl2.fitness)/2.0)
            cl2.setAccuracy(cl1.accuracy)
            cl2.setFitness(cl1.fitness)
        else:
            cl1.setFitness(model.fitness_reduction * cl1.fitness)
            cl2.setFitness(model.fitness_reduction * cl2.fitness)

        #-------------------------------------------------------
        # MUTATION OPERATOR 
        #-------------------------------------------------------
        model.timer.startTimeMutation()
        nowchanged = cl1.mutation(model,state,eventTime)
        howaboutnow = cl2.mutation(model,state,eventTime)
        model.timer.stopTimeMutation()

        if model.env.formatData.continuousCount > 0:
            cl1.rangeCheck(model)
            cl2.rangeCheck(model)
                                                                                                                                                      
        #-------------------------------------------------------
        # Event range probability correction
        #-------------------------------------------------------                                                                                                                                                                                                                                                                                           
        cl1.setEventProb()
        cl2.setEventProb()                                                                                                                                              

        if changed or nowchanged or howaboutnow:
            if nowchanged:
                model.trackingObj.mutationCount += 1
            if howaboutnow:
                model.trackingObj.mutationCount += 1
            if changed:
                model.trackingObj.crossOverCount += 1
            self.insertDiscoveredClassifiers(model,cl1, cl2, clP1, clP2) #Includes subsumption if activated.

#--------------------------------------------------------------------------------------------------------------------
# insertDiscoveredClassifiers: Inserts both discovered classifiers keeping the maximal size of the population and possibly doing GA subsumption. 
       # Checks for default rule (i.e. rule with completely general condition) prevents such rules from being added to the population.
#--------------------------------------------------------------------------------------------------------------------    
    def insertDiscoveredClassifiers(self,model,cl1,cl2,clP1,clP2):
        if model.do_GA_subsumption:
            model.timer.startTimeSubsumption()
            if len(cl1.specifiedAttList) > 0:
                self.subsumeClassifier(model,cl1, clP1, clP2)
            if len(cl2.specifiedAttList) > 0:
                self.subsumeClassifier(model,cl2, clP1, clP2)
            model.timer.stopTimeSubsumption()
        else:
            if len(cl1.specifiedAttList) > 0:
                self.addClassifierToPopulation(model,cl1,False)
            if len(cl2.specifiedAttList) > 0:
                self.addClassifierToPopulation(model,cl2,False)

#--------------------------------------------------------------------------------------------------------------------
# subsumeClassifier: Tries to subsume a classifier in the parents. If no subsumption is possible it tries to subsume it in the current set.
#--------------------------------------------------------------------------------------------------------------------               
    def subsumeClassifier(self, model,cl, cl1P, cl2P):
        """ Tries to subsume a classifier in the parents. If no subsumption is possible it tries to subsume it in the current set. """
        if cl1P!=None and cl1P.subsumes(model,cl): #called from classifier.py
            self.microPopSize += 1
            cl1P.updateNumerosity(1)
            model.trackingObj.subsumptionCount+=1
        elif cl2P!=None and cl2P.subsumes(model,cl):
            self.microPopSize += 1
            cl2P.updateNumerosity(1)
            model.trackingObj.subsumptionCount += 1
        else:
            if len(cl.specifiedAttList) > 0:
                self.addClassifierToPopulation(model, cl, False)

#--------------------------------------------------------------------------------------------------------------------
# selectClassifierRW: Selects parents using roulette wheel selection according to the fitness of the classifiers.
#--------------------------------------------------------------------------------------------------------------------
    def selectClassifierRW(self):
        setList = copy.deepcopy(self.correctSet)

        if len(setList) > 2:
            selectList = [None,None]
            currentCount = 0

            while currentCount < 2:
                fitSum = self.getFitnessSum(setList)

                choiceP = random.random() * fitSum
                i = 0
                sumCl = self.popSet[setList[i]].fitness
                while choiceP > sumCl:
                    i = i + 1
                    sumCl += self.popSet[setList[i]].fitness

                selectList[currentCount] = self.popSet[setList[i]]
                setList.remove(setList[i])
                currentCount += 1

        elif len(setList) == 2:
            selectList = [self.popSet[setList[0]], self.popSet[setList[1]]]
        elif len(setList) == 1:
            selectList = [self.popSet[setList[0]], self.popSet[setList[0]]]
        else:
            print("ClassifierSet: Error in parent selection.")
            
        return selectList

#--------------------------------------------------------------------------------------------------------------------
# getFitnessSum: Returns the sum of the fitnesses of all classifiers in the set
#--------------------------------------------------------------------------------------------------------------------
    def getFitnessSum(self, setList):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sumCl = 0.0
        for i in range(len(setList)):
            ref = setList[i]
            sumCl += self.popSet[ref].fitness
        return sumCl

#--------------------------------------------------------------------------------------------------------------------
# selectClassifierT: Selects parents using tournament selection according to the fitness of the classifiers
#--------------------------------------------------------------------------------------------------------------------
    def selectClassifierT(self,model):
        selectList = [None, None]
        currentCount = 0
        setList = self.correctSet

        while currentCount < 2:
            tSize = int(len(setList) * model.theta_sel)

            #Select tSize elements from correctSet
            posList = random.sample(setList,tSize)

            bestF = 0
            bestC = self.correctSet[0]
            for j in posList:
                if self.popSet[j].fitness > bestF:
                    bestF = self.popSet[j].fitness
                    bestC = j

            selectList[currentCount] = self.popSet[bestC]
            currentCount += 1

        return selectList
#--------------------------------------------------------------------------------------------------------------------
# getIterStampAverage: Returns the average of the time stamps in the correct set.
#--------------------------------------------------------------------------------------------------------------------
    def getIterStampAverage(self):
        """ Returns the average of the time stamps in the correct set. """
        sumCl=0
        numSum=0
        for i in range(len(self.correctSet)):
            ref = self.correctSet[i]
            sumCl += self.popSet[ref].timeStampGA * self.popSet[ref].numerosity
            numSum += self.popSet[ref].numerosity #numerosity sum of correct set
        if numSum != 0:
            return sumCl / float(numSum)
        else:
            return 0

#--------------------------------------------------------------------------------------------------------------------
# getInitStampAverage: Returns the averate inital time stamp of the classifiers 
#--------------------------------------------------------------------------------------------------------------------
    def getInitStampAverage(self):
        sumCl = 0.0
        numSum = 0.0
        for i in range(len(self.correctSet)):
            ref = self.correctSet[i]
            sumCl += self.popSet[ref].initTimeStamp * self.popSet[ref].numerosity
            numSum += self.popSet[ref].numerosity
        if numSum != 0:
            return sumCl / float(numSum)
        else:
            return 0

#--------------------------------------------------------------------------------------------------------------------
# setIterStamps: Sets the time stamp of all classifiers in the set to the current time. The current time is the number of exploration steps executed so far.
#--------------------------------------------------------------------------------------------------------------------        
    def setIterStamps(self, iterationCount):
        """ Sets the time stamp of all classifiers in the set to the current time. The current time is the number of exploration steps executed so far.  """
        for i in range(len(self.correctSet)):
            ref = self.correctSet[i]
            self.popSet[ref].updateTimeStamp(iterationCount)

#--------------------------------------------------------------------------------------------------------------------
# deletion: Returns the population size back to the maximum set by the user by deleting rules. 
#--------------------------------------------------------------------------------------------------------------------            
    def deletion(self,model):
        model.timer.startTimeDeletion()
        while self.microPopSize > model.N:
            self.deleteFromPopulation(model)
        model.timer.stopTimeDeletion()

#--------------------------------------------------------------------------------------------------------------------
# deleteFromPopulation: Deletes one classifier in the population.  The classifier that will be deleted is chosen by roulette wheel selection considering the deletion vote. Returns the macro-classifier which got decreased by one micro-classifier. 
#--------------------------------------------------------------------------------------------------------------------
    def deleteFromPopulation(self,model):
        meanFitness = self.getPopFitnessSum() / float(self.microPopSize)

        sumCl = 0.0
        voteList = []
        for cl in self.popSet:
            vote = cl.getDelProp(model,meanFitness)
            sumCl += vote
            voteList.append(vote)

        i = 0
        for cl in self.popSet:
            cl.deletionProb = voteList[i] / sumCl
            i += 1

        choicePoint = sumCl * random.random()  # Determine the choice point

        newSum = 0.0
        for i in range(len(voteList)):
            cl = self.popSet[i]
            newSum = newSum + voteList[i]
            if newSum > choicePoint:  # Select classifier for deletion
                # Delete classifier----------------------------------
                cl.updateNumerosity(-1)
                self.microPopSize -= 1
                if cl.numerosity < 1:  # When all micro-classifiers for a given classifier have been depleted.
                    self.removeMacroClassifier(i)
                    self.deleteFromMatchSet(i)
                    self.deleteFromCorrectSet(i)
                    model.trackingObj.deletionCount += 1
                return

#--------------------------------------------------------------------------------------------------------------------
# getPopFitnessSum: Returns the sum of the fitnesses of all classifiers in the set.
#--------------------------------------------------------------------------------------------------------------------            
    def getPopFitnessSum(self):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sumCl=0.0
        for cl in self.popSet:
            sumCl += cl.fitness *cl.numerosity
        return sumCl

#--------------------------------------------------------------------------------------------------------------------
# clearSets: Clears out references in the match and correct sets for the next learning iteration.
#--------------------------------------------------------------------------------------------------------------------
    def clearSets(self):
        """ Clears out references in the match and correct sets for the next learning iteration. """
        self.matchSet = []
        self.correctSet = []
        
#--------------------------------------------------------------------------------------------------------------------
# getAveGenerality: determine the average generality of rules in the population (# of attributes specified in each rule)
#--------------------------------------------------------------------------------------------------------------------
    def getAveGenerality(self,model):
        genSum = 0
        sumRuleRange = 0
        for cl in self.popSet:
            genSum += ((model.env.formatData.numAttributes - len(cl.condition))/float(model.env.formatData.numAttributes))*cl.numerosity
        if self.microPopSize == 0:
            aveGenerality = 0
        else:
            aveGenerality = genSum/float(self.microPopSize)
        for cl in self.popSet:
                high = cl.eventInterval[1]
                low = cl.eventInterval[0]
                if high > cons.env.formatData.eventList[1]:
                    high = cons.env.formatData.eventList[1]
                if low < cons.env.formatData.eventList[0]:
                    low = cons.env.formatData.eventList[0]
                sumRuleRange += (cl.eventInterval[1] - cl.eventInterval[0])*cl.numerosity
            eventRange = cons.env.formatData.eventList[1] - cons.env.formatData.eventList[0]
            self.aveEventRange = (sumRuleRange / float(self.microPopSize)) / float(eventRange)       
        return aveGenerality
        
    
#--------------------------------------------------------------------------------------------------------------------
# makeEvalMatchSet: Constructs a match set for evaluation purposes which does not activate either covering or deletion.
#--------------------------------------------------------------------------------------------------------------------
    def makeEvalMatchSet(self,model,state):
        for i in range(len(self.popSet)):
            cl = self.popSet[i]
            if cl.match(model,state):
                self.matchSet.append(i)
                
#--------------------------------------------------------------------------------------------------------------------
#  getAttributeSpecificityList: Determine the population-wide frequency of attribute specification
#--------------------------------------------------------------------------------------------------------------------
    def getAttributeSpecificityList(self,model):
        attributeSpecList = []
        for i in range(model.env.formatData.numAttributes):
            attributeSpecList.append(0)
        for cl in self.popSet:
            for ref in cl.specifiedAttList:
                attributeSpecList[ref] += cl.numerosity
        return attributeSpecList
    
#--------------------------------------------------------------------------------------------------------------------
# getAttributeAccuracyList: Get accuracy weighted specification
#--------------------------------------------------------------------------------------------------------------------
    def getAttributeAccuracyList(self,model):
        attributeAccList = []
        for i in range(model.env.formatData.numAttributes):
            attributeAccList.append(0.0)
        for cl in self.popSet:
            for ref in cl.specifiedAttList:
                attributeAccList[ref] += cl.numerosity * cl.accuracy
        return attributeAccList
