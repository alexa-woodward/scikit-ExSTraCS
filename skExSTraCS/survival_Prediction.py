"""
Name:        ExSTraCS_Prediction.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Modified:    August 25,2014
Description: Based on a given match set, this module uses a voting scheme to select the phenotype prediction for ExSTraCS.
             
---------------------------------------------------------------------------------------------------------------------------------------------------------
ExSTraCS V2.0: Extended Supervised Tracking and Classifying System - An advanced LCS designed specifically for complex, noisy classification/data mining tasks, 
such as biomedical/bioinformatics/epidemiological problem domains.  This algorithm should be well suited to any supervised learning problem involving 
classification, prediction, data mining, and knowledge discovery.  This algorithm would NOT be suited to function approximation, behavioral modeling, 
or other multi-step problems.  This LCS algorithm is most closely based on the "UCS" algorithm, an LCS introduced by Ester Bernado-Mansilla and 
Josep Garrell-Guiu (2003) which in turn is based heavily on "XCS", an LCS introduced by Stewart Wilson (1995).  
Copyright (C) 2014 Ryan Urbanowicz 
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the 
Free Software Foundation; either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABLILITY 
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, 
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules-------------------------------
#from exstracs_constants import *
from survival_DataManagement import *
import random
import numpy as np
from sklearn.neighbors import KernelDensity

#------------------------------------------------------

class Prediction:
    def __init__(self,model,population):  #now takes in population ( have to reference the match set to do prediction)  pop.matchSet
        """ Constructs the voting array and determines the prediction decision. """
        self.decision = None
        self.survProb = None
        self.survProbDist = None
        self.times = None
        self.matchCoverTimes = [] #create an empty list to append coverTimes from the matchset rules
        
        #-------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        #-------------------------------------------------------
        #Grab unique upper and lower bounds from every rule in the match set.
        #Order list to discriminate pools
        #Go through M and add vote to each pool (i.e. fitness*numerosity)
        #Determine which (shortest) span has the largest vote.  
        #Quick version is to take the centroid of this 'best' span
        #OR - identify any M rules that cover whole 'best' span, and use centroid voting that includes only these rules. 
        if len(population.matchSet) < 1:
            self.decision = None
        else:
            segmentList = []
            for ref in population.matchSet:
                cl = population.popSet[ref] 
                high = cl.eventInterval[1]
                low = cl.eventInterval[0]
                if not high in segmentList:
                    segmentList.append(high)
                if not low in segmentList:
                    segmentList.append(low)
            segmentList.sort()
            voteList = []
            for i in range(0,len(segmentList)-1):
                voteList.append(0)
                #PART 2
            for ref in population.matchSet:
                cl = population.popSet[ref] 
                high = cl.eventInterval[1]
                low = cl.eventInterval[0]
                    #j = 0
                for j in range(len(segmentList)-1):
                    if low <= segmentList[j] and high >= segmentList[j+1]:
                        voteList[j] += cl.numerosity * cl.fitness
            #PART 3
            bestVote = max(voteList)
            bestRef = voteList.index(bestVote)
            bestlow = segmentList[bestRef]
            besthigh = segmentList[bestRef+1]
            centroid = (bestlow + besthigh) / 2.0
            self.decision = centroid
            
#-----------------------------------------------------------------------------------------------------------------
# individualSurvivalProb: generates the survival probability distribution for a test instance 
#----------------------------------------------------------------------------------------------------------------- 
#need to figure out where this gets called 

    def individualSurvivalProbDist(self,model,population):
        for ref in population.matchSet:
            cl = population.popSet[ref]
            if len(cl.coverTimes) > 0:
                self.matchCoverTimes.append(cl.coverTimes)
        empDist = np.asarray(sorted(self.matchCoverTimes)).reshape((len(self.matchCoverTimes), 1)) #sort the correct times, set as the empricial distribution
        KDEmodel = KernelDensity(bandwidth=4, kernel='epanechnikov')
        KDEmodel.fit(empDist) #fit the KDE to the empirical distribution
      
        self.times = np.asarray([time for time in range(0, model.env.formatData.eventList[1]+1)]).reshape((len(values), 1))
        probabilities = exp(KDEmodel.score_samples(self.times)) #generate probabilities from the fitted model for each time point
        self.survProbDist = 1 - np.cumsum(probabilities) #1-integral(pdf) = 1-CDF = survival probs!
      #  return self.survProbDist #is this needed?

      
      #1. get coverage from other matched instances. Is this stored anywhere already? If not need to make new function to store these. 
      #2. Use KDE to estimate the PDF
      #3. Use: survival = 1 - np.cumsum(probabilities) to retrieve the survival probabilities


#-----------------------------------------------------------------------------------------------------------------
# getFitnessSum: Get the fitness Sum of rules in the rule-set. For continuous phenotype prediction.
#-----------------------------------------------------------------------------------------------------------------                          
    def getFitnessSum(self,population,low,high):
        fitSum = 0
        for ref in population.matchSet:
            cl = population.popSet[ref]
            if cl.eventInterval[0] <= low and cl.eventInterval[1] >= high: #if classifier range subsumes segment range.
                fitSum += cl.fitness
        return fitSum
      
#-----------------------------------------------------------------------------------------------------------------
# getProbabilities:  Returns probabilities of each event from the decision THIS
#-----------------------------------------------------------------------------------------------------------------       
    def getProbabilities(self):
        a = np.empty(len(sorted(self.probabilities.items())))
        counter = 0
        for k, v in sorted(self.probabilities.items()):
            a[counter] = v
            counter += 1
        return a      
    
#-----------------------------------------------------------------------------------------------------------------
# getDecision: returns the eventTime prediction
#-----------------------------------------------------------------------------------------------------------------                     
    def getDecision(self):
        return self.decision

#-----------------------------------------------------------------------------------------------------------------
# getSurvProb: returns the survival distribution, NOT CALLED ANYWHERE YET - CHANGE THIS - to "PREDICT_PROBA" in survival_ExSTraCS.py
#-----------------------------------------------------------------------------------------------------------------  
    def getSurvProbDist(self):
        return self.survProbDist

#-----------------------------------------------------------------------------------------------------------------
# plotSurvDist: NOT CALLED ANYWHERE YET
#-----------------------------------------------------------------------------------------------------------------    
    def plotSurvDist(self, survProb):
        plt.figure(figsize=(10, 10))
        pyplot.vlines(empDist, 0, 0.05, linestyles ="solid", colors ="k")
        pyplot.xlabel('time')
        pyplot.ylabel('survival probabilty')
        
        pyplot.plot(self.times[:], survProb)
