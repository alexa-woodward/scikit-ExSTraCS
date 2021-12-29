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
from exstracs_constants import *
import random
from sklearn.neighbors import KernelDensity
#------------------------------------------------------

class Prediction:
    def __init__(self, population):  #now takes in population ( have to reference the match set to do prediction)  pop.matchSet
        """ Constructs the voting array and determines the prediction decision. """
        self.decision = None
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
                
    def individualSurvivalProb(self)
      self.survProb = None
      
      #Need to
      #1. get coverage from other matched instances. Is this stored anywhere already? If not need to make new function to store these. 
      #2. Use KDE to estimate the PDF
      #3. Use: survival = 1 - np.cumsum(probabilities) to retrieve the survival probabilities


                        
    def getFitnessSum(self,population,low,high):
        """ Get the fitness Sum of rules in the rule-set. For continuous phenotype prediction. """
        fitSum = 0
        for ref in population.matchSet:
            cl = population.popSet[ref]
            if cl.eventInterval[0] <= low and cl.eventInterval[1] >= high: #if classifier range subsumes segment range.
                fitSum += cl.fitness
        return fitSum
    
                    
    def getDecision(self):
        """ Returns prediction decision. """
        return self.decision


    def getSet(self):
        """ Returns prediction decision. """
        return self.vote
