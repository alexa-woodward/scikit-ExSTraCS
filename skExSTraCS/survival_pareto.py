"""
Name:        ExSTraCS_Pareto.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     May 15, 2015
Modified:    May 15, 2015
Description: This module defines an individual Pareto front which defines the current best multiobjective fitness boundary used in the determination
of rule fitness.  
             
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
import copy
import math

class Pareto:
    def __init__(self):     #def __init__(self, EClass, Epoch):
        #Definte the parts of the Pareto Front
        self.paretoFrontAcc = []  
        self.paretoFrontRawCov = [] #will store the actual cover values
        self.coverMax = 0.0
        self.accuracyMax = 0.0
        
    #After updateing front calculate ans save AOC
    #Change pareto fitness calculator to calculate area under cerve
    
    def rebootPareto(self, frontAcc, frontRawCov):
        """ Reinitializes pareto front from loaded run. """
        #Reboot Accuracy
        for i in range(len(frontAcc)):
            self.paretoFrontAcc.append(float(frontAcc[i]))
        self.accuracyMax = self.paretoFrontAcc[-1]
        #Reboot Coverage
        for i in range(len(frontRawCov)):
            self.paretoFrontRawCov.append(float(frontRawCov[i]))                
        self.coverMax = self.paretoFrontRawCov[0]
        
    
    def updateFront(self, objectivePair):
        """  Handles process of checking and adjusting the fitness pareto front. """
        #Update any changes to the maximum Cov - automatically adds point if new cov max is found
        #print self.classID + ' ' + self.epochID + '-------------------------------------'
        #print objectivePair
        changedFront = False
        if len(self.paretoFrontAcc) == 0:
            #print "first point addition"
            self.accuracyMax = objectivePair[0]
            self.paretoFrontAcc.append(objectivePair[0])
            self.coverMax = objectivePair[1]
            self.paretoFrontRawCov.append(objectivePair[1])
            changedFront = True
            
        elif len(self.paretoFrontAcc) == 1:
            #print "second point addition"
            if objectivePair[1] > self.coverMax:
                #print 'A'
                self.coverMax = objectivePair[1]
                if objectivePair[0] > self.accuracyMax:
                    #print '1*'
                    #Replace Point
                    self.accuracyMax = objectivePair[0]
                    self.paretoFrontAcc[0] = objectivePair[0]
                    self.paretoFrontRawCov[0] = objectivePair[1]
                    changedFront = True
                else:
                    #Add point
                    self.paretoFrontAcc.insert(0,objectivePair[0])
                    self.paretoFrontRawCov.insert(0,objectivePair[1])
                    changedFront = True

            else:
                #print 'B'
                if objectivePair[0] > self.accuracyMax:
                    self.accuracyMax = objectivePair[0]
                    #print '1*'
                    self.paretoFrontAcc.append(objectivePair[0])
                    self.paretoFrontRawCov.append(objectivePair[1])
                    changedFront = True
                else:
                    pass
        else: #Automated check and adjust when there are 2 or more points already on the front.
            #print "LARGER POINT ADDITION"
            oldParetoFrontRawCov = copy.deepcopy(self.paretoFrontRawCov)
            oldParetoFrontAcc = copy.deepcopy(self.paretoFrontAcc)
            self.paretoFrontAcc.append(objectivePair[0])
            self.paretoFrontRawCov.append(objectivePair[1])
            front = self.pareto_frontier(self.paretoFrontRawCov, self.paretoFrontAcc)
            #print front
            self.paretoFrontRawCov = front[0]
            self.paretoFrontAcc = front[1]
            self.coverMax = max(self.paretoFrontRawCov)
            self.accuracyMax = max(self.paretoFrontAcc)
            
            if oldParetoFrontRawCov != self.paretoFrontRawCov or oldParetoFrontAcc != self.paretoFrontAcc:
                changedFront = True
                #print oldParetoFrontRawCov
#                 
#         if changedFront:
#             #self.AUC = self.PolygonArea(copy.deepcopy(self.paretoFrontAcc), copy.deepcopy(self.paretoFrontRawCov))
#             
#             #self.preFitMax = 0.0 #Reset because with new front, we can have a lower max (in particular 1 point fronts will have a max of 1.0 so otherwise we'd be stuck at 1.
#             #Determine maximum AUC ratio (i.e. maximum prefitness)
#             for i in range(len(self.paretoFrontAcc)):
#                 tempMaxAUC =  self.paretoFrontAcc[i]*self.paretoFrontRawCov[i]
#                 #print tempMaxAUC
#                 if tempMaxAUC > self.maxAUC:
#                     self.maxAUC = tempMaxAUC

                
                #tempParetoFitness = (self.paretoFrontAcc[i]*self.paretoFrontRawCov[i]/self.coverMax)/self.AUC
#                 tempParetoFitness = self.paretoFrontAcc[i]*self.paretoFrontRawCov[i]/self.AUC
#                 if tempParetoFitness > self.preFitMax:
#                     self.preFitMax = tempParetoFitness
 #NOTE, the best prefitMax might not be on the front - so this should be checked even when rule not updated.!!!!!!!!!!NOT FIXED YET

        self.verifyFront()  #TEMPORARY - DEBUGGING
        return changedFront
        

    def PolygonArea(self, accList, covList):
        #Shoelace Formula
        accList.insert(0,0)
        covList.insert(0,0)
        accList.insert(1,0)
        covList.insert(1,self.coverMax)
        accList.append(self.accuracyMax)
        covList.append(0)
        #print accList
        #print covList
        n = len(covList) # of corners
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += accList[i] * covList[j]#/self.coverMax
            area -= accList[j] * covList[i]#/self.coverMax
        area = abs(area) / 2.0
        #print area
        return area

#         accList.insert(0,0)
#         covList.insert(0,0)
#         accList.insert(1,0)
#         covList.insert(1,self.coverMax)
#         accList.append(self.accuracyMax)
#         covList.append(0)
#         #print accList
#         #print covList
#         n = len(covList) # of corners
#         area = 0.0
#         for i in range(n):
#             j = (i + 1) % n
#             area += accList[i] * covList[j]
#             area -= accList[j] * covList[i]
#         area = abs(area) / 2.0
#         #print area
#         return area

        
    def verifyFront(self):
        for i in range(len(self.paretoFrontAcc)-1):
            if self.paretoFrontAcc[i] > self.paretoFrontAcc[i+1]:
                print('ERROR: Accurcy error')
                x = 5/0
            if self.paretoFrontRawCov[i] < self.paretoFrontRawCov[i+1]:
                print('ERROR: Cov error')
                x = 5/0
          

    def getParetoFitness(self, objectivePair):
        """ Determines and returns the pareto fitness based on the proportional distance of a given point"""
        objectiveCoverVal = objectivePair[1] / self.coverMax
        #Special Case
        i = 0#-1
        foundPerp = False
        badDist = None
        while not foundPerp:
            #print i
            mainLineSlope = self.calcSlope(0, objectivePair[0], 0, objectiveCoverVal)   #check that we are alwasys storing objective cover val, so there is no error here (recent thought)
            if self.paretoFrontAcc[i] == objectivePair[0] and self.paretoFrontRawCov[i] == objectivePair[1]: #is point a point on front?
                #print "POINT ON FRONT"
                goodDist = 1
                badDist = 0
                foundPerp = True
            else:
                frontPointSlope = self.calcSlope(0, self.paretoFrontAcc[i], 0, self.paretoFrontRawCov[i]/self.coverMax) 
                if i == 0 and frontPointSlope >= mainLineSlope: #Special Case:  High Coverage boundary case
                    foundPerp = True
                    if objectiveCoverVal >= self.paretoFrontRawCov[i]/self.coverMax: #Over front treated like maximum indfitness
                        if objectivePair[0] >= self.paretoFrontAcc[i]:
                            goodDist = 1
                            badDist = 0
                        else:
    #                         goodDist = calcDistance(0,objectivePair[0],0,objectiveCoverVal)
    #                         badDist = calcDistance(0,paretoFrontAcc[i],0,paretoFrontRawCov[i]/coverMax) - goodDist
                            goodDist = self.calcDistance(0,objectivePair[0],0,1)
                            badDist = self.calcDistance(0,self.paretoFrontAcc[i],0,1) - goodDist
    #                         goodDist = objectivePair[0]
    #                         badDist = paretoFrontAcc[i]-objectivePair[0]
    #                         goodDist = calcDistance(0,objectivePair[0],paretoFrontRawCov[i]/coverMax,1)
    #                         badDist = calcDistance(paretoFrontAcc[i],objectivePair[0],paretoFrontRawCov[i]/coverMax,1)
                    elif objectiveCoverVal == self.paretoFrontRawCov[i]/self.coverMax: #On the boundary Line but not a point on the front. - some more minor penalty.
                        goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                        badDist = self.calcDistance(0,self.paretoFrontAcc[i],0,self.paretoFrontRawCov[i]/self.coverMax) - goodDist
                    else: #Maximum penalty case - point is a boundary case underneath the front.
                        goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                        badDist = self.calcDistance(self.paretoFrontAcc[i],objectivePair[0],self.paretoFrontRawCov[i]/self.coverMax, objectiveCoverVal)
                        goodDist += objectivePair[0]
                        badDist += self.paretoFrontAcc[i]-objectivePair[0]
                        
                elif i == len(self.paretoFrontAcc)-1 and frontPointSlope < mainLineSlope: #Special Case:  High Accuracy boundary case
                    foundPerp = True
                    if objectivePair[0] > self.paretoFrontAcc[i]: #Over front treated like maximum indfitness
                        goodDist = 1
                        badDist = 0
                        
                    elif objectivePair[0] == self.paretoFrontAcc[i]: #On the boundary Line but not a point on the front. - some more minor penalty.
                        if self.paretoFrontAcc[i] == 1.0:
                            goodDist = 1
                            badDist = 0
                        else:
                            goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                            badDist = self.calcDistance(0,self.paretoFrontAcc[i],0,self.paretoFrontRawCov[i]/self.coverMax) - goodDist
                        
#                         if objectiveCoverVal >= self.paretoFrontRawCov[i]/self.coverMax:
#                             goodDist = 1
#                             badDist = 0
#                         else:
#                             goodDist = self.calcDistance(0,1,0,objectiveCoverVal)
#                             badDist = self.calcDistance(0,1,0,self.paretoFrontRawCov[i]/self.coverMax) - goodDist
    #                         goodDist = objectiveCoverVal
    #                         badDist = paretoFrontRawCov[i]/coverMax-objectiveCoverVal        
                                      
#                     elif objectivePair[0] == self.paretoFrontAcc[i]: #On the boundary Line but not a point on the front. - some more minor penalty.
#                         goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
#                         badDist = self.calcDistance(0,self.paretoFrontAcc[i],0,self.paretoFrontRawCov[i]/self.coverMax) - goodDist
                    else: #Maximum penalty case - point is a boundary case underneath the front.
                        goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
                        badDist = self.calcDistance(self.paretoFrontAcc[i],objectivePair[0],self.paretoFrontRawCov[i]/self.coverMax, objectiveCoverVal)
                        goodDist += objectiveCoverVal
                        badDist += self.paretoFrontRawCov[i]/self.coverMax-objectiveCoverVal   
                        
                elif frontPointSlope >= mainLineSlope: #Normal Middle front situation (we work from high point itself up to low point - but not including)
                    foundPerp = True
                    #boundaryCalculation Rule = (x1-x0)(y2-y0)-(x2-x0)(y1-y0), where 0 and 1 represent two points with a line, and 2 represents some other point.
                    boundaryCalculation = (self.paretoFrontAcc[i]-self.paretoFrontAcc[i-1])*(objectiveCoverVal-self.paretoFrontRawCov[i-1]/self.coverMax) - (objectivePair[0]-self.paretoFrontAcc[i-1])*(self.paretoFrontRawCov[i]/self.coverMax-self.paretoFrontRawCov[i-1]/self.coverMax)
                    if boundaryCalculation > 0:
                        goodDist = 1
                        badDist = 0
                    else:
                        frontIntercept =  self.calcIntercept(self.paretoFrontAcc[i], self.paretoFrontAcc[i-1], self.paretoFrontRawCov[i]/self.coverMax, self.paretoFrontRawCov[i-1]/self.coverMax, mainLineSlope)
                        badDist = self.calcDistance(frontIntercept[0],objectivePair[0],frontIntercept[1],objectiveCoverVal)
                        goodDist = self.calcDistance(0,objectivePair[0],0,objectiveCoverVal)
    
                else:
                    i += 1
            
        paretoFitness = goodDist / float(goodDist + badDist)
        return paretoFitness  
        
        
    def calcDistance(self, y1, y2, x1, x2):
        distance = math.sqrt(math.pow(x2-x1,2) + math.pow(y2-y1,2))
        return distance
            
            
    def calcIntercept(self, y1a, y2a, x1a, x2a, mainLineSlope):
        """  Calculates the coordinates at which the two lines 'A' defined by points and 'B', defined by a point and slope, intersect """
        slopeA = self.calcSlope(y1a, y2a, x1a, x2a)
        slopeB = mainLineSlope
        if slopeA == 0:
            xIntercept = x1a
            yIntercept = slopeB*xIntercept
            
        else:
            xIntercept = (y2a - slopeA*x2a) / float(slopeB-slopeA)
            yIntercept = slopeB*xIntercept
        return [yIntercept,xIntercept]
    
    
#         slopeA = self.calcSlope(y1a, y2a, x1a, x2a)
#         slopeB = mainLineSlope
#         xIntercept = (slopeA*x1a - y1a) / float(slopeA - slopeB)
#         yIntercept = slopeA*(xIntercept - x1a) + y1a
#         return [yIntercept,xIntercept]
    

#    def calcIntercept(self, y1a, y2a, x1a, x2a, y1b, x1b):
#        """  Calculates the coordinates at which the two lines 'A' defined by points and 'B', defined by a point and slope, intersect """
#        slopeA = self.calcSlope(y1a, y2a, x1a, x2a)
#        slopeB = -1*slopeA
#        xIntercept = (slopeA*x1a - slopeB*x1b + y1b - y1a) / float(slopeA - slopeB)
#        yIntercept = slopeA*(xIntercept - x1a) + y1a
#        return [yIntercept,xIntercept]   


    def calcSlope(self, y1, y2, x1, x2):
        """ Calculate slope between two points """
        if x2-x1 == 0:
            slope = 0
        else:
            slope = (y2 - y1) / (x2 - x1)
        return slope
        
        
    def pareto_frontier(self, Xs, Ys, maxX = True, maxY = True):
        """ Code obtained online: http://oco-carbon.com/metrics/find-pareto-frontiers-in-python/"""
        myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
        #print myList
        p_front = [myList[0]]    
        #print p_front
        for pair in myList[1:]:
            if maxY: 
                #if pair[1] >= p_front[-1][1]:
                if pair[1] > p_front[-1][1]:
                    p_front.append(pair)
            else:
                #if pair[1] <= p_front[-1][1]:
                if pair[1] < p_front[-1][1]:
                    p_front.append(pair)
        #print p_front
        p_frontX = [pair[0] for pair in p_front]
        p_frontY = [pair[1] for pair in p_front]
        return p_frontX, p_frontY
        
        
