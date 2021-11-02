'''
Created on Mar 26, 2015

@author: Ryan
'''
#looks like given uniform difference we expect random prediction error of 0.3
#our setup allows rules outside range
#prediction should only consider within real data range values in making predictions.

import random
import math

#testRange = 10000000
#differenceSum = 0
#for i in range(testRange):
#    x = random.uniform(0,1)
#    y = random.uniform(0,1)
#    differenceSum += math.fabs(x-y)
#    
#AverageDiff = differenceSum /float(testRange)
#print AverageDiff


testRange = 1000
differenceSum = 0
for i in range(testRange):
    x = random.gauss(0.5,0.1)
    print(x)
    y = random.gauss(0.5,0.2)
    print(y)
    differenceSum += math.fabs(x-y)
    
AverageDiff = differenceSum /float(testRange)
print(AverageDiff)
