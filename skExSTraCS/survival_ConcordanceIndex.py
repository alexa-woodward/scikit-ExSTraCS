# https://github.com/sebp/scikit-survival/blob/master/sksurv/metrics.py#L157-L227 
#This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy
from sklearn.base import BaseEstimator

#event_indicator = array of eventStatus (was boolean, need to make binary for our purposes)
#event_time = array of eventTimes or 'y'
#order = see _estimate_concordance_index below, sorts the array of eventTimes
#estimate = predicted eventTime 'predList'

#trainedModel.score(dataFeatures,dataPhenotypes)
#from
# def score(self,X,y):
#     predList = self.predict(X)
#     return balanced_accuracy_score(y,predList)
class ConcordanceIndex:
    def __init__(self,dataFeatures,dataEventStatus,dataEventTimes,predList):
        
        self.predList = predList
        self.n_samples = len(dataEventTimes)
        self.tied_time = 0
        self.comparable = {}
        self.concordant = 0
        self.discordant = 0
        self.tied_risk = 0
        self.c_index = None
        self.order = numpy.argsort(dataEventTimes)
        
        i = 0
        while i < self.n_samples - 1:
            time_i = dataEventTimes[self.order[i]]
            start = i + 1
            end = start
            while end < self.n_samples and dataEventTimes[self.order[end]] == time_i:
                end += 1

            # check for tied event times
            event_at_same_time = dataEventStatus[self.order[i:end]]
            censored_at_same_time = ~event_at_same_time
            for j in range(i, end):
                if dataEventStatus[self.order[j]]:
                    mask = numpy.zeros(self.n_samples, dtype=bool)
                    mask[end:] = True
                    # an event is comparable to censored samples at same time point
                    mask[i:end] = censored_at_same_time
                    self.comparable[j] = mask
                    self.tied_time += censored_at_same_time.sum()
            i = end
        #return self.comparable, self.tied_time


    def _estimate_concordance_index(self,dataEventStatus, dataEventTimes, predList, tied_tol=1e-8):
        
        if len(self.comparable) == 0:
            print("Data has no comparable pairs, cannot estimate concordance index.")
        numerator = 0.0
        denominator = 0.0
        for ind, mask in self.comparable.items():
            est_i = self.predList[self.order[ind]]
            event_i = dataEventStatus[self.order[ind]]
            est = self.predList[self.order[mask]]
           
            assert event_i, 'got censored sample at index %d, but expected uncensored' % self.order[ind]
            ties = numpy.absolute(est - est_i) <= tied_tol
            n_ties = ties.sum()
            # an event should have a higher score
            con = est < est_i
            n_con = con[~ties].sum()

            numerator += n_con + 0.5 * n_ties
            denominator += mask.sum()

            self.tied_risk += n_ties
            self.concordant += n_con
            self.discordant += est.size - n_con - n_ties

        self.c_index = numerator / denominator
        print("c_index: ",self.c_index)
        return self.c_index #, self.concordant, self.discordant, self.tied_risk, self.tied_time


    def concordance_index_censored(self):
        """Concordance index for right-censored data
        The concordance index is defined as the proportion of all comparable pairs
        in which the predictions and outcomes are concordant.
        Two samples are comparable if (i) both of them experienced an event (at different times),
        or (ii) the one with a shorter observed survival time experienced an event, in which case
        the event-free subject "outlived" the other. A pair is not comparable if they experienced
        events at the same time.
        Concordance intuitively means that two samples were ordered correctly by the model.
        More specifically, two samples are concordant, if the one with a higher estimated
        risk score has a shorter actual survival time.
        When predicted risks are identical for a pair, 0.5 rather than 1 is added to the count
        of concordant pairs.
        See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb>`
        and [1]_ for further description.
        Parameters
        ----------
        event_indicator : array-like, shape = (n_samples,)
            Boolean array denotes whether an event occurred
        event_time : array-like, shape = (n_samples,)
            Array containing the time of an event or time of censoring
        estimate : array-like, shape = (n_samples,)
            Estimated risk of experiencing an event
        tied_tol : float, optional, default: 1e-8
            The tolerance value for considering ties.
            If the absolute difference between risk scores is smaller
            or equal than `tied_tol`, risk scores are considered tied.
        Returns
        -------
        cindex : float
            Concordance index
        concordant : int
            Number of concordant pairs
        discordant : int
            Number of discordant pairs
        tied_risk : int
            Number of pairs having tied estimated risks
        tied_time : int
            Number of comparable pairs sharing the same time
        See also
        --------
        concordance_index_ipcw
            Alternative estimator of the concordance index with less bias.
        References
        ----------
        .. [1] Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A,
               "Multivariable prognostic models: issues in developing models,
               evaluating assumptions and adequacy, and measuring and reducing errors",
               Statistics in Medicine, 15(4), 361-87, 1996.
        """

        return self.c_index
