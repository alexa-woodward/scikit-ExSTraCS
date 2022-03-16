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
from sksurv.metrics import brier_score
from nonparametric_estimators import CensoringDistributionEstimator, SurvivalFunctionEstimator
from utils import check_y_survival

#event_indicator = array of eventStatus (was boolean, need to make binary for our purposes)
#event_time = array of eventTimes or 'y'
#order = see _estimate_concordance_index below, sorts the array of eventTimes
#estimate = predicted eventTime 'predList'

#trainedModel.score(dataFeatures,dataPhenotypes)
#from
# def score(self,X,y):
#     predList = self.predict(X)
#     return balanced_accuracy_score(y,predList)
class Metrics:
    def __init__(self,dataFeatures_test,dataEventStatus,dataEventTimes,dataEvents_train, dataEvents_test,predList,predProbs):
        
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


    def _check_times(self,test_time, times):
        times = check_array(numpy.atleast_1d(times), ensure_2d=False, dtype=test_time.dtype)
        times = numpy.unique(times)
        print(times)
        if times.max() >= test_time.max() or times.min() < test_time.min():
            raise ValueError(
                'all times must be within follow-up time of test data: [{}; {}['.format(
                    test_time.min(), test_time.max()))

        return times


    def _check_estimate_2d(self, predProbs, test_time, time_points):
        predProbs = check_array(predProbs, ensure_2d=False, allow_nd=False)
        time_points = self._check_times(test_time, time_points)
        check_consistent_length(test_time, predProbs)

        if predProbs.ndim == 2 and predProbs.shape[1] != time_points.shape[0]:
            raise ValueError("expected estimate with {} columns, but got {}".format(
                time_points.shape[0], predProbs.shape[1]))

        return predProbs, time_points

    
    def _brier_score(self,dataEvents_train, dataEvents_test, predProbs, times):
        """Estimate the time-dependent Brier score for right censored data.
        The time-dependent Brier score is the mean squared error at time point :math:`t`:
        .. math::
            \\mathrm{BS}^c(t) = \\frac{1}{n} \\sum_{i=1}^n I(y_i \\leq t \\land \\delta_i = 1)
            \\frac{(0 - \\hat{\\pi}(t | \\mathbf{x}_i))^2}{\\hat{G}(y_i)} + I(y_i > t)
            \\frac{(1 - \\hat{\\pi}(t | \\mathbf{x}_i))^2}{\\hat{G}(t)} ,
        where :math:`\\hat{\\pi}(t | \\mathbf{x})` is the predicted probability of
        remaining event-free up to time point :math:`t` for a feature vector :math:`\\mathbf{x}`,
        and :math:`1/\\hat{G}(t)` is a inverse probability of censoring weight, estimated by
        the Kaplan-Meier estimator.
        See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Time-dependent-Brier-Score>`
        and [1]_ for details.
        Parameters
        ----------
        survival_train : structured array, shape = (n_train_samples,)
            Survival times for training data to estimate the censoring
            distribution from.
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.
        survival_test : structured array, shape = (n_samples,)
            Survival times of test data.
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.
        estimate : array-like, shape = (n_samples, n_times)
            Estimated risk of experiencing an event for test data at `times`.
            The i-th column must contain the estimated probability of
            remaining event-free up to the i-th time point.
        times : array-like, shape = (n_times,)
            The time points for which to estimate the Brier score.
            Values must be within the range of follow-up times of
            the test data `survival_test`.
        Returns
        -------
        times : array, shape = (n_times,)
            Unique time points at which the brier scores was estimated.
        brier_scores : array , shape = (n_times,)
            Values of the brier score.
        Examples
        --------
        >>> from sksurv.datasets import load_gbsg2
        >>> from sksurv.linear_model import CoxPHSurvivalAnalysis
        >>> from sksurv.metrics import brier_score
        >>> from sksurv.preprocessing import OneHotEncoder
        Load and prepare data.
        >>> X, y = load_gbsg2()
        >>> X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)
        >>> Xt = OneHotEncoder().fit_transform(X)
        Fit a Cox model.
        >>> est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)
        Retrieve individual survival functions and get probability
        of remaining event free up to 5 years (=1825 days).
        >>> survs = est.predict_survival_function(Xt)
        >>> preds = [fn(1825) for fn in survs]
        Compute the Brier score at 5 years.
        >>> times, score = brier_score(y, y, preds, 1825)
        >>> print(score)
        [0.20881843]
        See also
        --------
        integrated_brier_score
            Computes the average Brier score over all time points.
        References
        ----------
        .. [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher,
               "Assessment and comparison of prognostic classification schemes for survival data,"
               Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545, 1999.
        """
        test_event, test_time = check_y_survival(dataEvents_test)
        predProbs, times = self._check_estimate_2d(predProbs, test_time, times)
        if predProbs.ndim == 1 and times.shape[0] == 1:
            predProbs = predProbs.reshape(-1, 1)

        # fit IPCW estimator
        cens = CensoringDistributionEstimator().fit(dataEvents_train)
        # calculate inverse probability of censoring weight at current time point t.
        prob_cens_t = cens.predict_proba(times)
        prob_cens_t[prob_cens_t == 0] = numpy.inf
        # calculate inverse probability of censoring weights at observed time point
        prob_cens_y = cens.predict_proba(test_time)
        prob_cens_y[prob_cens_y == 0] = numpy.inf

        # Calculating the brier scores at each time point
        brier_scores = numpy.empty(times.shape[0], dtype=float)
        for i, t in enumerate(times):
            est = predProbs[:, i]
            is_case = (test_time <= t) & test_event
            is_control = test_time > t

            brier_scores[i] = numpy.mean(numpy.square(est) * is_case.astype(int) / prob_cens_y
                                         + numpy.square(1.0 - est) * is_control.astype(int) / prob_cens_t[i])

        return times, brier_scores


    def integrated_brier_score(datEventTimes_train, dataEventTimes_test, predProbs, times):
        """The Integrated Brier Score (IBS) provides an overall calculation of
        the model performance at all available times :math:`t_1 \\leq t \\leq t_\\text{max}`.
        The integrated time-dependent Brier score over the interval
        :math:`[t_1; t_\\text{max}]` is defined as
        .. math::
            \\mathrm{IBS} = \\int_{t_1}^{t_\\text{max}} \\mathrm{BS}^c(t) d w(t)
        where the weighting function is :math:`w(t) = t / t_\\text{max}`.
        The integral is estimated via the trapezoidal rule.
        See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Time-dependent-Brier-Score>`
        and [1]_ for further details.
        Parameters
        ----------
        survival_train : structured array, shape = (n_train_samples,)
            Survival times for training data to estimate the censoring
            distribution from.
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.
        survival_test : structured array, shape = (n_samples,)
            Survival times of test data.
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.
        estimate : array-like, shape = (n_samples, n_times)
            Estimated risk of experiencing an event for test data at `times`.
            The i-th column must contain the estimated probability of
            remaining event-free up to the i-th time point.
        times : array-like, shape = (n_times,)
            The time points for which to estimate the Brier score.
            Values must be within the range of follow-up times of
            the test data `survival_test`.
        Returns
        -------
        ibs : float
            The integrated Brier score.
        Examples
        --------
        >>> import numpy
        >>> from sksurv.datasets import load_gbsg2
        >>> from sksurv.linear_model import CoxPHSurvivalAnalysis
        >>> from sksurv.metrics import integrated_brier_score
        >>> from sksurv.preprocessing import OneHotEncoder
        Load and prepare data.
        >>> X, y = load_gbsg2()
        >>> X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)
        >>> Xt = OneHotEncoder().fit_transform(X)
        Fit a Cox model.
        >>> est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)
        Retrieve individual survival functions and get probability
        of remaining event free from 1 year to 5 years (=1825 days).
        >>> survs = est.predict_survival_function(Xt)
        >>> times = numpy.arange(365, 1826)
        >>> preds = numpy.asarray([[fn(t) for t in times] for fn in survs])
        Compute the integrated Brier score from 1 to 5 years.
        >>> score = integrated_brier_score(y, y, preds, times)
        >>> print(score)
        0.1815853064627424
        See also
        --------
        brier_score
            Computes the Brier score at specified time points.
        as_integrated_brier_score_scorer
            Wrapper class that uses :func:`integrated_brier_score`
            in its ``score`` method instead of the default
            :func:`concordance_index_censored`.
        References
        ----------
        .. [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher,
               "Assessment and comparison of prognostic classification schemes for survival data,"
               Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545, 1999.
        """
        # Computing the brier scores
        times, brier_scores = brier_score(dataEvents_train, dataEvents_test, predProbs, times)

        if times.shape[0] < 2:
            raise ValueError("At least two time points must be given")
