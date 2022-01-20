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
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted

from .exceptions import NoComparablePairException
from .nonparametric import CensoringDistributionEstimator, SurvivalFunctionEstimator
from .util import check_y_survival

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
    def __init__(self,????)
        
    def _get_comparable(self,dataEventStatus,dataEventTimes, order): 
        n_samples = len(dataEventTimes)
        print("number of samples: ", n_samples)
        tied_time = 0
        comparable = {}
        i = 0
        while i < n_samples - 1:
            time_i = model.env.dataEventTimes[order[i]]
            start = i + 1
            end = start
            while end < n_samples and dataEventTimes[order[end]] == time_i:
                end += 1

            # check for tied event times
            event_at_same_time = dataEventStatus[order[i:end]]
            censored_at_same_time = ~event_at_same_time
            for j in range(i, end):
                if dataEventIndicator[order[j]]:
                    mask = numpy.zeros(n_samples, dtype=bool)
                    mask[end:] = True
                    # an event is comparable to censored samples at same time point
                    mask[i:end] = censored_at_same_time
                    comparable[j] = mask
                    tied_time += censored_at_same_time.sum()
            i = end

        return comparable, tied_time


    def _estimate_concordance_index(self,dataEventStatus, dataEventTimes, predList, weights, tied_tol=1e-8):
        order = numpy.argsort(dataEventTimes)

        comparable, tied_time = _get_comparable(dataEventStatus, dataEventTimes, order)

        if len(comparable) == 0:
            raise NoComparablePairException(
                "Data has no comparable pairs, cannot estimate concordance index.")

        concordant = 0
        discordant = 0
        tied_risk = 0
        numerator = 0.0
        denominator = 0.0
        for ind, mask in comparable.items():
            est_i = predList[order[ind]]
            event_i = dataEventStatus[order[ind]]
            w_i = weights[order[ind]]

            est = estimate[order[mask]]

            assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

            ties = numpy.absolute(est - est_i) <= tied_tol
            n_ties = ties.sum()
            # an event should have a higher score
            con = est < est_i
            n_con = con[~ties].sum()

            numerator += w_i * n_con + 0.5 * w_i * n_ties
            denominator += w_i * mask.sum()

            tied_risk += n_ties
            concordant += n_con
            discordant += est.size - n_con - n_ties

        cindex = numerator / denominator
        return cindex, concordant, discordant, tied_risk, tied_time


    def concordance_index_censored(dself,dataEventStatus, dataEventTimes, predList, tied_tol=1e-8):
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
        dataEventStatus, dataEventTimes, predList = _check_inputs(
            dataEventStatus, dataEventTimes, predList) #don't need this

        w = numpy.ones_like(predList)

        return _estimate_concordance_index(event_indicator, event_time, estimate, w, tied_tol)




    def cumulative_dynamic_auc(survival_train, survival_test, estimate, times, tied_tol=1e-8):
        """Estimator of cumulative/dynamic AUC for right-censored time-to-event data.
        The receiver operating characteristic (ROC) curve and the area under the
        ROC curve (AUC) can be extended to survival data by defining
        sensitivity (true positive rate) and specificity (true negative rate)
        as time-dependent measures. *Cumulative cases* are all individuals that
        experienced an event prior to or at time :math:`t` (:math:`t_i \\leq t`),
        whereas *dynamic controls* are those with :math:`t_i > t`.
        The associated cumulative/dynamic AUC quantifies how well a model can
        distinguish subjects who fail by a given time (:math:`t_i \\leq t`) from
        subjects who fail after this time (:math:`t_i > t`).
        Given an estimator of the :math:`i`-th individual's risk score
        :math:`\\hat{f}(\\mathbf{x}_i)`, the cumulative/dynamic AUC at time
        :math:`t` is defined as
        .. math::
            \\widehat{\\mathrm{AUC}}(t) =
            \\frac{\\sum_{i=1}^n \\sum_{j=1}^n I(y_j > t) I(y_i \\leq t) \\omega_i
            I(\\hat{f}(\\mathbf{x}_j) \\leq \\hat{f}(\\mathbf{x}_i))}
            {(\\sum_{i=1}^n I(y_i > t)) (\\sum_{i=1}^n I(y_i \\leq t) \\omega_i)}
        where :math:`\\omega_i` are inverse probability of censoring weights (IPCW).
        To estimate IPCW, access to survival times from the training data is required
        to estimate the censoring distribution. Note that this requires that survival
        times `survival_test` lie within the range of survival times `survival_train`.
        This can be achieved by specifying `times` accordingly, e.g. by setting
        `times[-1]` slightly below the maximum expected follow-up time.
        IPCW are computed using the Kaplan-Meier estimator, which is
        restricted to situations where the random censoring assumption holds and
        censoring is independent of the features.
        This function can also be used to evaluate models with time-dependent predictions
        :math:`\\hat{f}(\\mathbf{x}_i, t)`, such as :class:`sksurv.ensemble.RandomSurvivalForest`
        (see :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Using-Time-dependent-Risk-Scores>`).
        In this case, `estimate` must be a 2-d array where ``estimate[i, j]`` is the
        predicted risk score for the i-th instance at time point ``times[j]``.
        Finally, the function also provides a single summary measure that refers to the mean
        of the :math:`\\mathrm{AUC}(t)` over the time range :math:`(\\tau_1, \\tau_2)`.
        .. math::
            \\overline{\\mathrm{AUC}}(\\tau_1, \\tau_2) =
            \\frac{1}{\\hat{S}(\\tau_1) - \\hat{S}(\\tau_2)}
            \\int_{\\tau_1}^{\\tau_2} \\widehat{\\mathrm{AUC}}(t)\\,d \\hat{S}(t)
        where :math:`\\hat{S}(t)` is the Kaplan–Meier estimator of the survival function.
        See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Time-dependent-Area-under-the-ROC>`,
        [1]_, [2]_, [3]_ for further description.
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
        estimate : array-like, shape = (n_samples,) or (n_samples, n_times)
            Estimated risk of experiencing an event of test data.
            If `estimate` is a 1-d array, the same risk score across all time
            points is used. If `estimate` is a 2-d array, the risk scores in the
            j-th column are used to evaluate the j-th time point.
        times : array-like, shape = (n_times,)
            The time points for which the area under the
            time-dependent ROC curve is computed. Values must be
            within the range of follow-up times of the test data
            `survival_test`.
        tied_tol : float, optional, default: 1e-8
            The tolerance value for considering ties.
            If the absolute difference between risk scores is smaller
            or equal than `tied_tol`, risk scores are considered tied.
        Returns
        -------
        auc : array, shape = (n_times,)
            The cumulative/dynamic AUC estimates (evaluated at `times`).
        mean_auc : float
            Summary measure referring to the mean cumulative/dynamic AUC
            over the specified time range `(times[0], times[-1])`.
        See also
        --------
        as_cumulative_dynamic_auc_scorer
            Wrapper class that uses :func:`cumulative_dynamic_auc`
            in its ``score`` method instead of the default
            :func:`concordance_index_censored`.
        References
        ----------
        .. [1] H. Uno, T. Cai, L. Tian, and L. J. Wei,
               "Evaluating prediction rules for t-year survivors with censored regression models,"
               Journal of the American Statistical Association, vol. 102, pp. 527–537, 2007.
        .. [2] H. Hung and C. T. Chiang,
               "Estimation methods for time-dependent AUC models with survival data,"
               Canadian Journal of Statistics, vol. 38, no. 1, pp. 8–26, 2010.
        .. [3] J. Lambert and S. Chevret,
               "Summary measure of discrimination in survival models based on cumulative/dynamic time-dependent ROC curves,"
               Statistical Methods in Medical Research, 2014.
        """
        test_event, test_time = check_y_survival(survival_test)
        estimate, times = _check_estimate_2d(estimate, test_time, times)

        n_samples = estimate.shape[0]
        n_times = times.shape[0]
        if estimate.ndim == 1:
            estimate = numpy.broadcast_to(estimate[:, numpy.newaxis], (n_samples, n_times))

        # fit and transform IPCW
        cens = CensoringDistributionEstimator()
        cens.fit(survival_train)
        ipcw = cens.predict_ipcw(survival_test)

        # expand arrays to (n_samples, n_times) shape
        test_time = numpy.broadcast_to(test_time[:, numpy.newaxis], (n_samples, n_times))
        test_event = numpy.broadcast_to(test_event[:, numpy.newaxis], (n_samples, n_times))
        times_2d = numpy.broadcast_to(times, (n_samples, n_times))
        ipcw = numpy.broadcast_to(ipcw[:, numpy.newaxis], (n_samples, n_times))

        # sort each time point (columns) by risk score (descending)
        o = numpy.argsort(-estimate, axis=0)
        test_time = numpy.take_along_axis(test_time, o, axis=0)
        test_event = numpy.take_along_axis(test_event, o, axis=0)
        estimate = numpy.take_along_axis(estimate, o, axis=0)
        ipcw = numpy.take_along_axis(ipcw, o, axis=0)

        is_case = (test_time <= times_2d) & test_event
        is_control = test_time > times_2d
        n_controls = is_control.sum(axis=0)

        # prepend row of infinity values
        estimate_diff = numpy.concatenate((numpy.broadcast_to(numpy.infty, (1, n_times)), estimate))
        is_tied = numpy.absolute(numpy.diff(estimate_diff, axis=0)) <= tied_tol

        cumsum_tp = numpy.cumsum(is_case * ipcw, axis=0)
        cumsum_fp = numpy.cumsum(is_control, axis=0)
        true_pos = cumsum_tp / cumsum_tp[-1]
        false_pos = cumsum_fp / n_controls

        scores = numpy.empty(n_times, dtype=float)
        it = numpy.nditer((true_pos, false_pos, is_tied), order="F", flags=["external_loop"])
        with it:
            for i, (tp, fp, mask) in enumerate(it):
                idx = numpy.flatnonzero(mask) - 1
                # only keep the last estimate for tied risk scores
                tp_no_ties = numpy.delete(tp, idx)
                fp_no_ties = numpy.delete(fp, idx)
                # Add an extra threshold position
                # to make sure that the curve starts at (0, 0)
                tp_no_ties = numpy.r_[0, tp_no_ties]
                fp_no_ties = numpy.r_[0, fp_no_ties]
                scores[i] = numpy.trapz(tp_no_ties, fp_no_ties)

        if n_times == 1:
            mean_auc = scores[0]
        else:
            surv = SurvivalFunctionEstimator()
            surv.fit(survival_test)
            s_times = surv.predict_proba(times)
            # compute integral of AUC over survival function
            d = -numpy.diff(numpy.r_[1.0, s_times])
            integral = (scores * d).sum()
            mean_auc = integral / (1.0 - s_times[-1])

        return scores, mean_auc




    class _ScoreOverrideMixin:
        def __init__(self, estimator, predict_func, score_func, score_index, greater_is_better):
            if not hasattr(estimator, predict_func):
                raise AttributeError("{!r} object has no attribute {!r}".format(estimator, predict_func))

            self.estimator = estimator
            self._predict_func = predict_func
            self._score_func = score_func
            self._score_index = score_index
            self._sign = 1 if greater_is_better else -1

        def _get_score_params(self):
            """Return dict of parameters passed to ``score_func``."""
            params = self.get_params(deep=False)
            del params["estimator"]
            return params

        def fit(self, X, y, **fit_params):
            self._train_y = numpy.array(y, copy=True)
            self.estimator_ = self.estimator.fit(X, y, **fit_params)
            return self

        def _do_predict(self, X):
            predict_func = getattr(self.estimator_, self._predict_func)
            return predict_func(X)

        def score(self, X, y):
            """Returns the score on the given data.
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Input data, where n_samples is the number of samples and
                n_features is the number of features.
            y : array-like of shape (n_samples,)
                Target relative to X for classification or regression;
                None for unsupervised learning.
            Returns
            -------
            score : float
            """
            estimate = self._do_predict(X)
            score = self._score_func(
                survival_train=self._train_y,
                survival_test=y,
                estimate=estimate,
                **self._get_score_params(),
            )
            if self._score_index is not None:
                score = score[self._score_index]
            return self._sign * score

        @if_delegate_has_method(delegate="estimator_")
        def predict(self, X):
            """Call predict on the estimator.
            Only available if estimator supports ``predict``.
            Parameters
            ----------
            X : indexable, length n_samples
                Must fulfill the input assumptions of the
                underlying estimator.
            """
            check_is_fitted(self, "estimator_")
            return self.estimator_.predict(X)

        @if_delegate_has_method(delegate="estimator_")
        def predict_cumulative_hazard_function(self, X):
            """Call predict_cumulative_hazard_function on the estimator.
            Only available if estimator supports ``predict_cumulative_hazard_function``.
            Parameters
            ----------
            X : indexable, length n_samples
                Must fulfill the input assumptions of the
                underlying estimator.
            """
            check_is_fitted(self, "estimator_")
            return self.estimator_.predict_cumulative_hazard_function(X)

        @if_delegate_has_method(delegate="estimator_")
        def predict_survival_function(self, X):
            """Call predict_survival_function on the estimator.
            Only available if estimator supports ``predict_survival_function``.
            Parameters
            ----------
            X : indexable, length n_samples
                Must fulfill the input assumptions of the
                underlying estimator.
            """
            check_is_fitted(self, "estimator_")
            return self.estimator_.predict_survival_function(X)


    class as_cumulative_dynamic_auc_scorer(_ScoreOverrideMixin, BaseEstimator):
        """Wraps an estimator to use :func:`cumulative_dynamic_auc` as ``score`` function.
        See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Using-Metrics-in-Hyper-parameter-Search>`
        for using it for hyper-parameter optimization.
        Parameters
        ----------
        estimator : object
            Instance of an estimator.
        times : array-like, shape = (n_times,)
            The time points for which the area under the
            time-dependent ROC curve is computed. Values must be
            within the range of follow-up times of the test data
            `survival_test`.
        tied_tol : float, optional, default: 1e-8
            The tolerance value for considering ties.
            If the absolute difference between risk scores is smaller
            or equal than `tied_tol`, risk scores are considered tied.
        Attributes
        ----------
        estimator_ : estimator
            Estimator that was fit.
        See also
        --------
        cumulative_dynamic_auc
        """

        def __init__(self, estimator, times, tied_tol=1e-8):
            super().__init__(
                estimator=estimator,
                predict_func="predict",
                score_func=cumulative_dynamic_auc,
                score_index=1,
                greater_is_better=True,
            )
            self.times = times
            self.tied_tol = tied_tol


    class as_concordance_index_ipcw_scorer(_ScoreOverrideMixin, BaseEstimator):
        """Wraps an estimator to use :func:`concordance_index_ipcw` as ``score`` function.
        See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Using-Metrics-in-Hyper-parameter-Search>`
        for using it for hyper-parameter optimization.
        Parameters
        ----------
        estimator : object
            Instance of an estimator.
        tau : float, optional
            Truncation time. The survival function for the underlying
            censoring time distribution :math:`D` needs to be positive
            at `tau`, i.e., `tau` should be chosen such that the
            probability of being censored after time `tau` is non-zero:
            :math:`P(D > \\tau) > 0`. If `None`, no truncation is performed.
        tied_tol : float, optional, default: 1e-8
            The tolerance value for considering ties.
            If the absolute difference between risk scores is smaller
            or equal than `tied_tol`, risk scores are considered tied.´
        Attributes
        ----------
        estimator_ : estimator
            Estimator that was fit.
        See also
        --------
        concordance_index_ipcw
        """

        def __init__(self, estimator, tau=None, tied_tol=1e-8):
            super().__init__(
                estimator=estimator,
                predict_func="predict",
                score_func=concordance_index_ipcw,
                score_index=0,
                greater_is_better=True,
            )
            self.tau = tau
            self.tied_tol = tied_tol


    class as_integrated_brier_score_scorer(_ScoreOverrideMixin, BaseEstimator):
        """Wraps an estimator to use the negative of :func:`integrated_brier_score` as ``score`` function.
        The estimator needs to be able to estimate survival functions via
        a ``predict_survival_function`` method.
        See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Using-Metrics-in-Hyper-parameter-Search>`
        for using it for hyper-parameter optimization.
        Parameters
        ----------
        estimator : object
            Instance of an estimator that provides ``predict_survival_function``.
        times : array-like, shape = (n_times,)
            The time points for which to estimate the Brier score.
            Values must be within the range of follow-up times of
            the test data `survival_test`.
        Attributes
        ----------
        estimator_ : estimator
            Estimator that was fit.
        See also
        --------
        integrated_brier_score
        """

        def __init__(self, estimator, times):
            super().__init__(
                estimator=estimator,
                predict_func="predict_survival_function",
                score_func=integrated_brier_score,
                score_index=None,
                greater_is_better=False,
            )
            self.times = times

        def _do_predict(self, X):
            predict_func = getattr(self.estimator_, self._predict_func)
            surv_fns = predict_func(X)
            times = self.times
            estimates = numpy.empty((len(surv_fns), len(times)))
            for i, fn in enumerate(surv_fns):
                estimates[i, :] = fn(times)
            return estimates
