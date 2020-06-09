import numpy as np
from sklearn import linear_model
from scipy.optimize import curve_fit
import itertools

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_estimator import CausalEstimator


class NonLinearMediationFormulaEstimator(CausalEstimator):
    """Compute direct, indirect and total effect of treatment using mediation formula in nonliniar system.
    """

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observed_common_causes = None
        self._observed_common_causes_names = ''
        self.logger.info("INFO: Using Nonlinear Mediation Formula Estimator")
        self.symbolic_estimator = self.construct_symbolic_estimator(self._target_estimand)
        self.logger.info(self.symbolic_estimator)

    def _estimate_effect(self):
        #Naive
        if (len(np.unique(self._mediator)) <= 2) and (len(np.unique(self._treatment)) <= 2) \
                and (len(np.unique(self._outcome)) <= 2):
            effect_estimate = self.naive_estimate()
            estimate = CausalEstimate(effect_estimate, None, None)

        else:
            # All treatments are set to the same constant value
            effect_estimate = self._do_new(self._treatment_value, self.mediator_new(self._control_value)) \
                              - self._do_new(self._control_value, self.mediator_new(self._control_value))

            estimate = CausalEstimate(estimate=effect_estimate,
                                target_estimand=None,
                                realized_estimand_expr=None,
                                intercept=None)
        return estimate

    def naive_estimate(self):
        g = [[0, 1], [1, 0]]  # E(Y |x=xi, z=zi)
        h = 0
        for i in range(2):
            for j in range(2):
                df_all_outcome = self._data.loc[self._data[self._treatment_name] == i][self._data[self._outcome_name] == j]
                df_outcome1 = self._data.loc[self._data[self._treatment_name] == i][self._data[self._outcome_name] == 1][
                    self._data[self._mediator_name] == j]
                if len(df_all_outcome) == 0:
                    g[i][j] = 0
                else:
                    g[i][j] = len(df_outcome1) / len(df_all_outcome)
        # E(Z|x=0)
        df_allz_notreatment = self._data.loc[self._data[self._outcome_name] == 0]
        df_withz_notreatment = self._data.loc[self._data[self._outcome_name] == 0][self._data[self._mediator_name] == 1]

        if len(df_allz_notreatment) != 0:
            h = len(df_withz_notreatment) / len(df_allz_notreatment)

        effect_estimate = (g[1][0] - g[0][0]) * (1 - h) + (g[1][1] - g[0][1]) * h

        return effect_estimate

    def mediator_new(self, treatment_val):
        interventional_treatment_2d = np.full((self._treatment.shape[0], 1), treatment_val)
        features = self._build_features()
        new_features = np.concatenate((interventional_treatment_2d, features[:,1: ]), axis=1)
        model = linear_model.LinearRegression()
        model.fit(features, self._mediator)
        interventional_mediator = model.predict(new_features)
        return interventional_mediator.mean()

    def construct_symbolic_estimator(self, estimand):
        expr = "b: " + ",".join(estimand.outcome_variable) + "~"
        var_list = estimand.treatment_variable + estimand.backdoor_variables
        expr += "+".join(var_list)
        if self._effect_modifier_names:
            interaction_terms = ["{0}*{1}".format(x[0], x[1]) for x in itertools.product(estimand.treatment_variable, self._effect_modifier_names)]
            expr += "+" + "+".join(interaction_terms)
        if self._mediator_name:
            interaction_terms = ["{0}*{1}".format(x[0], x[1]) for x in
                                 itertools.product(estimand.treatment_variable, self._mediator_name)]
            expr += "+" + "+".join(interaction_terms)
            interaction_terms = ["{0}*{1}".format(x[0], x[1]) for x in
                                 itertools.product(self._mediator_name, estimand.outcome_variable)]
            expr += "+" + "+".join(interaction_terms)
        return expr 

    def _build_features(self):
        n_samples = self._treatment.shape[0]
        treatment_2d = self._treatment.to_numpy().reshape((n_samples, 1))
        features = treatment_2d
        return features

    def _build_suitable_model(self, treatment_val, mediator_val):
        features = self._treatment.to_numpy()
        function = None
        if self._relation_function == 'linear':
            function = linear_func
        elif self._relation_function == 'logistic':
            function = logistic_func
        elif self._relation_function == 'log':
            function = logarithmic_func
        elif self._relation_function == 'exp':
            function = exp_func
        elif self._relation_function == 'sin':
            function = sinusoidal_func

        popt, _ = curve_fit(function, (features, self._mediator), self._outcome)
        outcome = function((treatment_val, mediator_val), *popt)
        return outcome

    def _do_new(self, x, z):
        interventional_treatment_2d = np.full((self._treatment.shape[0], 1), x)
        interventional_mediator_2d = np.full((self._mediator.shape[0], 1), z)
        features = self._build_features()
        new_features = np.concatenate((interventional_treatment_2d, features[:, 1:]), axis=1)
        interventional_outcomes = self._build_suitable_model(new_features, interventional_mediator_2d)
        return interventional_outcomes.mean()

def logarithmic_func(x, a, b, c, d):
    _x, _z = x
    return a*(b**_x[0]) + c + d*_z

def linear_func(x, a, b, c, d):
    _x, _z = x
    return a*(_x[0]**b) + c + d*_z

def logistic_func(x, L, b, c, d, f):
    _x, _z = x
    return L/(1 + c*np.exp(-b*_x[0])) + d + f*_z

def exp_func(x, a, b, c, d, f):
    _x, _z = x
    return a * np.exp(-b * (_x[0] - c)) + d + f*_z

def sinusoidal_func(x, a, b, c, d, f):
    _x, _z = x
    return a * np.sin(b * (_x[0] - np.radians(c))) + d + f*_z