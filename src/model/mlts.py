import copy
import logging
import os
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import lil_matrix, hstack, eye
from scipy.special import expit, softmax
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from utility.access_file import save_data, load_data

logger = logging.getLogger(__name__)
EPSILON = np.finfo(np.float).eps
EPSILON_BERN = 0.01
LOG_EPSILON = 0.0001
UPPER_BOUND = np.log(sys.float_info.max) * 10
LOWER_BOUND = np.log(sys.float_info.min) * 10
np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')


class mltS:
    def __init__(self, binarize_input_feature=True, normalize_input_feature=False,
                 use_external_features=True, cutting_point=3650, fit_intercept=True, subsample_input_size=0.3,
                 subsample_mini_input_size=0.3, subsample_labels_size=50, apply_peer=True, beta_peer=0.75,
                 apply_trusted_loss=False, cost_subsample_size=100,
                 calc_label_cost=True, calc_total_cost=False, label_closeness_sim=True,
                 corr_label_sim=True, corr_input_sim=True, decision_threshold=0.5,
                 apply_partial_label=False, use_trusted_partial_label=False, tau_partial=0.01, alpha_partial=0.75,
                 alpha_glob=0.1, num_neighbors=2, penalty='elasticnet', alpha_elastic=0.0001, l1_ratio=0.65,
                 fuse_weight=False, sigma=2, kappa=0.01, beta_source=0.01, lambdas=[0.01, 0.01, 0.01, 0.01, 0.01],
                 loss_threshold=0.05, early_stop=False, learning_type="optimal", lr=0.0001, lr0=0.0, delay_factor=1.0,
                 forgetting_rate=0.9, num_models=3, batch=30, max_inner_iter=100, num_epochs=3, num_jobs=2,
                 display_interval=2, shuffle=True, random_state=12345, log_path='../../log'):
        logging.basicConfig(filename=os.path.join(
            log_path, 'mltS_events'), level=logging.DEBUG)

        self.apply_partial_label = apply_partial_label
        self.use_trusted_partial_label = use_trusted_partial_label
        self.binarize_input_feature = binarize_input_feature
        self.normalize_input_feature = normalize_input_feature
        if normalize_input_feature:
            self.binarize_input_feature = False
        self.use_external_features = use_external_features
        self.apply_peer = apply_peer
        self.beta_peer = beta_peer
        self.cutting_point = cutting_point
        self.fit_intercept = fit_intercept
        self.decision_threshold = decision_threshold
        self.subsample_input_size = subsample_input_size
        self.subsample_mini_input_size = subsample_mini_input_size
        self.subsample_labels_size = subsample_labels_size
        self.label_closeness_sim = label_closeness_sim
        self.corr_label_sim = corr_label_sim
        self.corr_input_sim = corr_input_sim
        self.penalty = penalty
        self.fuse_weight = fuse_weight
        self.alpha_elastic = alpha_elastic
        self.alpha_partial = alpha_partial
        self.alpha_glob = alpha_glob
        self.tau_partial = tau_partial
        self.l1_ratio = l1_ratio
        self.sigma = sigma
        self.lambdas = lambdas
        self.lam_1 = lambdas[0]
        self.lam_2 = lambdas[1]
        self.lam_3 = lambdas[2]
        self.lam_4 = lambdas[3]
        self.lam_5 = lambdas[4]
        self.kappa = kappa
        self.beta_source = beta_source

        # compute cost for labels
        self.apply_trusted_loss = apply_trusted_loss
        self.cost_subsample_size = cost_subsample_size
        self.calc_label_cost = calc_label_cost
        if self.calc_label_cost is False:
            self.calc_label_cost = True
        self.calc_total_cost = calc_total_cost
        self.loss_threshold = loss_threshold
        self.early_stop = early_stop
        self.learning_type = learning_type
        self.lr = lr
        self.lr0 = lr0
        self.forgetting_rate = forgetting_rate
        self.delay_factor = delay_factor
        self.batch = batch
        self.max_inner_iter = max_inner_iter
        self.num_epochs = num_epochs
        self.num_jobs = num_jobs
        self.num_models = num_models
        self.num_neighbors = num_neighbors
        self.display_interval = display_interval
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = 0
        self.log_path = log_path
        warnings.filterwarnings("ignore", category=Warning)

    def __print_arguments(self, **kwargs):
        argdict = dict()
        for key, value in kwargs.items():
            argdict.update({key: value})
        if self.use_trusted:
            argdict.update(
                {'beta_source': 'Constant that applies for source specific weights: {0}'.format(self.beta_source)})
            argdict.update({'apply_trusted_loss': 'Apply discrepeny based on trusted source loss? {0}'.format(
                self.apply_trusted_loss)})
        argdict.update(
            {'apply_peer': 'Learn from peers? {0}'.format(self.apply_peer)})
        if self.apply_peer:
            argdict.update({
                'beta_peer': 'Constant that scales the amount to absorb information from global learner to individual learning: {0}'.format(
                    self.beta_peer)})
        argdict.update({'apply_partial_label': 'Fill the possible candidate labels for each example? {0}'.format(
            self.apply_partial_label)})
        if self.apply_partial_label:
            argdict.update({
                'use_trusted_partial_label': 'Use trusted data to fill the possible candidate labels for each example {0}'.format(
                    self.use_trusted_partial_label)})
            argdict.update(
                {'alpha_partial': 'Constant scaling the labeling information: {0}'.format(self.alpha_partial)})
            argdict.update(
                {'tau_partial': 'Cutoff threshold for partial labeling: {0}'.format(self.tau_partial)})
            argdict.update(
                {'num_neighbors': 'Number of neighbors used for partial labeling: {0}'.format(self.num_neighbors)})
        argdict.update(
            {'binarize': 'Binarize data? {0}'.format(self.binarize_input_feature)})
        argdict.update({'normalize_input_feature': 'Normalize data? {0}'.format(
            self.normalize_input_feature)})
        argdict.update({'use_external_features': 'Whether to use external features '
                                                 'that are included in data? {0}'.format(self.use_external_features)})
        argdict.update({'cutting_point': 'The cutting point after which binarize '
                                         'operation is halted in data: {0}'.format(self.cutting_point)})
        argdict.update({'fit_intercept': 'Whether the intercept should be estimated '
                                         'or not? {0}'.format(self.fit_intercept)})
        argdict.update({'decision_threshold': 'The decision cutoff threshold: {0}'.format(
            self.decision_threshold)})
        argdict.update({'subsample_input_size': 'Subsampling inputs: {0}'.format(
            self.subsample_input_size)})
        argdict.update(
            {'subsample_mini_input_size': 'Mini batch input size: {0}'.format(self.subsample_mini_input_size)})
        argdict.update({'subsample_labels_size': 'Subsampling labels: {0}'.format(
            self.subsample_labels_size)})
        argdict.update({'cost_subsample_size': 'Subsampling size '
                                               'for computing a cost: {0}'.format(self.cost_subsample_size)})
        argdict.update({'alpha_glob': 'Constant controlling trade-off between local and global weights: {0}'.format(
            self.alpha_glob)})
        argdict.update(
            {'penalty': 'The penalty (aka regularization term): {0}'.format(self.penalty)})
        if self.penalty == "elasticnet":
            argdict.update(
                {'alpha_elastic-elastic': 'Constant controlling elastic term: {0}'.format(self.alpha_elastic)})
            argdict.update(
                {'l1_ratio': 'The elastic net mixing parameter: {0}'.format(self.l1_ratio)})
        argdict.update({'fuse_weight': 'Adjust parameters using a provided '
                                       'similarity matrix? {0}'.format(self.fuse_weight)})
        if self.fuse_weight:
            argdict.update({'sigma': 'Constant that scales the amount of laplacian norm regularization '
                                     'paramters: {0}'.format(self.sigma)})
        argdict.update(
            {'lambdas': 'Six hyper-parameters for constraints: {0}'.format(self.lambdas)})
        argdict.update({'label_closeness_sim': 'Whether to enforce labels similarity '
                                               'constraint? {0}'.format(self.label_closeness_sim)})
        argdict.update({'corr_label_sim': 'Whether to enforce labels correlation '
                                          'constraint from dataset? {0}'.format(self.corr_label_sim)})
        argdict.update({'corr_input_sim': 'Whether to enforce instances correlation '
                                          'constraint from a dataset? {0}'.format(self.corr_input_sim)})
        argdict.update({'calc_label_cost': 'Whether to include labels cost? {0}'.format(
            self.calc_label_cost)})
        argdict.update({'calc_total_cost': 'Whether to compute total cost? {0}'.format(
            self.calc_total_cost)})
        argdict.update(
            {'loss_threshold': 'A cutoff threshold between two consecutive rounds: {0}'.format(self.loss_threshold)})
        argdict.update(
            {'early_stop': 'Whether to apply early stopping criteria? {0}'.format(self.early_stop)})
        argdict.update(
            {'learning_type': 'The learning rate schedule: {0}'.format(self.learning_type)})
        if self.learning_type == "optimal":
            argdict.update({'lr': 'The learning rate: {0}'.format(self.lr)})
            argdict.update(
                {'lr0': 'The initial learning rate: {0}'.format(self.lr0)})
        else:
            argdict.update({'forgetting_rate': 'Forgetting rate to control how quickly old '
                                               'information is forgotten: {0}'.format(self.forgetting_rate)})
            argdict.update({'delay_factor': 'Delay factor down weights '
                                            'early iterations: {0}'.format(self.delay_factor)})
        argdict.update(
            {'batch': 'Number of examples to use in each iteration: {0}'.format(self.batch)})
        argdict.update({'max_inner_iter': 'Number of inner loops inside an optimizer: {0}'.format(
            self.max_inner_iter)})
        argdict.update(
            {'num_epochs': 'Number of loops over training set: {0}'.format(self.num_epochs)})
        argdict.update(
            {'num_jobs': 'Number of parallel workers: {0}'.format(self.num_jobs)})
        argdict.update(
            {'display_interval': 'How often to evaluate? {0}'.format(self.display_interval)})
        argdict.update(
            {'shuffle': 'Shuffle the dataset? {0}'.format(self.shuffle)})
        argdict.update(
            {'random_state': 'The random number generator: {0}'.format(self.random_state)})
        argdict.update(
            {'log_path': 'Logs are stored in: {0}'.format(self.log_path)})

        args = list()
        for key, value in argdict.items():
            args.append(value)
        args = [str(item[0] + 1) + '. ' + item[1]
                for item in zip(list(range(len(args))), args)]
        args = '\n\t\t'.join(args)
        print('\t>> The following arguments are applied:\n\t\t{0}'.format(
            args), file=sys.stderr)
        logger.info(
            '\t>> The following arguments are applied:\n\t\t{0}'.format(args))

    def __shffule(self, num_samples):
        if self.shuffle:
            idx = np.arange(num_samples)
            np.random.shuffle(idx)
            return idx

    def __check_bounds(self, X):
        X = np.clip(X, LOWER_BOUND, UPPER_BOUND)
        if len(X.shape) > 1:
            if X.shape[0] == X.shape[1]:
                min_x = np.min(X) + EPSILON
                max_x = np.max(X) + EPSILON
                X = X - min_x
                X = X / (max_x - min_x)
                X = 2 * X - 1
        return X

    def __init_variables(self, num_samples):

        self.coef_label_input = np.zeros(
            shape=(self.num_models, self.num_labels, self.input_feature_size))
        self.intercept_label_input = np.zeros(
            shape=(self.num_models, self.num_labels, 1))
        self.coef_label_input_glob = np.zeros(
            shape=(self.num_labels, self.input_feature_size))
        self.intercept_label_input_glob = np.zeros(shape=(self.num_labels, 1))

        # initialize a similarity matrix
        if self.corr_label_sim or self.corr_input_sim:
            init_gamma = 100.
            init_var = 1. / init_gamma
            self.S = np.random.gamma(
                shape=init_gamma, scale=init_var, size=(num_samples, num_samples))
            np.fill_diagonal(self.S, 0)
            self.S = self.S / np.sum(self.S, axis=0)[:, np.newaxis]
            i_lower = np.tril_indices(num_samples, -1)
            self.S[i_lower] = self.S.T[i_lower]
            self.S = lil_matrix(self.S)

        if self.estimate_graph:
            tmp = self.cutting_point + self.num_labels
            self.pi = lil_matrix((tmp, tmp))

        # initialize a weighting for learning algorithm
        self.omega = np.repeat(a=1 / self.num_models, repeats=self.num_models)

    def __solver(self, X, y, coef, intercept, sample_weight=None):
        """Initialize logistic regression variables."""
        penalty = "elasticnet"
        if self.penalty != "elasticnet":
            penalty = "none"
        estimator = SGDClassifier(loss='log', penalty=penalty, alpha=self.alpha_elastic,
                                  l1_ratio=self.l1_ratio, fit_intercept=self.fit_intercept,
                                  max_iter=self.max_inner_iter, shuffle=self.shuffle,
                                  n_jobs=self.num_jobs, random_state=self.random_state,
                                  warm_start=True, average=True)
        estimator.fit(X=X, y=y, coef_init=coef,
                      intercept_init=intercept, sample_weight=sample_weight)
        return estimator.coef_[0], estimator.intercept_

    def __optimal_learning_rate(self, alpha):
        def _loss(p, y):
            z = p * y
            # approximately equal and saves the computation of the log
            if z > 18:
                return np.exp(-z)
            if z < -18:
                return -z
            return np.log(1.0 + np.exp(-z))

        typw = np.sqrt(1.0 / np.sqrt(alpha))
        # computing lr0, the initial learning rate
        initial_eta0 = typw / max(1.0, _loss(-typw, 1.0))
        # initialize t such that lr at first sample equals lr0
        optimal_init = 1.0 / (initial_eta0 * alpha)
        return optimal_init

    def __sigmoid(self, X):
        return expit(X)

    def __softmax(self, X, axis=None):
        return softmax(X, axis=axis)

    def __log_logistic(self, X, negative=True):
        param = 1
        if negative:
            param = -1
        X = np.clip(X, EPSILON, 1 - EPSILON)
        X = param * np.log(1 + np.exp(X))
        return X

    def __norm_l21(self, M):
        if M.size == 0:
            return 0.0
        if len(M.shape) == 2:
            ret = np.sum(np.power(M, 2), axis=1)
        else:
            ret = np.power(M, 2)
        ret = np.sum(np.sqrt(ret))
        return ret

    def __norm_elastic(self, M):
        if M.size == 0:
            return 0.0
        ret = self.l1_ratio * np.linalg.norm(M, 1)
        ret += (1 - self.l1_ratio) / 2 * np.square(np.linalg.norm(M))
        ret = ret * self.alpha_elastic
        return ret

    def __grad_l21_norm(self, M):
        if len(M.shape) == 2:
            D = 1 / (2 * np.linalg.norm(M, axis=1))
            ret = np.dot(np.diag(D), M)
        else:
            D = (2 * np.linalg.norm(M) + EPSILON)
            ret = M / D
        return ret

    def __fuse_label_weight(self, M, label_idx, model_idx):
        a_min = -1
        a_max = 1
        extract_idx = np.nonzero(M[label_idx])[0]
        L_coef = np.dot(self.coef_label_input[model_idx].T, M)
        L_coef = np.clip(L_coef, a_min=a_min, a_max=a_max)
        L_coef = L_coef[:, extract_idx]
        L_coef = np.divide(L_coef, 2)
        L_coef[L_coef == np.inf] = 0.
        L_coef[L_coef == -np.inf] = 0.
        np.nan_to_num(L_coef, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        L_coef = np.mean(L_coef, axis=1) / self.sigma
        return np.clip(L_coef, a_min=a_min, a_max=a_max)

    def __normalize_laplacian(self, A, return_adj=False, norm_adj=False):
        def __scale_diagonal(D):
            D = D.sqrt()
            D = D.power(-1)
            return D

        A.setdiag(values=0)
        D = lil_matrix(A.sum(axis=1))
        D = D.multiply(eye(D.shape[0]))
        if return_adj:
            if norm_adj:
                D_inv = D.power(-1)
                A = D_inv.dot(A)
            return A
        else:
            L = D - A
            D = __scale_diagonal(D=D) / self.sigma
            return D.dot(L.dot(D))

    def __preprocess_features(self, X):
        if self.binarize_input_feature:
            if self.use_external_features:
                X[:, :self.cutting_point] = preprocessing.binarize(
                    X[:, :self.cutting_point])
            else:
                X = preprocessing.binarize(X)
            X = lil_matrix(X)
        if self.normalize_input_feature:
            if self.use_external_features:
                X[:, :self.cutting_point] = preprocessing.normalize(
                    X[:, :self.cutting_point])
            else:
                X = preprocessing.normalize(X)
            X = lil_matrix(X)
        return X

    def __label_prob(self, X, labels, model_idx=0, transform=False, meta_predict=False):
        if len(labels) == 0:
            labels = np.arange(self.num_labels)
        if not meta_predict:
            coef_intercept = self.coef_label_input[model_idx][labels]
            if self.fit_intercept:
                X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
                coef_intercept = np.hstack(
                    (self.intercept_label_input[model_idx][labels], coef_intercept))
        else:
            coef_intercept = self.coef_label_input_glob[labels]
            if self.fit_intercept:
                X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
                coef_intercept = np.hstack(
                    (self.intercept_label_input_glob[labels], coef_intercept))
        prob_label = self.__sigmoid(np.dot(X, coef_intercept.T)) + EPSILON
        if not transform:
            prob_label = np.mean(prob_label, axis=0)
        return prob_label

    def __feed_forward(self, X, y, model_idx, batch_idx, current_progress, total_progress):
        X = X.toarray()
        y = y.toarray()
        num_samples = X.shape[0]

        num_labels_example = np.sum(y, axis=0)
        weight_labels = 1 / num_labels_example
        weight_labels[weight_labels == np.inf] = 0.0
        weight_labels = weight_labels / np.sum(weight_labels)
        labels = np.unique(np.where(y == 1)[1])
        if labels.shape[0] > self.subsample_labels_size:
            labels = np.random.choice(
                labels, self.subsample_labels_size, replace=False, p=weight_labels[labels])
            labels = np.sort(labels)

        # compute probability of labels
        prob = self.__label_prob(X=X, labels=labels, model_idx=model_idx, transform=True)

        # compute probability of labels
        tmp = np.zeros((num_samples, self.num_labels)) + EPSILON
        tmp[:, labels] = prob
        prob = tmp
        prob[np.where(y == 0)] = EPSILON
        desc = '\t\t\t--> Computed {0:.4f}%...'.format(
            ((current_progress + batch_idx) / total_progress * 100))
        print(desc, end="\r")
        return prob

    def __compute_loss(self, X, y, y_trust, trusted, model_idx, batch_idx, current_progress, total_progress):
        current_progress += batch_idx + 1
        desc = '\t\t\t--> Calculating discrepancy: {0:.2f}%...'.format(
            (current_progress / total_progress) * 100)
        print(desc, end="\r")

        prob_label = self.__label_prob(X=X.toarray(), labels=np.arange(self.num_labels), model_idx=model_idx,
                                       transform=False)
        if trusted:
            loss = np.mean(-(y_trust.toarray() * np.log(prob_label + LOG_EPSILON) + (1 - y_trust.toarray()) * np.log(
                1 - prob_label + LOG_EPSILON)),
                           axis=0)
        else:
            loss = np.mean(-(y.toarray() * np.log(prob_label + LOG_EPSILON) + (1 - y.toarray()) * np.log(
                1 - prob_label + LOG_EPSILON)), axis=0)
        return loss

    def __optimize_omega(self, discrepancy_table, learning_rate):
        omega = self.omega
        hist_omega = self.omega
        mean_discrepancy = discrepancy_table.mean(axis=1)
        old_cost = np.inf
        for idx in np.arange(self.max_inner_iter):
            desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...'.format(
                "Omega's parameters", 100)
            if idx + 1 == self.max_inner_iter:
                print(desc)
                logger.info(desc)
            else:
                print(desc, end="\r")

            tmp = self.beta_source * \
                  np.power(omega.T.dot(omega) / self.num_models, -
                  0.5) * (omega / self.num_models)
            omega = mean_discrepancy + tmp
            omega = hist_omega - learning_rate * omega
            new_cost = self.beta_source * \
                       np.sqrt(
                           np.sum(np.square(omega[:, np.newaxis]) * (1 / self.num_models)))
            new_cost += 2 * np.sum(omega * mean_discrepancy)
            hist_omega = self.__softmax(X=omega)
            if new_cost < old_cost:
                old_cost = new_cost
                self.omega = hist_omega

    def __optimize_theta_label(self, X, y, S, M, learning_rate, model_idx, batch_idx, current_progress, total_progress):
        X = X.toarray()
        y = y.toarray()

        if self.corr_input_sim:
            L = self.__normalize_laplacian(A=S)
            tmp = preprocessing.normalize(X=X)
            if self.fit_intercept:
                XtLX = np.dot(np.dot(tmp[:, 1:].T, L), tmp[:, 1:])
            else:
                XtLX = np.dot(np.dot(tmp.T, L), tmp)

        labels = np.arange(self.num_labels)
        if self.num_labels > self.subsample_labels_size:
            labels = np.random.choice(
                labels, self.subsample_labels_size, replace=False)
            labels = np.sort(labels)

        count = 1
        current_progress += batch_idx * (model_idx + 1) * len(labels)
        for label_idx in labels:
            desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...'.format("Label's parameters",
                                                                  ((current_progress + count) / total_progress) * 100)
            if total_progress == current_progress + count:
                print(desc)
            else:
                print(desc, end="\r")
            count += 1
            gradient = 0.0

            if self.apply_peer:
                tmp = self.omega[model_idx] + EPSILON_BERN
                M_1 = np.random.binomial(n=2, p=tmp, size=1)[0]
                if M_1 == 0:
                    if self.use_trusted:
                        tmp = self.labels_confidence[model_idx][label_idx] + \
                              EPSILON_BERN
                    else:
                        tmp = np.random.choice(2)
                    if tmp > 1:
                        tmp = 0.99
                    M_2 = np.random.binomial(n=2, p=tmp, size=1)[0]
                    if M_2 == 1:
                        # gradient of Theta^path = Theta^path_old + learning_type * gradient value of Theta^path
                        tmp = self.beta_peer * \
                              self.coef_label_input[model_idx][label_idx]
                        tmp += (1 - self.beta_peer) * \
                               self.coef_label_input_glob[label_idx]
                        # tmp += -1 * (self.coef_label_input_glob[label_idx] - self.coef_label_input[model_idx][label_idx])
                        # tmp = self.coef_label_input[model_idx][label_idx] - tmp * learning_rate
                        self.coef_label_input[model_idx][label_idx] = self.__check_bounds(
                            tmp)
                        if self.fit_intercept:
                            tmp = self.beta_peer * \
                                  self.intercept_label_input[model_idx][label_idx]
                            tmp += (1 - self.beta_peer) * \
                                   self.intercept_label_input_glob[label_idx]
                            # tmp += -1 * (self.intercept_label_input_glob[label_idx] - self.intercept_label_input[model_idx][label_idx])
                            # tmp = self.intercept_label_input[model_idx][label_idx] - tmp * learning_rate
                            self.intercept_label_input[model_idx][label_idx] = tmp
                        continue

            if len(np.unique(y[:, label_idx])) < 2:
                coef_intercept_label = self.coef_label_input[model_idx][label_idx]
                X_tmp = X
                if self.fit_intercept:
                    X_tmp = np.concatenate(
                        (np.ones((X.shape[0], 1)), X), axis=1)
                    coef_intercept_label = np.hstack(
                        (self.intercept_label_input[model_idx][label_idx], coef_intercept_label))
                cond = -(2 * y[:, label_idx] - 1)
                coef = np.dot(X_tmp, coef_intercept_label)
                coef = np.multiply(coef, cond)
                logit = 1 / (np.exp(-coef) + 1)
                coef = np.multiply(X_tmp, cond[:, np.newaxis])
                coef = np.multiply(coef, logit[:, np.newaxis])
                coef = np.mean(coef, axis=0)
                del logit, coef_intercept_label
                if self.fit_intercept:
                    self.coef_label_input[model_idx][label_idx] = self.coef_label_input[model_idx][
                                                                      label_idx] - learning_rate * coef[1:]
                    self.intercept_label_input[model_idx][label_idx] = coef[0]
                else:
                    self.coef_label_input[model_idx][label_idx] = self.coef_label_input[model_idx][
                                                                      label_idx] - learning_rate * coef
                if self.penalty != "l21":
                    l1 = self.l1_ratio * \
                         np.sign(self.coef_label_input[model_idx][label_idx])
                    l2 = (1 - self.l1_ratio) * 2 * \
                         self.coef_label_input[model_idx][label_idx]
                    if self.penalty == "elasticnet":
                        gradient += self.alpha_elastic * (l1 + l2)
                    if self.penalty == "l1":
                        gradient += self.alpha_elastic * l1
                    if self.penalty == "l2":
                        gradient += self.alpha_elastic * l2
            else:
                coef = np.reshape(self.coef_label_input[model_idx][label_idx],
                                  newshape=(1, self.coef_label_input[model_idx][label_idx].shape[0]))
                intercept = 0.0
                if self.fit_intercept:
                    intercept = self.intercept_label_input[model_idx][label_idx]
                coef, intercept = self.__solver(
                    X=X, y=y[:, label_idx], coef=coef, intercept=intercept)
                self.coef_label_input[model_idx][label_idx] = coef
                if self.fit_intercept:
                    self.intercept_label_input[model_idx][label_idx] = intercept

            if self.fuse_weight:
                gradient += self.__fuse_label_weight(
                    M=M, label_idx=label_idx, model_idx=model_idx)

            # compute the constraint X^T * L * X * Theta^path
            if self.corr_input_sim:
                gradient += np.dot(XtLX,
                                   self.coef_label_input[model_idx][label_idx].T)

            # compute the constraint lambda_5 * D_Theta^path * Theta^path
            if self.penalty == "l21":
                gradient += self.lam_5 * \
                            self.__grad_l21_norm(
                                M=self.coef_label_input[model_idx][label_idx])

            # gradient of Theta^path = Theta^path_old + learning_type * gradient value of Theta^path
            tmp = self.coef_label_input[model_idx][label_idx] - \
                  learning_rate * gradient
            self.coef_label_input[model_idx][label_idx] = self.__check_bounds(
                tmp)

    def __optimize_s(self, X, y, S, learning_rate, batch_idx, batch, current_progress, total_progress):
        desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...'.format("S", (
                (current_progress + batch_idx + 1) / total_progress) * 100)
        if (current_progress + batch_idx + 1) != total_progress:
            print(desc, end="\r")
        if (current_progress + batch_idx + 1) == total_progress:
            print(desc)
            logger.info(desc)

        def __func_jac_s():
            gradient = 0.0
            coef_label_input = np.mean(self.coef_label_input, axis=0)
            gradient += self.lam_4 * np.dot(y, y.T)
            gradient += np.dot(np.dot(np.dot(X, coef_label_input.T),
                                      coef_label_input), X.T)
            gradient += 2 * self.kappa * (S - 1)
            return gradient

        X = X.toarray()
        y = y.toarray()
        S = S.toarray()
        num_samples = X.shape[0]

        gradient = __func_jac_s()
        S = S - learning_rate * gradient
        S = S / np.sum(S, axis=1)
        S[S < 0] = 0
        np.fill_diagonal(S, 0)
        i_lower = np.tril_indices(num_samples, -1)
        S[i_lower] = S.T[i_lower]
        self.S[batch:batch + self.batch, batch:batch +
                                               self.batch] = lil_matrix(S) / self.num_models

    def __optimize_pi(self, X, y, batch_idx, current_progress, total_progress):
        desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...'.format("pi", (
                (current_progress + batch_idx + 1) / total_progress) * 100)
        if (current_progress + batch_idx + 1) != total_progress:
            print(desc, end="\r")
        if (current_progress + batch_idx + 1) == total_progress:
            print(desc)
            logger.info(desc)

        W = hstack((X, y))
        W = W.T.dot(W)
        pi = self.__normalize_laplacian(A=W, return_adj=True, norm_adj=True)
        self.pi = self.pi + lil_matrix(pi)
        self.pi = self.pi / self.pi.sum(1)
        self.pi = lil_matrix(self.pi)

    def __graph_construction(self, X, pi):
        X = lil_matrix(X)
        if self.num_neighbors > X.shape[0]:
            self.num_neighbors = X.shape[0]
        nbrs = NearestNeighbors(n_neighbors=self.num_neighbors, radius=1.0, algorithm='ball_tree',
                                leaf_size=30, metric='euclidean', n_jobs=self.num_jobs)
        nbrs.fit(X=X)
        N_x = nbrs.kneighbors_graph(X=X)
        W = lil_matrix(cosine_distances(X=X))
        W = W.multiply(N_x)
        D = np.linalg.inv(np.diag(np.array(W.sum(axis=1)).flatten()))
        W = lil_matrix(W.dot(D))
        del D, N_x, nbrs
        F = X[:, :self.cutting_point].dot(
            pi[:self.cutting_point, self.cutting_point:])
        F = F.dot(pi[self.cutting_point:, self.cutting_point:]) / 2
        np.nan_to_num(F / F.sum(0), copy=False,
                      nan=0.0, posinf=0.0, neginf=0.0)
        F = lil_matrix(F)
        return F, W

    def __optimize_partial_label(self, pi, W, F_init):
        F = F_init.dot(pi[self.cutting_point:, self.cutting_point:]) / 2
        np.nan_to_num(F / F.sum(0), copy=False,
                      nan=0.0, posinf=0.0, neginf=0.0)
        F_init = self.alpha_partial * F + (1 - self.alpha_partial) * F_init
        W = W.dot(F)
        min_tmp = np.min(W)
        max_tmp = np.max(W)
        W = W.toarray() - min_tmp
        W = W / (max_tmp - min_tmp)
        W[W >= 1 / (W.shape[1] * self.tau_partial)] = 1
        return lil_matrix(F_init), lil_matrix(W)

    def __optimize_global(self, X, y, X_trust, y_trust, pi, learning_rate, model_idx, batch_idx, current_progress,
                          total_progress):
        X_train = X.toarray()
        y_train = y.toarray()
        if self.use_trusted_partial_label:
            X_train = X_trust.toarray()
            y_train = y_trust.toarray()

        max_inner_iter = 1
        if self.apply_partial_label:
            max_inner_iter = self.max_inner_iter
            F, W = self.__graph_construction(X_train, pi)

        labels = np.arange(self.num_labels)
        if self.num_labels > self.subsample_labels_size:
            labels = np.random.choice(
                labels, self.subsample_labels_size, replace=False)
            labels = np.sort(labels)

        count = 1
        current_progress += batch_idx * \
                            (model_idx + 1) * len(labels) * max_inner_iter
        for i in np.arange(max_inner_iter):
            for label_idx in labels:
                desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...'.format("Global's parameters",
                                                                      ((
                                                                               current_progress + count) / total_progress) * 100)
                if total_progress == current_progress + count:
                    print(desc)
                else:
                    print(desc, end="\r")
                count += 1
                gradient = 0.0

                if len(np.unique(y_train[:, label_idx])) < 2:
                    X_tmp = X_train
                    coef_intercept_label = self.coef_label_input[model_idx][label_idx]
                    if self.fit_intercept:
                        X_tmp = np.concatenate(
                            (np.ones((X_train.shape[0], 1)), X_train), axis=1)
                        coef_intercept_label = np.hstack(
                            (self.intercept_label_input[model_idx][label_idx], coef_intercept_label))
                    cond = -(2 * y_train[:, label_idx] - 1)
                    coef = np.dot(X_tmp, coef_intercept_label)
                    coef = np.multiply(coef, cond)
                    logit = 1 / (np.exp(-coef) + 1)
                    coef = np.multiply(X_tmp, cond[:, np.newaxis])
                    coef = np.multiply(coef, logit[:, np.newaxis])
                    coef = np.mean(coef, axis=0)
                    del logit, coef_intercept_label
                    if self.fit_intercept:
                        if self.use_trusted:
                            self.coef_label_input_glob[label_idx] = self.coef_label_input_glob[
                                                                        label_idx] - learning_rate * coef[1:] * \
                                                                    self.labels_confidence[model_idx][
                                                                        label_idx]
                        else:
                            self.coef_label_input_glob[label_idx] = self.coef_label_input_glob[
                                                                        label_idx] - learning_rate * coef[1:]
                        self.intercept_label_input_glob[label_idx] = coef[0]
                    else:
                        if self.use_trusted:
                            self.coef_label_input_glob[label_idx] = self.coef_label_input_glob[
                                                                        label_idx] - learning_rate * coef * \
                                                                    self.labels_confidence[model_idx][
                                                                        label_idx]
                        else:
                            self.coef_label_input_glob[label_idx] = self.coef_label_input_glob[
                                                                        label_idx] - learning_rate * coef
                    if self.penalty != "l21":
                        l1 = self.l1_ratio * \
                             np.sign(self.coef_label_input_glob[label_idx])
                        l2 = (1 - self.l1_ratio) * 2 * \
                             self.coef_label_input_glob[label_idx]
                        if self.penalty == "elasticnet":
                            gradient += self.alpha_elastic * (l1 + l2)
                        if self.penalty == "l1":
                            gradient += self.alpha_elastic * l1
                        if self.penalty == "l2":
                            gradient += self.alpha_elastic * l2
                else:
                    coef = np.reshape(self.coef_label_input[model_idx][label_idx],
                                      newshape=(1, self.coef_label_input[model_idx][label_idx].shape[0]))
                    intercept = 0.0
                    if self.fit_intercept:
                        intercept = self.intercept_label_input[model_idx][label_idx]
                    coef, intercept = self.__solver(X=X_train, y=y_train[:, label_idx], coef=coef,
                                                    intercept=intercept)
                    self.coef_label_input_glob[label_idx] = coef
                    if self.fit_intercept:
                        self.intercept_label_input_glob[label_idx] = intercept

                dist_coef = -1 * self.alpha_glob * (
                        self.coef_label_input[model_idx][label_idx] - self.coef_label_input_glob[label_idx])
                if self.use_trusted:
                    gradient += dist_coef * self.omega[model_idx]
                else:
                    gradient += dist_coef / self.num_models

                if self.fit_intercept:
                    intercept = self.intercept_label_input_glob[label_idx] - self.intercept_label_input[model_idx][
                        label_idx]
                    if self.use_trusted:
                        intercept = intercept * self.omega[model_idx]
                    else:
                        intercept = intercept / self.num_models
                    self.intercept_label_input_glob[label_idx] = self.intercept_label_input_glob[
                                                                     label_idx] - learning_rate * intercept

                if self.penalty == "l21":
                    gradient += self.lam_5 * \
                                self.__grad_l21_norm(
                                    M=self.coef_label_input_glob[label_idx])

                gradient = self.coef_label_input_glob[label_idx] - \
                           learning_rate * gradient
                self.coef_label_input_glob[label_idx] = self.__check_bounds(
                    gradient)

            if self.apply_partial_label:
                if self.use_trusted_partial_label:
                    H = self.__label_prob(
                        X=X_trust.toarray(), labels=[], transform=True, meta_predict=True)
                else:
                    H = self.__label_prob(
                        X=X.toarray(), labels=[], transform=True, meta_predict=True)
                H = lil_matrix(H)
                F = F.multiply(H)
                F, tmp = self.__optimize_partial_label(pi=pi, W=W, F_init=F)
                tmp = y_train + tmp
                tmp[tmp >= 0.9] = 1
                tmp[tmp < 0.9] = 0
                y_train = np.array(tmp, dtype=np.int)

    def __batch_backward(self, X, y, X_trust, y_trust, pi, L, learning_rate, samples_idx,
                         subset_model_idx=False):
        print('  \t\t<<<------------<<<------------<<<')
        print('  \t\t>> Feed-Backward step...')
        logger.info('\t\t>> Feed-Backward step...')
        parallel = Parallel(n_jobs=self.num_jobs,
                            prefer="threads", verbose=max(0, self.verbose - 1))
        tmp_idx = np.unique(samples_idx)
        list_batches = np.arange(start=0, stop=len(tmp_idx), step=self.batch)

        # calculate discrepancy between models' outputs and trusted sources
        if self.use_trusted:
            current_progress = 0
            list_batches_trust = np.arange(
                start=0, stop=X_trust.shape[0], step=self.batch)
            total_progress = (len(list_batches) +
                              len(list_batches_trust)) * self.num_models
            discrepancy = np.zeros((self.num_models, self.num_labels))
            for model_idx in np.arange(self.num_models):
                tmp_idx = samples_idx
                if subset_model_idx:
                    tmp_idx = samples_idx[model_idx]

                X_tmp = X[tmp_idx]
                y_tmp = y[model_idx][tmp_idx]

                loss_1 = parallel(delayed(self.__compute_loss)(X_tmp[batch:batch + self.batch],
                                                               y_tmp[batch:batch +
                                                                           self.batch],
                                                               None, False, model_idx, batch_idx,
                                                               current_progress, total_progress)
                                  for batch_idx, batch in enumerate(list_batches))
                current_progress += len(list_batches)
                loss_2 = parallel(delayed(self.__compute_loss)(X_trust[batch:batch + self.batch], None,
                                                               y_trust[batch:batch +
                                                                             self.batch],
                                                               True, model_idx, batch_idx,
                                                               current_progress, total_progress)
                                  for batch_idx, batch in enumerate(list_batches_trust))
                current_progress += len(list_batches_trust)
                loss_1 = np.mean(np.vstack(loss_1), axis=0)
                loss_2 = np.mean(np.vstack(loss_2), axis=0)
                if self.apply_trusted_loss:
                    discrepancy[model_idx] = np.abs(
                        loss_1 - loss_2) + loss_2 + EPSILON
                else:
                    discrepancy[model_idx] = np.abs(loss_1 - loss_2) + EPSILON
            desc = '\t\t\t--> Calculating discrepancy: {0:.2f}%...'.format(100)
            logger.info(desc)
            print(desc)

            # optimize omega
            self.__optimize_omega(
                discrepancy_table=discrepancy, learning_rate=learning_rate)
            self.discrepancy_labels = discrepancy
            self.discrepancy_labels[self.discrepancy_labels > 1] = 1
            self.labels_confidence = 1 - discrepancy
            self.labels_confidence[self.labels_confidence <= 0] = EPSILON
            self.labels_confidence = self.__softmax(
                self.labels_confidence, axis=0)
            del discrepancy

        # optimize Theta^path
        current_progress = 0
        num_labels = self.num_labels
        if num_labels > self.subsample_labels_size:
            num_labels = self.subsample_labels_size
        for model_idx in np.arange(self.num_models):
            tmp_idx = samples_idx
            if subset_model_idx:
                tmp_idx = samples_idx[model_idx]
            subsamples_size = int(np.ceil(
                len(tmp_idx) * self.subsample_input_size * self.subsample_mini_input_size))
            if len(tmp_idx) >= subsamples_size:
                tmp_idx = np.random.choice(
                    a=tmp_idx, size=subsamples_size, replace=False)

            X_tmp = X[tmp_idx]
            y_tmp = y[model_idx][tmp_idx]
            if self.corr_label_sim or self.corr_input_sim:
                S = self.S[tmp_idx[:, None], tmp_idx]
            list_batches = np.arange(
                start=0, stop=len(tmp_idx), step=self.batch)
            total_progress = len(list_batches) * self.num_models * num_labels
            if self.corr_label_sim or self.corr_input_sim:
                parallel(delayed(self.__optimize_theta_label)(X_tmp[batch:batch + self.batch],
                                                              y_tmp[batch:batch +
                                                                          self.batch],
                                                              S[batch:batch + self.batch,
                                                              batch:batch + self.batch],
                                                              L, learning_rate, model_idx,
                                                              batch_idx, current_progress, total_progress)
                         for batch_idx, batch in enumerate(list_batches))
            else:
                parallel(delayed(self.__optimize_theta_label)(X_tmp[batch:batch + self.batch],
                                                              y_tmp[batch:batch +
                                                                          self.batch],
                                                              None, L, learning_rate, model_idx,
                                                              batch_idx, current_progress, total_progress)
                         for batch_idx, batch in enumerate(list_batches))
            current_progress = len(list_batches) * (model_idx + 1) * num_labels

        # optimize pi
        if self.estimate_graph:
            current_progress = 0
            for model_idx in np.arange(self.num_models):
                if self.use_trusted_partial_label:
                    subsamples_size = int(
                        np.ceil(X_trust.shape[0] * self.subsample_input_size))
                    tmp_idx = np.arange(X_trust.shape[0])
                    if len(tmp_idx) >= subsamples_size:
                        tmp_idx = np.random.choice(
                            a=tmp_idx, size=subsamples_size, replace=False)
                    X_tmp = X_trust[tmp_idx, :self.cutting_point]
                    y_tmp = y_trust[tmp_idx]
                else:
                    tmp_idx = samples_idx
                    if subset_model_idx:
                        tmp_idx = samples_idx[model_idx]
                    subsamples_size = int(
                        np.ceil(len(tmp_idx) / self.num_models * self.subsample_input_size))
                    if len(tmp_idx) >= subsamples_size:
                        tmp_idx = np.random.choice(
                            a=tmp_idx, size=subsamples_size, replace=False)
                    X_tmp = X[tmp_idx, :self.cutting_point]
                    y_tmp = y[model_idx][tmp_idx]
                list_batches = np.arange(
                    start=0, stop=len(tmp_idx), step=self.batch)
                total_progress = len(list_batches) * self.num_models
                parallel(delayed(self.__optimize_pi)(X_tmp[batch:batch + self.batch],
                                                     y_tmp[batch:batch +
                                                                 self.batch],
                                                     batch_idx, current_progress,
                                                     total_progress)
                         for batch_idx, batch in enumerate(list_batches))
                current_progress = len(list_batches) * (model_idx + 1)

        # optimize global parameters
        current_progress = 0
        num_labels = self.num_labels
        if num_labels > self.subsample_labels_size:
            num_labels = self.subsample_labels_size
        max_inner_iter = 1
        if self.apply_partial_label:
            max_inner_iter = self.max_inner_iter

        for model_idx in np.arange(self.num_models):
            if subset_model_idx:
                tmp_idx = samples_idx[model_idx]
            else:
                tmp_idx = samples_idx

            subsamples_size = int(
                np.ceil(len(tmp_idx) * self.subsample_input_size))
            if len(tmp_idx) >= subsamples_size:
                tmp_idx = np.random.choice(
                    a=tmp_idx, size=subsamples_size, replace=False)
            X_tmp = X[tmp_idx]
            y_tmp = y[model_idx][tmp_idx]

            list_batches = np.arange(
                start=0, stop=len(tmp_idx), step=self.batch)
            total_progress = len(list_batches) * \
                             self.num_models * num_labels * max_inner_iter

            if self.apply_partial_label and self.use_trusted_partial_label:
                if self.estimate_graph:
                    pi = self.pi
                subsamples_size = int(
                    np.ceil(X_trust.shape[0] * self.subsample_input_size))
                tmp_trusted_idx = np.arange(X_trust.shape[0])
                if len(tmp_trusted_idx) >= subsamples_size:
                    tmp_trusted_idx = np.random.choice(
                        a=tmp_trusted_idx, size=subsamples_size, replace=False)
                X_trust_tmp = X_trust[tmp_trusted_idx]
                y_trust_tmp = y_trust[tmp_trusted_idx]
                parallel(delayed(self.__optimize_global)(X_tmp[batch:batch + self.batch],
                                                         y_tmp[batch:batch +
                                                                     self.batch],
                                                         X_trust_tmp, y_trust_tmp,
                                                         pi, learning_rate, model_idx, batch_idx,
                                                         current_progress, total_progress)
                         for batch_idx, batch in enumerate(list_batches))
            else:
                parallel(delayed(self.__optimize_global)(X_tmp[batch:batch + self.batch],
                                                         y_tmp[batch:batch +
                                                                     self.batch],
                                                         None, None, pi, learning_rate,
                                                         model_idx, batch_idx, current_progress,
                                                         total_progress)
                         for batch_idx, batch in enumerate(list_batches))
            current_progress = len(list_batches) * \
                               (model_idx + 1) * num_labels * max_inner_iter

        # optimize S
        if self.corr_label_sim or self.corr_input_sim:
            current_progress = 0
            for model_idx in np.arange(self.num_models):
                if subset_model_idx:
                    tmp_idx = samples_idx[model_idx]
                else:
                    tmp_idx = samples_idx

                subsamples_size = int(
                    np.ceil(len(tmp_idx) / self.num_models * self.subsample_input_size))
                if len(tmp_idx) >= subsamples_size:
                    tmp_idx = np.random.choice(
                        a=tmp_idx, size=subsamples_size, replace=False)
                X_tmp = X[tmp_idx]
                y_tmp = y[model_idx][tmp_idx]
                if self.corr_label_sim or self.corr_input_sim:
                    S = self.S[tmp_idx[:, None], tmp_idx]

                list_batches = np.arange(
                    start=0, stop=len(tmp_idx), step=self.batch)
                total_progress = len(list_batches) * self.num_models
                parallel(delayed(self.__optimize_s)(X_tmp[batch:batch + self.batch],
                                                    y_tmp[batch:batch + self.batch],
                                                    S[batch:batch + self.batch,
                                                    batch:batch + self.batch],
                                                    learning_rate, batch_idx, batch, current_progress,
                                                    total_progress)
                         for batch_idx, batch in enumerate(list_batches))
                current_progress = len(list_batches) * (model_idx + 1)
            print("\n", end="\r")

    def __cost_label(self, X, y, s_cost_x, model_idx, label_idx, use_glob, current_progress, total_progress):
        desc = '\t\t\t--> Calculating {0} cost: {1:.2f}%...'.format('label', (
                ((current_progress + label_idx + 1) / total_progress) * 100))
        print(desc, end="\r")
        if use_glob:
            coef_intercept_label = self.coef_label_input_glob[label_idx]
            if self.fit_intercept:
                coef_intercept_label = np.hstack(
                    (self.intercept_label_input_glob[label_idx], coef_intercept_label))
        else:
            coef_intercept_label = self.coef_label_input[model_idx][label_idx]
            if self.fit_intercept:
                coef_intercept_label = np.hstack(
                    (self.intercept_label_input[model_idx][label_idx], coef_intercept_label))
        cond = -(2 * y[:, label_idx] - 1)
        coef = np.dot(X, coef_intercept_label)
        coef = np.multiply(coef, cond)
        cost_label = -np.mean(self.__log_logistic(coef))
        if not use_glob:
            if self.calc_total_cost:
                # cost 1/2 * S_q,k ||Theta^path X_q - Theta^path X_k||_2^2
                if self.corr_input_sim:
                    cost_label += s_cost_x[label_idx]
                M = self.coef_label_input[model_idx][label_idx]
                if self.use_trusted:
                    M = self.coef_label_input_glob[label_idx]
                if self.penalty == "l21":
                    # ||Theta^path||_2^2
                    cost_label += self.lam_5 * self.__norm_l21(M=M)
                else:
                    cost_label = self.__norm_elastic(M=M)
        return cost_label

    def __total_cost(self, X, y, S, sample_idx):
        print('  \t\t>> Compute cost...')
        logger.info('\t\t>> Compute cost...')

        # hyper-parameters
        s_cost = 0.0
        s_cost_x = 0.0
        s_cost_y = 0.0
        u_cost = 0.0
        cost_label = 0.0
        cost_bag = 0.0

        # properties of dataset
        num_samples = X.shape[0]
        X = X.toarray()
        parallel = Parallel(n_jobs=self.num_jobs, prefer="threads",
                            verbose=max(0, self.verbose - 1))

        if self.fit_intercept:
            X = np.concatenate((np.ones((num_samples, 1)), X), axis=1)

        if self.corr_label_sim or self.corr_input_sim:
            L = self.__normalize_laplacian(S)

        if self.calc_total_cost:
            # cost S
            if self.corr_label_sim or self.corr_input_sim:
                S = S.toarray()
                s_cost = np.dot(S, np.ones((S.shape[0], 1)))
                s_cost = np.sum(s_cost - 1, axis=1)
                s_cost = self.kappa * np.linalg.norm(s_cost)

        # estimate expected cost over all labels
        if self.calc_label_cost:
            # if self.use_trusted
            current_progress = 0
            total_progress = self.num_models * self.num_labels
            for model_idx in np.arange(self.num_models):
                y_tmp = y[model_idx][sample_idx].toarray()
                if self.calc_total_cost:
                    # cost (lambda_4 / 2) * S_q,k ||y_q - y_k||_2^2
                    if self.corr_label_sim:
                        s_cost_y += self.lam_4 * \
                                    np.trace(np.dot(np.dot(y_tmp.T, L), y_tmp))

                        # cost 1/2 * S_q,k ||Theta^path X_q - Theta^path X_k||_2^2
                    if self.corr_input_sim:
                        tmp = np.dot(
                            X[:, 1:], self.coef_label_input[model_idx].T)
                        s_cost_x = np.diag(np.dot(np.dot(tmp.T, L), tmp))
                        del tmp
                results = parallel(
                    delayed(self.__cost_label)(X, y_tmp, s_cost_x, model_idx, label_idx, False, current_progress,
                                               total_progress)
                    for label_idx in np.arange(self.num_labels))
                cost_label += np.mean(results)
                current_progress = (model_idx + 1) * self.num_labels
            del results

            current_progress = 0
            total_progress = self.num_models * self.num_labels
            for model_idx in np.arange(self.num_models):
                y_tmp = y[model_idx][sample_idx].toarray()
                results = parallel(
                    delayed(self.__cost_label)(X, y_tmp, s_cost_x, model_idx, label_idx, True, current_progress,
                                               total_progress)
                    for label_idx in np.arange(self.num_labels))
                cost_label += np.mean(results)
                current_progress = (model_idx + 1) * self.num_labels
            del results

        cost = cost_bag + cost_label + u_cost + s_cost + s_cost_y + EPSILON
        cost /= self.num_models
        print("")
        return cost

    def fit(self, X, y_dict, X_trust=None, y_trust=None, pi=None, A=None, model_name='mltS', model_path="../../model",
            result_path=".",
            display_params: bool = True):
        if X is None:
            raise Exception("Please provide a dataset.")
        if y_dict is None:
            raise Exception("Please provide labels for the dataset.")
        if self.fuse_weight:
            if A is None:
                raise Exception(
                    "Please provide a similarity matrix over labels.")
        self.estimate_graph = False
        if self.apply_partial_label:
            if pi is None:
                self.estimate_graph = True

        self.use_ensemble = True
        self.use_trusted = False
        if X_trust is not None:
            if y_trust is None:
                raise Exception(
                    "Please provide trusted labels for the dataset.")
            assert X.shape[1] == X_trust.shape[1]
            assert X_trust.shape[0] == y_trust.shape[0]
            y_trust[y_trust == - 1] = 0
            self.use_trusted = True
            self.use_ensemble = False

        # collect properties from data
        self.input_feature_size = X.shape[1]
        y = list()
        idx_name = list()
        self.num_labels = y_dict[list(y_dict.keys())[0]].shape[1]
        self.source_names = list(y_dict.keys())
        if len(y_dict) == 1:
            y_dict = y_dict[self.source_names[0]]
            assert X.shape[0] == y_dict.shape[0]
            y_dict[y_dict == -1] = 0
            for idx in np.arange(self.num_models):
                idx_name.append(self.source_names[0] + '_' + str(idx + 1))
                y.append(y_dict)
        else:
            self.num_models = len(y_dict)
            for idx, item in y_dict.items():
                idx_name.append(idx)
                item[item == -1] = 0
                y.append(item)
                assert self.num_labels == item.shape[1]
                assert X.shape[0] == item.shape[0]
        del y_dict

        # preprocessing features
        X = self.__preprocess_features(X)
        if self.use_trusted:
            X_trust = self.__preprocess_features(X_trust)

        if display_params:
            self.__print_arguments(
                use_trusted='Use trusted sources: {0}'.format(
                    self.use_trusted),
                use_ensemble='Use ensemble based approach: {0}'.format(
                    self.use_ensemble),
                num_models='Number of learning algorithms: {0}'.format(
                    self.num_models),
                num_labels='Number of labels: {0}'.format(self.num_labels))
            time.sleep(2)

        cost_file_name = model_name + "_cost.txt"
        save_data('', file_name=cost_file_name, save_path=result_path,
                  mode='w', w_string=True, print_tag=False)

        if self.learning_type == "optimal":
            optimal_init = self.__optimal_learning_rate(alpha=self.lr)

        L = None
        if self.fuse_weight:
            L = self.__normalize_laplacian(A=A)

        n_epochs = self.num_epochs + 1
        num_samples = X.shape[0]
        self.__init_variables(num_samples=num_samples)

        print('\t>> Training mltS model...')
        logger.info('\t>> Training mltS model...')
        selected_samples = list()
        model_sample_idx = list()
        old_cost = np.inf
        high_cost = 0.0
        timeref = time.time()

        for epoch in np.arange(start=1, stop=n_epochs):
            desc = '\t   {0:d})- Epoch count ({0:d}/{1:d})...'.format(
                epoch, n_epochs - 1)
            print(desc)
            logger.info(desc)

            # shuffle dataset
            if len(selected_samples) == 0:
                sample_idx = self.__shffule(num_samples=num_samples)
                X = X[sample_idx, :]
                for idx, tmp in enumerate(y):
                    y[idx] = y[idx][sample_idx, :]
                del tmp

            if self.learning_type == "optimal":
                # usual optimization technique
                learning_rate = 1.0 / (self.lr * (optimal_init + epoch - 1))
            else:
                # using variational inference sgd
                learning_rate = np.power((epoch + self.delay_factor), -self.forgetting_rate)

            # set epoch time
            start_epoch = time.time()

            selected_samples = np.arange(num_samples)
            self.__batch_backward(X=X, y=y, X_trust=X_trust, y_trust=y_trust, pi=pi, L=L,
                                  learning_rate=learning_rate, samples_idx=selected_samples)

            end_epoch = time.time()
            self.is_fit = True

            # compute loss
            # report loss from global parameter too
            ss_cost = selected_samples
            if self.cost_subsample_size < len(ss_cost):
                ss_cost = np.random.choice(
                    selected_samples, self.cost_subsample_size, replace=False)

            S = None
            if self.corr_label_sim or self.corr_input_sim:
                S = self.S[ss_cost[:, None], ss_cost]
            new_cost = self.__total_cost(X=X[ss_cost], y=y, S=S, sample_idx=ss_cost)

            print('\t\t  ## Epoch {0} took {1} seconds...'.format(
                epoch, round(end_epoch - start_epoch, 3)))
            logger.info('\t\t  ## Epoch {0} took {1} seconds...'.format(
                epoch, round(end_epoch - start_epoch, 3)))
            data = str(epoch) + '\t' + str(round(end_epoch -
                                                 start_epoch, 3)) + '\t' + str(new_cost) + '\n'
            save_data(data=data, file_name=cost_file_name, save_path=result_path, mode='a', w_string=True,
                      print_tag=False)

            # Save models parameters based on test frequencies
            if (epoch % self.display_interval) == 0 or epoch == n_epochs - 1:
                print(
                    '\t\t  --> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_cost, old_cost))
                logger.info(
                    '\t\t  --> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_cost, old_cost))
                if old_cost >= new_cost or epoch == n_epochs - 1:
                    S_name = model_name + '_S.pkl'
                    pi_name = model_name + '_pi.pkl'
                    omega_name = model_name + '_omega.pkl'
                    discrepancy_table_name = model_name + '_disc.pkl'
                    confidence_table_name = model_name + '_confidence.pkl'
                    model_file_name = model_name + '.pkl'

                    if self.corr_label_sim or self.corr_input_sim:
                        print(
                            '\t\t  --> Storing the mltS\'s S parameters to: {0:s}'.format(S_name))
                        logger.info(
                            '\t\t  --> Storing the mltS\'s U parameters to: {0:s}'.format(S_name))
                        if old_cost >= new_cost:
                            save_data(data=lil_matrix(self.S), file_name=S_name, save_path=model_path,
                                      mode="wb", print_tag=False)
                        if epoch == n_epochs - 1:
                            S_name = model_name + '_S_final.pkl'
                            print(
                                '\t\t  --> Storing the mltS\'s S parameters to: {0:s}'.format(S_name))
                            logger.info(
                                '\t\t  --> Storing the mltS\'s U parameters to: {0:s}'.format(S_name))
                            save_data(data=lil_matrix(self.S), file_name=S_name, save_path=model_path,
                                      mode="wb", print_tag=False)
                        self.S = None

                    if self.estimate_graph:
                        print(
                            '\t\t  --> Storing the mltS\'s pi parameters to: {0:s}'.format(pi_name))
                        logger.info(
                            '\t\t  --> Storing the mltS\'s pi parameters to: {0:s}'.format(pi_name))
                        if old_cost >= new_cost:
                            save_data(data=lil_matrix(self.pi), file_name=pi_name, save_path=model_path,
                                      mode="wb", print_tag=False)
                        if epoch == n_epochs - 1:
                            pi_name = model_name + '_pi_final.pkl'
                            print(
                                '\t\t  --> Storing the mltS\'s pi parameters to: {0:s}'.format(pi_name))
                            logger.info(
                                '\t\t  --> Storing the mltS\'s pi parameters to: {0:s}'.format(pi_name))
                            save_data(data=lil_matrix(self.pi), file_name=pi_name, save_path=model_path,
                                      mode="wb", print_tag=False)
                        self.pi = None

                    if old_cost >= new_cost:
                        if self.use_trusted:
                            print(
                                '\t\t  --> Storing the source specific weights to: {0:s}'.format(omega_name))
                            logger.info(
                                '\t\t  --> Storing the source specific weights to: {0:s}'.format(omega_name))
                            save_data(data=lil_matrix(self.omega), file_name=omega_name, save_path=model_path,
                                      mode="wb", print_tag=False)

                            print(
                                '\t\t  --> Storing the discrepancy table to: {0:s}'.format(discrepancy_table_name))
                            logger.info(
                                '\t\t  --> Storing the discrepancy table to: {0:s}'.format(discrepancy_table_name))
                            save_data(data=lil_matrix(self.discrepancy_labels),
                                      file_name=discrepancy_table_name, save_path=model_path, mode="wb",
                                      print_tag=False)

                            print(
                                '\t\t  --> Storing the confidence table to: {0:s}'.format(confidence_table_name))
                            logger.info(
                                '\t\t  --> Storing the confidence table to: {0:s}'.format(confidence_table_name))
                            save_data(data=lil_matrix(self.labels_confidence),
                                      file_name=confidence_table_name, save_path=model_path, mode="wb",
                                      print_tag=False)

                        print(
                            '\t\t  --> Storing the mltS model to: {0:s}'.format(model_file_name))
                        logger.info(
                            '\t\t  --> Storing the mltS model to: {0:s}'.format(model_file_name))
                        save_data(data=copy.copy(self), file_name=model_file_name, save_path=model_path, mode="wb",
                                  print_tag=False)
                    if epoch == n_epochs - 1:
                        model_file_name = model_name + '_final.pkl'
                        omega_name = model_name + '_omega_final.pkl'
                        discrepancy_table_name = model_name + '_disc_final.pkl'
                        confidence_table_name = model_name + '_confidence_final.pkl'

                        if self.use_trusted:
                            print(
                                '\t\t  --> Storing the source specific weights to: {0:s}'.format(omega_name))
                            logger.info(
                                '\t\t  --> Storing the source specific weights to: {0:s}'.format(omega_name))
                            save_data(data=lil_matrix(self.omega), file_name=omega_name, save_path=model_path,
                                      mode="wb", print_tag=False)

                            print(
                                '\t\t  --> Storing the discrepancy table to: {0:s}'.format(discrepancy_table_name))
                            logger.info(
                                '\t\t  --> Storing the discrepancy table to: {0:s}'.format(discrepancy_table_name))
                            save_data(data=lil_matrix(self.discrepancy_labels),
                                      file_name=discrepancy_table_name, save_path=model_path, mode="wb",
                                      print_tag=False)

                            print(
                                '\t\t  --> Storing the confidence table to: {0:s}'.format(confidence_table_name))
                            logger.info(
                                '\t\t  --> Storing the confidence table to: {0:s}'.format(confidence_table_name))
                            save_data(data=lil_matrix(self.labels_confidence),
                                      file_name=confidence_table_name, save_path=model_path, mode="wb",
                                      print_tag=False)

                        print(
                            '\t\t  --> Storing the mltS model to: {0:s}'.format(model_file_name))
                        logger.info(
                            '\t\t  --> Storing the mltS model to: {0:s}'.format(model_file_name))
                        save_data(data=copy.copy(self), file_name=model_file_name, save_path=model_path, mode="wb",
                                  print_tag=False)

                    if epoch != n_epochs - 1:
                        if self.corr_label_sim or self.corr_input_sim:
                            self.S = load_data(
                                file_name=S_name, load_path=model_path, tag="mltS\'s S parameters")
                        if self.estimate_graph:
                            self.pi = load_data(
                                file_name=pi_name, load_path=model_path, tag="mltS\'s pi parameters")

                    if self.early_stop:
                        relative_change = np.abs(
                            (new_cost - old_cost) / old_cost)
                        desc = '\t\t  --> There is a little improvement in the cost '
                        desc += '(< {0}) for epoch {1}, hence, training is terminated...'.format(self.loss_threshold,
                                                                                                 epoch)
                        if relative_change < self.loss_threshold:
                            print(desc)
                            logger.info(desc)
                            break
                    old_cost = new_cost
        print('\t  --> Training consumed %.2f mintues' %
              (round((time.time() - timeref) / 60., 3)))
        logger.info('\t  --> Training consumed %.2f mintues' %
                    (round((time.time() - timeref) / 60., 3)))

    def __predict(self, X, model_idx, batch_idx, meta_predict, source_model_idx, current_progress, total_progress):

        X = X.toarray()
        if source_model_idx != -1:
            model_idx = source_model_idx
        prob_label = self.__label_prob(X=X, labels=list(), model_idx=model_idx, transform=True,
                                       meta_predict=meta_predict)
        desc = '\t\t--> Computed {0:.4f}%...'.format(
            ((current_progress + batch_idx) / total_progress * 100))
        print(desc, end="\r")
        return prob_label

    def __batch_predict(self, X, meta_predict=True, source_model_idx=-1, list_batches=None):
        num_models = self.num_models
        if meta_predict or source_model_idx != -1:
            num_models = 1
        prob_label_pred = np.zeros(
            (num_models, X.shape[0], self.num_labels)) + EPSILON
        current_progress = 0
        total_progress = num_models * len(list_batches)
        parallel = Parallel(n_jobs=self.num_jobs, prefer="threads",
                            verbose=max(0, self.verbose - 1))
        for model_idx in np.arange(num_models):
            results = parallel(delayed(self.__predict)(X[batch:batch + self.batch],
                                                       model_idx, batch_idx,
                                                       meta_predict, source_model_idx,
                                                       current_progress,
                                                       total_progress)
                               for batch_idx, batch in enumerate(list_batches))
            current_progress = (model_idx + 1) * len(list_batches)

            # merge result
            prob_label = results
            prob_label_pred[model_idx] = np.vstack(prob_label)
            del prob_label

        desc = '\t\t--> Computed {0:.4f}%...'.format(100)
        logger.info(desc)
        print(desc)

        if self.meta_adaptive:
            prob_label_pred[prob_label_pred >= self.decision_threshold] = 1
            prob_label_pred[prob_label_pred != 1] = 0
            if source_model_idx == -1:
                prob_label_pred = np.multiply(
                    prob_label_pred, self.labels_confidence[:, np.newaxis])
            prob_label = np.sum(prob_label_pred, axis=0)
        elif self.meta_omega:
            prob_label_pred[prob_label_pred >= self.decision_threshold] = 1
            prob_label_pred[prob_label_pred != 1] = 0
            if source_model_idx == -1:
                prob_label_pred = np.multiply(prob_label_pred, self.omega[:, np.newaxis, np.newaxis])
            prob_label = np.sum(prob_label_pred, axis=0)
        elif self.soft_voting:
            prob_label = np.mean(prob_label_pred, axis=0)
        else:
            prob_label = np.max(prob_label_pred, axis=0)

        return prob_label

    def predict(self, X, estimate_prob=False, meta_predict=True, meta_adaptive=True, meta_omega=False,
                pref_model=None, pref_rank=False, top_k_rank=500, subsample_labels_size=10, soft_voting=False,
                apply_t_criterion=False, adaptive_beta=0.45, decision_threshold=0.5, batch_size=30, num_jobs=1):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")

        desc = '\t>> Predicting using mltS model...'
        print(desc)
        logger.info(desc)

        self.batch = batch_size
        self.decision_threshold = decision_threshold
        self.num_jobs = num_jobs
        self.subsample_labels_size = subsample_labels_size
        self.soft_voting = soft_voting
        self.meta_adaptive = meta_adaptive
        self.meta_omega = meta_omega
        self.pref_rank = pref_rank
        if pref_rank:
            self.soft_voting = True
        if meta_predict:
            self.meta_adaptive = False
            self.meta_omega = False
            self.soft_voting = False
            pref_model = None
        if not self.use_trusted:
            self.meta_adaptive = False
            self.meta_omega = False
        source_model_idx = -1
        if pref_model is not None:
            if pref_model in self.source_names:
                source_model_idx = self.source_names.index(pref_model)
        if batch_size < 0:
            self.batch = 30
        if decision_threshold < 0:
            self.decision_threshold = 0.5
        if num_jobs < 0:
            self.num_jobs = 1
        if apply_t_criterion:
            estimate_prob = False

        if self.binarize_input_feature:
            if self.use_external_features:
                X[:, :self.cutting_point] = preprocessing.binarize(
                    X[:, :self.cutting_point])
            else:
                X = preprocessing.binarize(X)
            X = lil_matrix(X)
        if self.normalize_input_feature:
            if self.use_external_features:
                X[:, :self.cutting_point] = preprocessing.normalize(
                    X[:, :self.cutting_point])
            else:
                X = preprocessing.normalize(X)
            X = lil_matrix(X)

        num_samples = X.shape[0]
        list_batches = np.arange(start=0, stop=num_samples, step=self.batch)
        prob_label = self.__batch_predict(X=X, meta_predict=meta_predict, source_model_idx=source_model_idx,
                                          list_batches=list_batches)

        if apply_t_criterion and not pref_rank:
            maxval = np.max(prob_label, axis=1) * adaptive_beta
            for sidx in np.arange(prob_label.shape[0]):
                prob_label[sidx][prob_label[sidx] >= maxval[sidx]] = 1

        if not estimate_prob:
            if pref_rank:
                labels_idx = np.argsort(-prob_label)[:, :top_k_rank]
                for idx in np.arange(prob_label.shape[0]):
                    prob_label[idx, labels_idx[idx]] = 1
            else:
                prob_label[prob_label >= self.decision_threshold] = 1
            prob_label[prob_label != 1] = 0
        return lil_matrix(prob_bag), lil_matrix(prob_label)

    def predict_partial_labels(self, X, y, pi, tau_partial=0.005, build_up=True, meta_predict=True, pref_model=None,
                               soft_voting=False, decision_threshold=0.5, batch_size=30, num_jobs=1):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")

        desc = '\t>> Fill labels using mltS model...'
        print(desc)
        logger.info(desc)

        self.batch = batch_size
        self.decision_threshold = decision_threshold
        self.num_jobs = num_jobs
        self.soft_voting = soft_voting
        if meta_predict:
            self.soft_voting = False
            pref_model = None
        source_model_idx = -1
        if pref_model is not None:
            if pref_model in self.source_names:
                source_model_idx = self.source_names.index(pref_model)
        if batch_size < 0:
            self.batch = 30
        if decision_threshold < 0:
            self.decision_threshold = 0.5
        if num_jobs < 0:
            self.num_jobs = 1

        if self.binarize_input_feature:
            if self.use_external_features:
                X[:, :self.cutting_point] = preprocessing.binarize(
                    X[:, :self.cutting_point])
            else:
                X = preprocessing.binarize(X)
            X = lil_matrix(X)
        if self.normalize_input_feature:
            if self.use_external_features:
                X[:, :self.cutting_point] = preprocessing.normalize(
                    X[:, :self.cutting_point])
            else:
                X = preprocessing.normalize(X)
            X = lil_matrix(X)

        num_samples = X.shape[0]
        list_batches = np.arange(start=0, stop=num_samples, step=self.batch)
        prob_label = self.__batch_predict(X=X, meta_predict=meta_predict, source_model_idx=source_model_idx,
                                          list_batches=list_batches)
        prob_label = lil_matrix(prob_label)
        F, W = self.__graph_construction(X, pi)
        F = F.multiply(prob_label)
        F, tmp = self.__optimize_partial_label(pi=pi, W=W, F_init=F)
        y = y + tmp
        y[y >= 1] = 1
        y[y < 1] = 0
        y = np.array(y, dtype=np.int)
        return lil_matrix(y)
