import os, sys

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../"))


import utilities.experimentutils as exputil
import utilities.datautils as datautil
import utilities.configutils as cfgutil

import numpy as np
from collections import defaultdict, deque
from sklearn.datasets import base as bunch
from sklearn.externals.joblib import Parallel, delayed, logger
from time import time

from experiment.base import Experiment

# import multiprocessing as mp
import dill
import copy_reg
# import types

# from pathos.multiprocessing import ProcessingPool as Pool


__author__ = "mramire8"


class ExperimentJobs(Experiment):

    def __init__(self, config, verbose=False, debug=False):
        super(ExperimentJobs, self).__init__(config, verbose=verbose, debug=debug)
        self.save_all = True
        self.validation_set = None

    def _setup_options(self, config_obj):

        super(ExperimentJobs,self)._setup_options(config_obj)

        # experiment related config
        config = cfgutil.get_section_options(config_obj, 'experiment')
        self.validation_set = config['validation_set']

    def vectorize(self, data):
        def sentence_iterator(data):
            for doc in data:
                for snip in doc:
                    yield snip


        data.train.bow = self.vct.fit_transform(data.train.data)
        data.test.bow = self.vct.transform(data.test.data)

        #Processing train
        snippets = self.sent_tokenizer.tokenize_sents(data.train.data)
        snippet_bow = self.vct.transform(sentence_iterator(snippets))
        sizes = np.array([len(snip) for snip in snippets])
        cost = np.array([[self.costfn(si, cost_model=self.cost_model) for si in s] for s in snippets])

        ranges = np.cumsum(sizes)
        data.train.snippets = [snippet_bow[0 if i == 0 else ranges[i-1]:ranges[i]] for i in range(len(sizes))]
        data.train.sizes= sizes
        data.train.snippet_cost = cost

        # Processing tests
        snippets = self.sent_tokenizer.tokenize_sents(data.test.data)
        snippet_bow = self.vct.transform(sentence_iterator(snippets))
        sizes = np.array([len(snip) for snip in snippets])
        cost = np.array([[self.costfn(si, cost_model=self.cost_model) for si in s] for s in snippets])

        ranges = np.cumsum(sizes)
        data.test.snippets = [snippet_bow[0 if i == 0 else ranges[i-1]:ranges[i]] for i in range(len(sizes))]
        data.test.sizes= sizes
        data.test.snippet_cost = cost


        data.train.data = None
        data.train.data = None
        self.vct = None

        return data

    def bootstrap(self, pool, bt, train, bt_method=None):
        train = super(ExperimentJobs, self).bootstrap(pool, bt, train, bt_method=bt_method)
        if self.validation_set == 'train':
            pool.validation = train.index
        return train

    def start(self, n_jobs=1, pre_dispatch='2*n_jobs'):
        trial = []
        self._setup_options(self.config)
        print self.get_name()
        t0 = time()
        self.data = datautil.load_dataset(self.dataname, self.data_path, categories=self.data_cat, rnd=self.seed,
                                          shuffle=True, percent=self.split, keep_subject=True)
        self.print_lap("Loaded", t0)

        self.data = self.vectorize(self.data)

        cv = self.cross_validation_data(self.data, folds=self.folds, trials=self.trials, split=self.split)

        seeds = np.arange(len(cv)) * 10 + 10

        expert = exputil.get_expert(cfgutil.get_section_options(self.config, 'expert'), size=(len(self.data.train.target),self.data.train.sizes.max()))

        expert.fit(self.data.train.bow, y=self.data.train.target, vct=self.vct)

        lrnr_setup= {'vct':self.vct, "sent_tk":self.sent_tokenizer,  "cost_model":self.cost_model,
                     'validation_set':self.validation_set}
        lrnr_type = cfgutil.get_section_option(self.config, 'learner', 'type')
        if lrnr_type in ['utility-cheat','const-cheat','const-cheat-noisy']:
            lrnr_setup.update({'snip_model':expert.oracle})

        learners = [exputil.get_learner(cfgutil.get_section_options(self.config, 'learner'),
                                        seed=s, **lrnr_setup) for s in seeds]
        self.print_lap("\nPreprocessed", t0)
        # ===================================
        parallel = Parallel(n_jobs=n_jobs, verbose=True,
                            pre_dispatch=pre_dispatch)
        scores = parallel(delayed(self.main_loop_jobs,check_pickle=False)(learners[t], expert, self.budget, self.bootstrap_size,
                                                  self.data, tr[0],tr[1], t)
                         for t, tr in enumerate(cv))
        # ===================================

        self.print_lap("\nDone trials", t0)

        # save the results

        self.report_results(scores)

    def safe_sample(self, data, train_idx, test_idx):

        pool, test = exputil.sample_data(data, train_idx, test_idx)

        rnd_set = range(pool.target.shape[0])
        self.rnd_state.shuffle(rnd_set)

        remaining = deque(rnd_set)

        if self.validation_set == 'test':
            pool.validation_set = test
            pool.validation = test_idx if len(test_idx) >0 else range(test.bow.shape[0])
            pool.remaining = remaining
        elif self.validation_set == 'heldout':
            pool.validation_set = pool
            pool.remaining, pool.validation = self.split_validation(remaining)
        elif self.validation_set == 'train':
            pool.validation_set = pool
            pool.remaining = remaining
            pool.validation = None
        else:
            raise ValueError("Oops, the validations set %s is not available. Check configuration file. " % (self.validation_set))

        return pool, test

    def update_pool(self, pool, query, labels, train):
        super(ExperimentJobs, self).update_pool(pool, query, labels, train)
        if self.validation_set == 'train':
            pool.validation = train.index

        return pool, train

    def main_loop_jobs(self, learner, expert, budget, bootstrap, data, train_idx, test_idx, t):

        iteration = 0
        current_cost = 0

        pool, test = self.safe_sample(data, train_idx, test_idx)

        ## record keeping
        results = self._start_results()

        ## keep track of current training
        train = bunch.Bunch(index=[], target=[])
        query = []
        labels = []
        query_true_labels = []
        classes = np.unique(pool.target)
        query_size = [0] * self.step
        print "Starting trial %s" % (t)

        while current_cost <= budget and iteration <= self.max_iteration and len(pool.remaining) > self.step:
            if iteration == 0:
                # bootstrap
                train = self.bootstrap(pool, bootstrap, train, bt_method=self.bootstrap_method)

                learner = self.retrain(learner, pool, train)
            else:
                # select query and query labels
                query = learner.next_query(pool, self.step)
                query_index = [di for di, _ in query]
                query_true_labels = pool.target[query_index]
                query_size = np.exp((np.array([pool.snippet_cost[i][j] for i,j in query])-self.cost_model['intercept'])/self.cost_model['slope'])

                labels = expert.label(self.get_query(pool,query), y=query_true_labels,
                                      size=pool.sizes[query_index], index=query)

                # update pool and cost
                pool, train = self.update_pool(pool, query, labels, train)
                current_cost = self.update_cost(current_cost, pool, query)

                # re-train the learner
                learner = self.retrain(learner, pool, train)

                if self.debug:
                    self._debug(learner, expert, query)

            # evaluate student
            step_results = self.evaluate(learner, test)

            # evalutate oracle
            step_oracle = self.evaluate_oracle(query_true_labels, labels, labels=classes)

            # record results
            results = self.update_run_results(results, step_results, step_oracle, current_cost,iteration,
                                              query_size=query_size, trial=t)

            iteration += 1

        return results
