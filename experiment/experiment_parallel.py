import os, sys

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../"))

from sklearn import metrics
import utilities.experimentutils as exputil
import utilities.datautils as datautil
import utilities.configutils as cfgutil
from sklearn import cross_validation
import numpy as np
from collections import defaultdict, deque
from sklearn.datasets import base as bunch
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.base import clone
from time import time

from experiment.base import Experiment

__author__ = "mramire8"


class ExperimentJobs(Experiment):

    def __init__(self, config, verbose=False, debug=False):
        super(ExperimentJobs, self).__init__(config, verbose=verbose, debug=debug)
        self.save_all = True

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
        cost = np.array([[self.costfn(si) for si in s] for s in snippets])

        ranges = np.cumsum(sizes)
        data.train.snippets = [snippet_bow[0 if i == 0 else ranges[i-1]:ranges[i]] for i in range(len(sizes))]
        data.train.sizes= sizes
        data.train.snippet_cost = cost

        # Processing tests
        snippets = self.sent_tokenizer.tokenize_sents(data.test.data)
        snippet_bow = self.vct.transform(sentence_iterator(snippets))
        sizes = np.array([len(snip) for snip in snippets])
        cost = np.array([[self.costfn(si) for si in s] for s in snippets])

        ranges = np.cumsum(sizes)
        data.test.snippets = [snippet_bow[0 if i == 0 else ranges[i-1]:ranges[i]] for i in range(len(sizes))]
        data.test.sizes= sizes
        data.test.snippet_cost = cost

        return data


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

        # learner = exputil.get_learner(cfgutil.get_section_options(self.config, 'learner'),
        #                               vct=self.vct, sent_tk=self.sent_tokenizer, seed=0,  cost_model=self.cost_model)
        #
        learners = [exputil.get_learner(cfgutil.get_section_options(self.config, 'learner'),
                                      vct=self.vct, sent_tk=self.sent_tokenizer, seed=s,  cost_model=self.cost_model) for s in seeds]


        expert = exputil.get_expert(cfgutil.get_section_options(self.config, 'expert'), size=len(self.data.train.data))

        expert.fit(self.data.train.bow, y=self.data.train.target, vct=self.vct)


        t = 0

        self.print_lap("\nPreprocessed", t0)

        parallel = Parallel(n_jobs=n_jobs, verbose=True,
                            pre_dispatch=pre_dispatch)
        scores = parallel(delayed(self.main_loop_jobs,check_pickle=False)(lrnr, expert, self.budget, self.bootstrap_size,
                                                  self.data, train,test)
                         for lrnr in learners for train, test in cv)
        # return np.array(scores)[:, 0]

        self.print_lap("\nDone trials", t0)


        # save the results
        print scores
        self.report_results(scores)

    def safe_sample(self, data, train_idx, test_idx):

        pool, test = exputil.sample_data(data, train_idx, test_idx)

        rnd_set = range(pool.target.shape[0])
        self.rnd_state.shuffle(rnd_set)

        remaining = deque(rnd_set)

        pool.remaining, pool.validation = self.split_validation(remaining)

        return pool, test

    def main_loop_jobs(self, learner, expert, budget, bootstrap, data, train_idx, test_idx):

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
        while current_cost <= budget and iteration <= self.max_iteration and len(pool.remaining) > self.step:
            if iteration == 0:
                # bootstrap
                train = self.bootstrap(pool, bootstrap, train, bt_method=self.bootstrap_method)

                learner = self.retrain(learner, pool, train)
            else:
                # select query and query labels
                query = learner.next_query(pool, self.step)

                query_true_labels = pool.target[[di for di, _ in query]]

                labels = expert.label(self.get_query(pool,query), y=query_true_labels)

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
            results = self.update_run_results(results, step_results, step_oracle, current_cost, trial=(seed-10)/10)

            iteration += 1

        return results
