__author__ = "mramire8"

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
from learner.strategy import BootstrapFromEach
from sklearn.datasets import base as bunch
from time import time
from sklearn.utils import safe_indexing

class Experiment(object):
    """Main experiment class to run according to configuration"""
    # def __init__(self, dataname, learner, expert, trials=5, folds=1, split=.5, costfn=None):
    def __init__(self, config, verbose=False, debug=False):
        super(Experiment, self).__init__()
        self.verbose = verbose
        self.debug = debug
        self.save_all = True
        self.config = config

        self.dataname = None
        self.data_cat = None
        self.data = None
        self.data_path = None

        self.trials = None
        self.folds = None
        self.split = None
        self.costfn = None
        self.cost_model = None
        self.cost_base = 25
        self.budget = None
        self.max_iteration = None
        self.step = None
        self.bootstrap_size = None
        self.seed = None
        self.output = None

        self.rnd_state = np.random.RandomState(32564)
        self.remaining = None
        self.vct = exputil.get_vectorizer(cfgutil.get_section_options(config, 'data'))
        self.sent_tokenizer = None

    def vectorize(self, data):
        data.train.bow = self.vct.fit_transform(data.train.data)
        data.test.bow = self.vct.transform(data.test.data)
        return data

    def _sample_data(self, data, train_idx, test_idx):

        def sentence_iterator(data):
            for doc in data:
                for snip in doc:
                    yield snip

        sample = bunch.Bunch(train=bunch.Bunch(), test=bunch.Bunch())

        if len(test_idx) > 0: #if there are test indexes
            sample.train.data = np.array(data.data, dtype=object)[train_idx]
            sample.test.data = np.array(data.data, dtype=object)[test_idx]

            sample.train.target = data.target[train_idx]
            sample.test.target = data.target[test_idx]

            sample.train.bow = self.vct.fit_transform(sample.train.data)
            sample.test.bow = self.vct.transform(sample.test.data)

            sample.target_names = data.target_names
            sample.train.remaining = []

        else:
            ## Just shuffle the data and vectorize
            sample = data
            data_lst = np.array(data.train.data, dtype=object)
            data_lst = data_lst[train_idx]
            sample.train.data = data_lst

            sample.train.target = data.train.target[train_idx]

            sample.train.bow = self.vct.fit_transform(data.train.data)
            sample.test.bow = self.vct.transform(data.test.data)

            sample.train.remaining = []

        snippets = self.sent_tokenizer.tokenize_sents(sample.train.data)
        snippet_bow = self.vct.transform(sentence_iterator(snippets))
        sizes = np.array([len(snip) for snip in snippets])
        cost = np.array([[self.costfn(si, cost_model=self.cost_model) for si in s] for s in snippets])

        sample.train.snippets=snippet_bow
        sample.train.sizes=sizes
        sample.train.snippet_cost = cost

        snippets = self.sent_tokenizer.tokenize_sents(sample.test.data)
        snippet_bow = self.vct.transform(sentence_iterator(snippets))
        sizes = np.array([len(snip) for snip in snippets])
        # cost = np.array([self.costfn(s) for s in sentence_iterator(snippets)])
        cost = np.array([[self.costfn(si, cost_model=self.cost_model) for si in s] for s in snippets])

        sample.test.snippets=snippet_bow
        sample.test.sizes=sizes
        sample.test.snippet_cost = cost

        return sample.train, sample.test

    def cross_validation_data(self, data, **config):

        if 'train' in data.keys():
            n = data.train.target.shape[0]
        else:
            n = data.target.shape[0]

        cv = None

        if config['folds'] == 1 and 'test' not in data.keys():
            cv = cross_validation.ShuffleSplit(n, n_iter=config['trials'], test_size=config['split'],
                                               random_state=self.rnd_state)
            config['folds'] = 1
        elif 'test' in data.keys():
            cv = cross_validation.ShuffleSplit(n, n_iter=config['trials'], test_size=0.0,
                                               random_state=self.rnd_state)
            config['folds'] = 1

        else:
            cv = cross_validation.KFold(n, n_folds=config['folds'], random_state=self.rnd_state)
        return cv

    def _setup_options(self, config_obj):

        # experiment related config
        config = cfgutil.get_section_options(config_obj, 'experiment')
        self.trials = config['trials']
        self.folds = config['folds']
        self.max_iteration = config['maxiter']
        self.step = config['stepsize']
        self.budget = config['budget']
        self.prefix = config['fileprefix']
        self.output = config['outputdir']
        self.seed = config['seed']
        self.dataname = config['data']
        self.bootstrap_size, self.bootstrap_method = exputil.get_bootstrap(config)
        self.costfn = exputil.get_costfn(config['costfunction'])

        if 'cost_model' in config.keys():
            self.cost_model = config['cost_model']
            # self.cost_base = config['cost_base']

        # data related config
        config = cfgutil.get_section_options(config_obj, 'data')
        self.split = config['split']
        self.data_cat = config['categories']
        self.limit = config['limit']
        self.data_path = config['path']

        #data related config
        config = cfgutil.get_section_options(config_obj, 'expert')
        args = {}
        if 'snip_size' in config:
            args.update({'snip_size':config['snip_size']})
        self.sent_tokenizer = exputil.get_tokenizer(config['sent_tokenizer'], **args)

        try:
            if not os.path.exists(self.output):
                os.makedirs(self.output)
        except OSError:
            pass

    def print_lap(self, msg, t0):
        t1 = time()
        print "%s %.3f secs (%.3f mins)" % (msg, (t1-t0), (t1-t0)/60)

    def start(self):
        trial = []
        self._setup_options(self.config)
        print self.get_name()
        t0 = time()
        self.data = datautil.load_dataset(self.dataname, self.data_path, categories=self.data_cat, rnd=self.seed,
                                          shuffle=True, percent=self.split, keep_subject=True)
        self.print_lap("Loaded", t0)
        # self.data = self.vectorize(self.data)
        cv = self.cross_validation_data(self.data, folds=self.folds, trials=self.trials, split=self.split)
        t = 0
        for train_index, test_index in cv:
            # get the data of this cv iteration
            # train, test = exputil.sample_data(self.data, train_index, test_index)
            train, test = self._sample_data(self.data, train_index, test_index)
            self.print_lap("\nSampled", t0)
            # get the expert and student
            learner = exputil.get_learner(cfgutil.get_section_options(self.config, 'learner'),
                                          vct=self.vct, sent_tk=self.sent_tokenizer, seed=(t * 10 + 10),  cost_model=self.cost_model)

            expert = exputil.get_expert(cfgutil.get_section_options(self.config, 'expert'), size=len(train.data))

            expert.fit(train.bow, y=train.target, vct=self.vct)

            # do active learning
            results = self.main_loop(learner, expert, self.budget, self.bootstrap_size, train, test)

            self.print_lap("\nTrial %s" % t, t0)

            # save the results
            trial.append(results)
            t += 1
        self.report_results(trial)

    def get_name(self):
        cfg = cfgutil.get_section_options(self.config, 'learner')
        post = cfgutil.get_section_option(self.config, 'experiment', 'fileprefix')
        name = "data-{}-lrn-{}-ut-{}-{}".format(self.dataname, cfg['type'], cfg['loss_function'],
                                                               post)
        return name

    def bootstrap(self, pool, bt, train, bt_method=None):
        # get a bootstrap
        # if bt_method is None:
        bt_obj = BootstrapFromEach(None, seed=self.seed)

        initial = bt_obj.bootstrap(pool, step=bt, shuffle=False)

        # update initial training data
        train.index = initial
        train.target = pool.target[initial].tolist()
        if bt_method == 'amt-tfe':
            train.target = pool.alltarget[initial].tolist()
        else:
            for q in train.index:
                pool.remaining.remove(q)
            # pool.validation = train.index

        return train

    def update_cost(self, current_cost, pool, query):
        cost = 0
        for di, si in query:
            cost += pool.snippet_cost[di][si]
        return current_cost + cost

    def evaluate(self, learner, test):
        prediction = learner.predict(test.bow)
        pred_proba = learner.predict_proba(test.bow)
        accu = metrics.accuracy_score(test.target, prediction)
        auc = metrics.roc_auc_score(test.target, pred_proba[:, 1])
        return {'auc': auc, 'accuracy': accu}

    def evaluate_oracle(self, true_labels, predictions, labels=None):
        cm = np.zeros((2,2))

        try:
            t = np.array([[x,y] for x, y in zip(true_labels, predictions) if y is not None])
            if len(t) > 0:
                cm = metrics.confusion_matrix(t[:,0], t[:,1], labels=labels)
        except AttributeError:
            pass
        return cm

    def update_run_results(self, results, step, oracle, cost, iteration, trial=None):

        results['accuracy'][cost].append(step['accuracy'])
        results['auc'][cost].append(step['auc'])
        try:
            results['ora_accu'][cost].append(oracle)
        except Exception:
            pass
        oracle_text = ""
        if self.verbose:
            if iteration == 0:
                print "\nIT\tACCU\tAUC\tT0\tF1\tF0\tT1"
            print "{3}-{4}\t{0:0.2f}\t{1:.3f}\t{2:.3f}\t".format(cost, step['accuracy'], step['auc'], iteration, trial),
            try:
                print "\t".join(["{0:.3f}".format(x) for x in np.reshape(oracle, 4)])
                oracle_text = "\t".join(["{0:.3f}".format(x) for x in np.reshape(oracle, 4)])
            except Exception:
                oracle_text = ""
                pass

        if self.save_all:
                if trial is not None:
                    output_name = self.output + "/" + self.get_name()  + "-accu-all-%s.txt" %(trial)
                else:
                    output_name = self.output + "/" + self.get_name()  + "-accu-all.txt"
                with open(output_name, "a") as f:
                    if iteration == 0:
                       f.write("IT\tACCU\tAUC\tT0\tF1\tF0\tT1\n")
                    to_print = "{0:0.2f}\t{1:.3f}\t{2:.3f}\t{3}\n".format(cost, step['accuracy'], step['auc'],oracle_text)
                    f.write(to_print)

        return results

    def update_pool(self, pool, query, labels, train):
        for q, l in zip(query, labels):
            pool.remaining.remove(q[0])
            train.index.append(q[0])
            train.target.append(l)

        return pool, train

    def retrain(self, learner, pool, train):

        # return learner.fit(X, y, train_index=train.index)

        return learner.fit(pool, train=train)

    def get_query(self, data, query):

        if hasattr(data.snippets, "shape"):
            ranges = np.cumsum(data.sizes)
            queries = []
            for di, si in query:
                queries.append(data.snippets[0 if di==0 else ranges[di-1]:ranges[di]][si])
        else:

            queries = [data.snippets[d][np.ix_([s])] for d,s in query]

        return queries

    def split_validation(self, remaining):
        remaining = list(remaining)
        n = len(remaining)
        half = int(n * self.split)
        return deque(remaining[:half]), list(remaining[half:])

    def main_loop(self, learner, expert, budget, bootstrap, pool, test):

        iteration = 0
        current_cost = 0
        rnd_set = range(pool.target.shape[0])
        self.rnd_state.shuffle(rnd_set)
        remaining = deque(rnd_set)
        pool.remaining, pool.validation = self.split_validation(remaining)

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
            results = self.update_run_results(results, step_results, step_oracle, current_cost, iteration)

            iteration += 1

        return results

    def _start_results(self):
        r = {}
        r['accuracy'] = defaultdict(lambda: [])
        r['auc'] = defaultdict(lambda: [])
        r['ora_accu'] = defaultdict(lambda: [])
        return r

    def _get_iteration(self, iteration):
        cost = sorted(iteration.keys())
        perf = [np.mean(iteration[xi]) for xi in cost]
        std = [np.std(iteration[xi]) for xi in cost]
        n = [np.size(iteration[xi]) for xi in cost]

        return cost, perf, std, n

    def _get_cm_iteration(self, iteration):
        cost = sorted(iteration.keys())
        perf = [np.mean(iteration[xi], axis=0).reshape(4) for xi in cost]
        std = [np.std(iteration[xi]) for xi in cost]
        n = [np.size(iteration[xi]) for xi in cost]

        return cost, perf, std, n

    def _extrapolate(self, t_perf, t_cost, cost_25, step_size=10):
        # def extrapolate_trials(tr, cost_25=8.2, step_size=10):
        '''
        Extrapolate the x-axis information per trial, to create an average later.
        Trials is a list of each trial performance, where each trial has the cost and
        the performance. The x-axis is extrapolated based on cost of 25-word segments.
        The cost of each iteration is cost of 25-words times number of queries. C(25) * step.
        '''
        cost_delta = cost_25 * step_size  # Cost of 25 words based on user study

        ext_perf = []
        ext_cost = []

        cost, data  = t_cost, t_perf

        trial_data = np.array(data)

        i = 0
        current_c = np.ceil(cost[0] / cost_delta) * cost_delta

        while i < trial_data.shape[0] - 1:  # while reaching end of rows
            a = trial_data[i]
            a1 = trial_data[i + 1]
            c = cost[i]
            c1 = cost[i+1]
            if c <= current_c <= c1:
                m = (a1 - a) / (c1 - c) * (current_c - c)
                z = m + a

                ext_cost.append(current_c)
                ext_perf.append(z)

                current_c += cost_delta
            if c1 < current_c:
                i += 1

        return ext_cost, ext_perf

    def report_results(self, results):
        output_name = self.output + "/" + self.get_name()
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        accu = []
        auc = []
        ora = []
        cost = []
        for tr in results:
            c, p, s, n = self._get_iteration(tr['accuracy'])
            # c, p = self._extrapolate(p,c,self.cost_model[self.cost_base],self.trials)
            c, p = self._extrapolate(p, c, self.costfn(self.cost_base, cost_model=self.cost_model), self.trials)
            accu.append(p)
            cost.append(c)
            c, p, s, n = self._get_iteration(tr['auc'])
            # c, p = self._extrapolate(p,c,self.cost_model[self.cost_base],self.trials)
            c, p = self._extrapolate(p, c, self.costfn(self.cost_base, cost_model=self.cost_model), self.trials)
            auc.append(p)
            c, p, s, n = self._get_cm_iteration(tr['ora_accu'])
            # c, p = self._extrapolate(p,c,self.cost_model[self.cost_base],self.trials)
            c, p = self._extrapolate(p, c, self.costfn(self.cost_base, cost_model=self.cost_model), self.trials)
            ora.append(p)
        min_x = min([len(m) for m in cost])

        p = np.mean([a[:min_x] for a in accu], axis=0)
        c = np.mean([a[:min_x] for a in cost], axis=0)
        s = np.std([a[:min_x] for a in accu], axis=0)
        exputil.print_file(c, p, s, open(output_name + "-accu.txt", "w"))
        p = np.mean([a[:min_x] for a in auc], axis=0)
        s = np.std([a[:min_x] for a in auc], axis=0)
        exputil.print_file(c, p, s, open(output_name + "-auc.txt", "w"))
        p = np.mean([a[:min_x] for a in ora], axis=0)
        s = np.std([a[:min_x] for a in ora], axis=0)
        exputil.print_cm_file(c, p, s, open(output_name + "-oracle-cm.txt", "w"))

    def _debug(self, learner, expert, query):
        print "\t".join("di:{} - si:{}".format(*q) for q in query)

