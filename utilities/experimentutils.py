from sklearn.datasets import base as bunch
import numpy as np
from nltk import RegexpTokenizer
from nltk.stem import PorterStemmer
from sklearn.utils import safe_indexing
import re

def get_query(data, sizes, query):

    ranges = np.cumsum(sizes)
    queries = []
    for di in query:
        queries.append(data[0 if di==0 else ranges[di-1]:ranges[di]])

    return queries


def sample_data(data, train_idx, test_idx):
    sample = bunch.Bunch(train=bunch.Bunch(), test=bunch.Bunch(), target_names=None)

    # sample.target_names = data.target_names

    # sample.train.data = safe_indexing(data.train.data,train_idx)
    sample.train.target = safe_indexing(data.train.target,train_idx)
    sample.train.bow = safe_indexing(data.train.bow,train_idx)
    sample.train.remaining = []
    sample.train.validation = []
    sample.train.revisit = []

    sample.train.snippets=safe_indexing(data.train.snippets,train_idx)
    sample.train.sizes=safe_indexing(data.train.sizes,train_idx)
    sample.train.snippet_cost = safe_indexing(data.train.snippet_cost,train_idx)


    if len(test_idx) > 0: #if there are test indexes
        # sample.test.data = safe_indexing(data.train.target,test_idx)
        sample.test.target = safe_indexing(data.train.target,test_idx)
        sample.test.bow = safe_indexing(data.train.bow,train_idx)
        sample.test.snippets=safe_indexing(data.train.snippets,train_idx)
        sample.test.sizes=safe_indexing(data.train.sizes,train_idx)
        sample.test.snippet_cost = safe_indexing(data.train.snippet_cost,train_idx)

    else:
        sample.test = data.test

    return sample.train, sample.test

def stemming(doc):

    wnl = PorterStemmer()
    mytokenizer = RegexpTokenizer('\\b\\w+\\b')

    return [wnl.stem(t) for t in mytokenizer.tokenize(doc)]


def get_vectorizer(config):
    limit = config['limit']
    vectorizer = config['vectorizer']

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    if vectorizer == 'tfidf':
        return TfidfVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1))
    elif vectorizer == "tfidfvocab":
        vocab = open(config['vocabulary']).readlines()
        vocab = [v.strip() for v in vocab]
        return TfidfVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1),
                               vocabulary=vocab)
    elif vectorizer == 'counts':
        return CountVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=True, ngram_range=(1,1),
                      token_pattern=re.compile(r'(?u)\b\w+\b'), tokenizer=StemTokenizer())
    elif vectorizer == 'bow':
        from datautils import StemTokenizer
        return CountVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=True, ngram_range=(1, 3),
                      token_pattern='\\b\\w+\\b', tokenizer=StemTokenizer())
    else:
        return None


def get_classifier(cl_name, **kwargs):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    clf = None
    if cl_name is not None:
        if cl_name in "mnb":
            alpha = 1
            if 'parameter' in kwargs:
                alpha = kwargs['parameter']
            clf = MultinomialNB(alpha=alpha)
        elif cl_name == "lr" or cl_name == "lrl1":
            c = 1
            if 'parameter' in kwargs:
                c = kwargs['parameter']
            clf = LogisticRegression(penalty="l1", C=c)
        elif cl_name == "lrl2":
            c = 1
            if 'parameter' in kwargs:
                c = kwargs['parameter']
            clf = LogisticRegression(penalty="l2", C=c)
        else:
            raise ValueError("We need a classifier name for the student [lr|mnb]")
    return clf


# def get_learner(learn_config, vct=None, sent_tk=None, seed=None, cost_model=None):
def get_learner(learn_config, **kwargs):

    vct = kwargs['vct']
    sent_tk = kwargs['vct']
    cost_model = kwargs['cost_model']
    seed = kwargs['seed']

    from learner.base import Learner
    cl_name = learn_config['model']
    clf = get_classifier(cl_name, parameter=learn_config['parameter'])
    learner = Learner(clf)


    # utlity document - snippet method
    if learn_config['type'] == 'utility-based':
        from learner.utility_based import UtilityBasedLearner
        learner = UtilityBasedLearner(clf, snippet_fn=None, utility_fn=None, seed=seed)
    elif learn_config['type'] == 'utility-firstk':
        from learner.utility_based import FirstK
        learner = FirstK(clf, snippet_fn=None, utility_fn=None, seed=seed)
        raise ValueError("Oops, check this method first.")
    elif learn_config['type'] == 'utility-cheat':
        from learner.utility_based import JointCheat
        snp_model = kwargs['snip_model']
        learner = JointCheat(clf, snippet_fn=None, utility_fn=None, seed=seed, snip_model=snp_model)

    # random documents - snippet method
    elif learn_config['type'] == 'const-utility':
        from learner.sequential_utility import Sequential
        learner = Sequential(clf, snippet_fn=None, utility_fn=None, seed=seed)
    elif learn_config['type'] == 'const-firstk':
        from learner.sequential_utility import FirstK
        learner = FirstK(clf, snippet_fn=None, utility_fn=None, seed=seed)
    elif learn_config['type'] == 'const-cheat':
        from learner.sequential_utility import SequentialJointCheat
        snp_model = kwargs['snip_model']
        learner = SequentialJointCheat(clf, snippet_fn=None, utility_fn=None, seed=seed, snip_model=snp_model)
    elif learn_config['type'] == 'const-cheat-noisy':
        from learner.sequential_utility import SequentialJointNoisyCheat
        snp_model = kwargs['snip_model']
        learner = SequentialJointNoisyCheat(clf, snippet_fn=None, utility_fn=None, seed=seed, snip_model=snp_model)
    else:
        raise ValueError("We don't know {} leaner".format(learn_config['type']))
    learner.set_loss_function(learn_config['loss_function'])
    learner.set_sent_tokenizer(sent_tk)
    learner.set_vct(vct)
    learner.set_cost_model(cost_model)
    learner.set_cost_fn(get_costfn(learn_config['cost_function']))

    if hasattr(learner,'validation_method'):
        method = 'eval'
        if kwargs['validation_set'] == 'train':
            method = 'cross-validation'
        elif 'validation_method' in learn_config:
            method = learn_config['validation_method']
        learner.set_validation_method(method)

    return learner


def get_expert(config, size=None):

    from expert.experts import PredictingExpert, PerfectReluctantDocumentExpert,\
        TrueExpert, NoisyExpert, ReluctantPredictingExpert
    from expert.noisy_expert import NoisyReluctantExpert

    cl_name = config['model']
    clf = get_classifier(cl_name, parameter=config['parameter'])

    if config['type'] == 'true':
        expert = TrueExpert(None)
    elif config['type'] == 'pred':
        expert = PredictingExpert(clf)
    elif config['type'] == 'noisy':
        p = config['noise_p']
        expert = NoisyExpert(None, p)
    elif config['type'] == 'perfectreluctant': # reluctant based on unc threshold
        p = config['threshold']
        expert = PerfectReluctantDocumentExpert(clf, p)
    elif config['type'] == 'noisyreluctant': # reluctant based on unc threshold
        p = config['threshold']
        args = {'factor': config['scale'], 'data_size': size}
        expert = NoisyReluctantExpert(clf, p, **args)
    elif config['type'] == 'reluctantpredicting': # reluctant based on unc threshold, predicting based on clf
        p = config['threshold']
        expert = ReluctantPredictingExpert(clf, p)
    else:
        raise Exception("We don't know {} expert".format(config['type']))

    return expert

def get_bootstrap(config):
    bt = config['bootstrap']
    if 'bootstrap_type' in config:
        mt = config['bootstrap_type']
    else:
        mt = None

    return bt, mt

def get_tokenizer(tk_name, **kwargs):
    if tk_name == 'nltk':
        import nltk
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        return sent_detector
    elif tk_name == 'snippet':
        from snippet_tokenizer import SnippetTokenizer
        k = (1,1)
        if 'snip_size' in kwargs:
            k = kwargs['snip_size']
        sent_detector = SnippetTokenizer(k=k)
        return sent_detector
    elif tk_name == 'windowsnippet':
        from snippet_tokenizer import WindowSnippetTokenizer
        k = (1,1)
        if 'snip_size' in kwargs:
            k = kwargs['snip_size']
        sent_detector = WindowSnippetTokenizer(k=k)
        return sent_detector
    elif tk_name == 'firstksnippet':
        from snippet_tokenizer import FirstWindowSnippetTokenizer
        k = (1,1)
        if 'snip_size' in kwargs:
            k = kwargs['snip_size']
        sent_detector = FirstWindowSnippetTokenizer(k=k)
        return sent_detector
    else:
        raise Exception("Unknown sentence tokenizer")


def get_costfn(fn_name):
    import costutils
    return getattr(costutils, fn_name)


def print_file(cost, mean, std, f):
    f.write("COST\tMEAN\tSTDEV\n")
    for a, b, c in zip(cost, mean, std):
        f.write("{0:.3f}\t{1:.3f}\t{2:.3f}\n".format(a, b, c))
    f.close()


def print_cm_file(cost, mean, std, f):
    f.write("COST\tT0\tF1\tF0\tT1\tSTDEV\n")
    for a, b, c in zip(cost, mean, std):
        f.write("{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}\n".format(a, *b))
    f.close()
