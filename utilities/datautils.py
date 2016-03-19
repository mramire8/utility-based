from sklearn.datasets import load_files
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import base as bunch
import numpy as np
import pickle
import gzip
from sklearn.cross_validation import ShuffleSplit
from os.path import isfile
import codecs
import json
from zipfile import ZipFile


class StemTokenizer(object):
    def __init__(self):
        from nltk import RegexpTokenizer
        from nltk.stem import PorterStemmer

        self.wnl = PorterStemmer()
        self.mytokenizer = RegexpTokenizer('\\b\\w+\\b')

    def __call__(self, doc):
        return [self.wnl.stem(t) for t in self.mytokenizer.tokenize(doc)]


def keep_header_subject(text, keep_subject=False):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    """
    _before, _blankline, after = text.partition('\n\n')

    sub = [l for l in _before.split("\n") if "Subject:" in l]
    if keep_subject:
        final = sub[0] + "\n" + after
    else:
        final = after
    return final


def load_imdb(path, subset="all", shuffle=True, rnd=2356):
    """
    load text files from IMDB movie reviews from folders to memory
    :param path: path of the root directory of the data
    :param subset: what data will be loaded, train or test or all
    :param shuffle:
    :param rnd: ranom seed value
    :param vct: vectorizer
    :return: :raise ValueError:
    """

    data = bunch.Bunch()

    if subset in ('train', 'test'):
        data[subset] = load_files("{0}/{1}".format(path, subset), encoding="latin-1", load_content=True,
                                  random_state=rnd)
    elif subset == "all":
        data["train"] = load_files("{0}/{1}".format(path, "train"), encoding="latin-1", load_content=True,
                                   random_state=rnd)
        data["test"] = load_files("{0}/{1}".format(path, "test"), encoding="latin-1", load_content=True,
                                  random_state=rnd)
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

    if shuffle:
        random_state = np.random.RandomState(rnd)
        indices = np.arange(data.train.target.shape[0])
        random_state.shuffle(indices)
        data.train.filenames = data.train.filenames[indices]
        data.train.target = data.train.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.train.data, dtype=object)
        data_lst = data_lst[indices]
        data.train.data = data_lst
        data.test.data = np.array(data.test.data, dtype=object)

    data = minimum_size(data, min_size=1)

    return data


def process_tweets(data_users):
    import json
    data = []
    for user in data_users:
        timeline = user.split("\n")
        for tweet in timeline:
            t = json.loads(tweet)
            data.append(t['text'])
    return "######".join(t for t in data)


def load_gender_twitter(path, subset="all", shuffle=True, rnd=2356, percent=.5):
    """
    load text files from IMDB movie reviews from folders to memory
    :param path: path of the root directory of the data
    :param subset: what data will be loaded, train or test or all
    :param shuffle:
    :param rnd: ranom seed value
    :param vct: vectorizer
    :return: :raise ValueError:
    """
    from sklearn.cross_validation import ShuffleSplit
    data = bunch.Bunch()

    if subset in ('train', 'test'):
        pass
    elif subset == "all":
        data = load_files(path, encoding="latin1", load_content=True, random_state=rnd)
        data.data = np.array([process_tweets(d) for d in data.data], dtype=object)
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

    indices = ShuffleSplit(len(data.data), n_iter=1, test_size=percent, random_state=rnd)
    for train_ind, test_ind in indices:
        data = bunch.Bunch(train=bunch.Bunch(data=data.data[train_ind], target=data.target[train_ind],
                                             filenames=data.filenames[train_ind], target_names=data.target_names),
                           test=bunch.Bunch(data=data.data[test_ind], target=data.target[test_ind],
                                            filenames=data.filenames[test_ind], target_names=data.target_names))

    if shuffle:
        random_state = np.random.RandomState(rnd)
        indices = np.arange(data.train.target.shape[0])
        random_state.shuffle(indices)
        data.train.filenames = data.train.filenames[indices]
        data.train.target = data.train.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.train.data, dtype=object)
        data_lst = data_lst[indices]
        data.train.data = data_lst
        data.test.data = np.array(data.test.data, dtype=object)

    data = minimum_size(data)

    return data


def load_aviation(path, subset="all", shuffle=True, rnd=2356, percent=None, keep_suject=False):
    """
    load text files from Aviation-auto dataset from folders to memory. It will return a 25-75 percent train test split
    :param path: path of the root directory of the data
    :param subset: what data will be loaded, train or test or all
    :param shuffle:
    :param rnd: random seed value
    :param vct: vectorizer
    :return: :raise ValueError:
    """
    from sklearn.cross_validation import ShuffleSplit

    data = bunch.Bunch()
    if subset in ('train', 'test'):
        raise Exception("We are not ready for train test aviation data yet")
    elif subset == "all":
        data = load_files(path, encoding="latin1", load_content=True,
                          random_state=rnd)
        data.data = np.array([keep_header_subject(text, keep_subject=keep_suject) for text in data.data], dtype=object)
        data.data = np.array([remove_greeting(text) for text in data.data], dtype=object)
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

    indices = ShuffleSplit(len(data.data), n_iter=1, test_size=percent, random_state=rnd)
    for train_ind, test_ind in indices:
        data = bunch.Bunch(train=bunch.Bunch(data=data.data[train_ind], target=data.target[train_ind],
                                             filenames=data.filenames[train_ind], target_names=data.target_names),
                           test=bunch.Bunch(data=data.data[test_ind], target=data.target[test_ind],
                                            filenames=data.filenames[test_ind], target_names=data.target_names))

    if shuffle:
        random_state = np.random.RandomState(rnd)
        indices = np.arange(data.train.target.shape[0])
        random_state.shuffle(indices)
        data.train.filenames = data.train.filenames[indices]
        data.train.target = data.train.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.train.data, dtype=object)
        data_lst = data_lst[indices]
        data.train.data = data_lst
        data.test.data = np.array(data.test.data, dtype=object)

    data = minimum_size(data)
    return data


def remove_greeting(text):
    """
    Given an email like text return the text without the greeting part, e.g., "Hi Darling, ..."
    :param text:
    :return:
    """
    import re
    if len(text) <= 0:
        return text

    parts = text.split("\n")
    first = parts[0] if len(parts[0]) >0 else parts[1]

    res = re.search("(\W|^)(Hi|hello|hey|good morning|good evening|good afternoon|thanks|thankyou|thank you)(\W|$)", first, re.IGNORECASE)

    first_tk = first.strip().split()

    result = text

    if (res is not None) or (len(first_tk) <=3 ):
        ## if there is a salutation i
        if len(first_tk) > 0 and not first_tk[-1][-1].isalnum() :
            # if the last character  of the last token is a special characters
            result =  "\n".join(parts[1:])
        elif (res is not None):
            # if we found greeting words in the  first line
            result = "\n".join(parts[1:])

    return result


def minimum_size(data, min_size=10):
    for part in data.keys():
        if len(data[part].data) != len(data[part].target):
            raise Exception("There is something wrong with the data")
        # filtered = [(x, y) for x, y in zip(data[part].data, data[part].target) if len(x.strip()) >= 10]
        data[part].data = np.array([doc.replace("<br />", " ") for doc in data[part].data], dtype=object)
        filtered = np.array([len(x.strip()) for x in data[part].data])
        data[part].data = data[part].data[filtered >= min_size]
        data[part].target = data[part].target[filtered >= min_size]
    return data


def minimum_size_sraa(data, min_size=10):
    if len(data.data) != len(data.target):
        raise Exception("There is something wrong with the data")
    filtered = np.array([len(x.strip()) for x in data.data])
    data.data = data.data[filtered >= 10]
    data.target = data.target[filtered >= min_size]
    return data


def load_20newsgroups(category=None, shuffle=True, rnd=1):
    categories = {'religion': ['alt.atheism', 'talk.religion.misc'],
                  'graphics': ['comp.graphics', 'comp.windows.x'],
                  'hardware': ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'],
                  'baseball': ['rec.sport.baseball', 'sci.crypt']}
    cat = None
    if category is not None:
        cat = categories[category]

    data = bunch.Bunch()
    data.train = fetch_20newsgroups(subset='train', categories=cat, remove=('headers', 'footers', 'quotes'),
                                    shuffle=shuffle, random_state=rnd)

    # data.train.data = np.array([keep_header_subject(text) for text in data.train.data], dtype=object)
    data.train.data = np.array(data.train.data, dtype=object)
    data.test = fetch_20newsgroups(subset='test', categories=cat, remove=('headers', 'footers', 'quotes'),
                                   shuffle=shuffle, random_state=rnd)

    # data.test.data = np.array([keep_header_subject(text) for text in data.test.data], dtype=object)
    data.test.data = np.array(data.test.data, dtype=object)
    data = minimum_size(data)

    if shuffle:
        random_state = np.random.RandomState(rnd)
        indices = np.arange(data.train.target.shape[0])
        random_state.shuffle(indices)
        data.train.filenames = data.train.filenames[indices]
        data.train.target = data.train.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.train.data, dtype=object)
        data_lst = data_lst[indices]
        data.train.data = data_lst

    return data


def preprocess(string, lowercase, collapse_urls, collapse_mentions):
    import re

    if not string:
        return ""
    if lowercase:
        string = string.lower()
    if collapse_urls:
        string = re.sub('http\S+', 'THIS_IS_A_URL', string)
    if collapse_mentions:
        string = re.sub('@\S+', 'THIS_IS_A_MENTION', string)

    return string


def timeline_to_doc(user, *args):
    tweets = []
    for tw in user:
        tweets.append(preprocess(tw['text'], *args))
    return tweets


def user_to_doc(users, *args):
    timeline = []
    user_names = []
    user_id = []

    for user in users:
        timeline.append(timeline_to_doc(user, *args))
        user_names.append(user[0]['user']['name'])
        user_id.append(user[0]['user']['screen_name'])
    return user_id, user_names, timeline


def get_date(date_str):
    import datetime

    return datetime.datetime.strptime(date_str.strip('"'), "%a %b %d %H:%M:%S +0000 %Y")


def read_data(filename):

    all_data= []
    target= []
    try:
        with ZipFile(filename, "r") as zfile:
            for name in zfile.namelist():
                with zfile.open(name, 'rU') as readFile:
                    lines = readFile.readlines() #.decode('utf8')
                    all_data.extend(json.loads(line)[0] for line in lines)
                    target.extend([name] * len(lines))
                    print len(lines)
    except Exception as e:
        raise RuntimeError('Oops, something is wrong and data cannot be uploaded.')
    return np.array(all_data), target


def convert_tweet_2_data(data_path, rnd):
    """
    Convert tweet time lines into dataset
    :param data_path:
    :param vct:
    :return: bunch.Bunch
        Bunch with the data in train and test from twitter bots and human accounts
    """
    twits, target = read_data(data_path + "/twitter_v2.zip")

    all_data = np.array([(g,t) for g,t in zip(twits,target) if get_date(g['created_at']).year > 2011])

    twits = all_data[:,0]

    target = all_data[:,1]

    data = bunch_users(twits,target,  True, True, True, rnd, class_name=np.unique(target))

    return data


def bunch_users(twits, target, lowercase, collapse_urls, collapse_mentions, rnd, class_name=None):
    labels = None
    if labels is None:
        labels = [0, 1]

    _, _, timeline = user_to_doc(twits, lowercase, collapse_urls, collapse_mentions)

    user_text = ["_THIS_IS_A_SEPARATOR_".join(t) for t in timeline]
    data = bunch.Bunch(data=user_text, target=target, target_names=class_name,
                       user_timeline=timeline)

    random_state = np.random.RandomState(rnd)

    indices = np.arange(len(data.target))
    random_state.shuffle(indices)
    data.target = np.array(data.target)[indices]
    data_lst = np.array(data.data, dtype=object)
    data_lst = data_lst[indices]
    data.data = data_lst
    data.user_timeline = np.array(data.user_timeline)[indices]
    data.target_names = class_name
    return data


def load_twitter(path, shuffle=True, rnd=1, percent=.5):
    """
    load text files from twitter data
    :param path: path of the root directory of the data
    :param subset: what data will be loaded, train or test or all
    :param shuffle:
    :param rnd: random seed value
    :param vct: vectorizer
    :return: :raise ValueError:
    """

    data = convert_tweet_2_data(path, rnd)
    # data = minimum_size_sraa(data)

    return data


def load_arxiv(path, category=None, subset="all", shuffle=True, rnd=2356, percent=.5):
    """
    load text files from Aviation-auto dataset from folders to memory. It will return a 25-75 percent train test split
    :param path: path of the root directory of the data
    :param subset: what data will be loaded, train or test or all
    :param shuffle:
    :param rnd: random seed value
    :param vct: vectorizer
    :return: :raise ValueError:
    """


    categories = dict(ml=['cs.AI', 'stat.ML'], db=['cs.DB', 'cs.IR'],
                      ne=['cs.NE', 'cs.SI'])

    cat = None
    if category is not None:
        cat = categories[category]

    data = bunch.Bunch()

    if subset in ('train', 'test'):
        raise ValueError("We are not ready for train test aviation data yet")
    elif subset == "all":
        data = load_files(path, encoding="latin1", load_content=True, random_state=rnd, categories=cat)
        data.data = np.array(data.data, dtype=object)
    else:
        raise ValueError("Subset can only be 'train', 'test' or 'all', got '%s'" % subset)

    indices = ShuffleSplit(len(data.data), n_iter=1, test_size=percent, random_state=rnd)
    for train_ind, test_ind in indices:
        data = bunch.Bunch(train=bunch.Bunch(data=data.data[train_ind], target=data.target[train_ind],
                                             filenames=data.filenames[train_ind], target_names=data.target_names),
                           test=bunch.Bunch(data=data.data[test_ind], target=data.target[test_ind],
                                            filenames=data.filenames[test_ind], target_names=data.target_names))

    if shuffle:
        random_state = np.random.RandomState(rnd)
        indices = np.arange(data.train.target.shape[0])
        random_state.shuffle(indices)
        data.train.filenames = data.train.filenames[indices]
        data.train.target = data.train.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.train.data, dtype=object)
        data_lst = data_lst[indices]
        data.train.data = data_lst
        data.test.data = np.array(data.test.data, dtype=object)

    data = minimum_size(data, min_size=1)
    return data

def load_amazon(path, shuffle=True, rnd=2356, percent=.5):
    """
    load text files from Aviation-auto dataset from folders to memory. It will return a 25-75 percent train test split
    :param path: path of the root directory of the data
    :param subset: what data will be loaded, train or test or all
    :param shuffle:
    :param rnd: random seed value
    :param vct: vectorizer
    :return: :raise ValueError:
    """

    data = bunch.Bunch()

    try:
        if isfile(path + "/amazon_sampled_target.pkl") and isfile(path + "/amazon_sampled_text.txt.gz"):
            targets = np.array(pickle.load(open(path + "/amazon_sampled_target.pkl", 'rb')))
            # text = np.array([d.decode('latin1') for d in gzip.open(path + "/amazon_sampled_text.txt.gz", 'rt').readlines()])
            with gzip.open(path + "/amazon_sampled_text.txt.gz", 'r') as f:
                text = np.array([line.decode('latin1') for line in f])
        else:
            raise IOError("Oops, one of the files is not here %s" % path)
    except Exception as excp:
        raise ValueError("Oops, We cannot load the data, something happend")

    indices = ShuffleSplit(len(text), n_iter=1, test_size=percent, random_state=rnd)
    for train_ind, test_ind in indices:
        data = bunch.Bunch(train=bunch.Bunch(data=text[train_ind], target=targets[train_ind],
                                             target_names=['neg','pos']),
                           test=bunch.Bunch(data=text[test_ind], target=targets[test_ind],
                                            target_names=['neg','pos']))

    if shuffle:
        random_state = np.random.RandomState(rnd)
        indices = np.arange(data.train.target.shape[0])
        random_state.shuffle(indices)
        data.train.target = data.train.target[indices]
        data.train.data = data.train.data[indices]

        data.test.data = np.array(data.test.data, dtype=object)

    return data


# def load_dataset(name, path, categories=None, rnd=2356, shuffle=True, percent=.5, keep_subject=False, labels=None):
def load_dataset(name, path, **kwargs):
    data = bunch.Bunch()
    categories=None
    rnd=2356
    shuffle=True
    percent=.5
    keep_subject=False
    labels=None

    if 'categories' in kwargs.keys():
        categories = kwargs['categories']
    if 'rnd' in kwargs.keys():
        rnd = kwargs['rnd']
    if 'shuffle' in kwargs.keys():
        shuffle = kwargs['shuffle']
    if 'percent' in kwargs.keys():
        percent = kwargs['percent']
    if 'keep_subject' in kwargs.keys():
        keep_subject = kwargs['keep_subject']
    if 'labels' in kwargs.keys():
        labels = kwargs['labels']


    if "imdb" == name:
        ########## IMDB MOVIE REVIEWS ###########
        data = load_imdb(path, shuffle=shuffle, rnd=rnd)  # should bring data as is
    elif "20news" in name:
        ########## 20 news groups ######
        data = load_20newsgroups(category=categories, shuffle=shuffle, rnd=rnd)
    elif "arxiv" in name:
        ##########  arxiv dataset ######
        data = load_arxiv(path, category=categories, subset='all', shuffle=shuffle, rnd=rnd, percent=percent)
    elif "amazon" in name:
        ##########  arxiv dataset ######
        data = load_amazon(path, shuffle=shuffle, rnd=rnd, percent=percent)
    elif "twitter" in name:
        ##########  arxiv dataset ######
        data = load_twitter(path, shuffle=shuffle, rnd=rnd, percent=percent)
    else:
        raise Exception("We do not know {} dataset".format(name.upper()))

    return data