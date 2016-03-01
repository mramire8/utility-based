from sklearn.datasets import load_files
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import base as bunch
import numpy as np


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

    data = minimum_size(data, min_size=100)

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


def bunch_users(class1, class2, lowercase, collapse_urls, collapse_mentions, rnd, class_name=None):
    labels = None
    if labels is None:
        labels = [0, 1]

    user_id, user_names, timeline = user_to_doc(class1, lowercase, collapse_urls, collapse_mentions)
    user_id2, user_names2, timeline2 = user_to_doc(class2, lowercase, collapse_urls, collapse_mentions)
    target = [labels[0]] * len(user_id)
    user_id.extend(user_id2)
    user_names.extend(user_names2)
    timeline.extend(timeline2)
    target.extend([labels[1]] * len(user_id2))
    user_text = ["######".join(t) for t in timeline]
    data = bunch.Bunch(data=user_text, target=target, user_id=user_id,
                       user_name=user_names, user_timeline=timeline)

    random_state = np.random.RandomState(rnd)

    indices = np.arange(len(data.target))
    random_state.shuffle(indices)
    data.target = np.array(data.target)[indices]
    data_lst = np.array(data.data, dtype=object)
    data_lst = data_lst[indices]
    data.data = data_lst
    data.user_id = np.array(data.user_id)[indices]
    data.user_name = np.array(data.user_name)[indices]
    data.user_timeline = np.array(data.user_timeline)[indices]
    data.target_names = class_name
    return data


def get_date(date_str):
    import datetime

    return datetime.datetime.strptime(date_str.strip('"'), "%a %b %d %H:%M:%S +0000 %Y")


def convert_tweet_2_data(data_path, rnd):
    """
    Convert tweet time lines into dataset
    :param data_path:
    :param vct:
    :return: bunch.Bunch
        Bunch with the data in train and test from twitter bots and human accounts
    """
    good = get_tweets_file(data_path + "/good.json")

    bots = get_tweets_file(data_path + "/bots.json")

    gds = [g for g in good if get_date(g[0]['created_at']).year > 2013]
    bts = [b for b in bots if get_date(b[0]['created_at']).year > 2013]

    data = bunch_users(gds, bts, True, True, True, rnd, class_name=['good', 'bots'])

    return data


def get_tweets_file(path):
    import json

    f = open(path)

    i = 0
    users = []
    data = []
    last = 0
    for line in f:
        data = line.split("]][[")
        last = len(data)

    for i, tweets in enumerate(data):
        if i == 0:
            t = json.loads(tweets[1:] + "]")
        elif i == (last - 1):
            t = json.loads("[" + tweets[:-1])
        else:
            t = json.loads("[" + tweets + "]")
        users.append(t)

    return users


def load_twitter(path, shuffle=True, rnd=1):
    """
    load text files from twitter data
    :param path: path of the root directory of the data
    :param subset: what data will be loaded, train or test or all
    :param shuffle:
    :param rnd: random seed value
    :param vct: vectorizer
    :return: :raise ValueError:
    """

    data = bunch.Bunch()

    data = convert_tweet_2_data(path, rnd)
    data = minimum_size_sraa(data)

    if shuffle:
        random_state = np.random.RandomState(rnd)
        indices = np.arange(data.target.shape[0])
        random_state.shuffle(indices)
        data.target = data.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[indices]
        data.data = data_lst

    return data


# ARXIV_HOME = 'C:/Users/mramire8/Documents/Datasets/arxiv'
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
    from sklearn.cross_validation import ShuffleSplit

    categories = dict(ml=['cs.AI', 'cs.LG'], db=['cs.DB', 'cs.IR'],
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

    # data = minimum_size(data)
    return data


def load_dataset(name, path, categories=None, rnd=2356, shuffle=True, percent=.5, keep_subject=False, labels=None):
    data = bunch.Bunch()

    if "imdb" == name:
        ########## IMDB MOVIE REVIEWS ###########
        data = load_imdb(path, shuffle=shuffle, rnd=rnd)  # should bring data as is
    elif "20news" in name:
        ########## 20 news groups ######
        data = load_20newsgroups(category=categories, shuffle=shuffle, rnd=rnd)
    elif "arxiv" in name:
        ##########  arxiv dataset ######
        data = load_arxiv(path, category=categories, subset='all', shuffle=shuffle, rnd=rnd, percent=percent)
    else:
        raise Exception("We do not know {} dataset".format(name.upper()))

    return data