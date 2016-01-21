__author__ = 'maru'

import numpy as np
import itertools as it


class SnippetTokenizer(object):

    def __init__(self, k=(1,1)):
        import nltk
        self.sent_tk = nltk.data.load('tokenizers/punkt/english.pickle')
        self.k = k
        self.separator = " "
        self.split_bound = '\\b\\w+\\b'

    def tokenize_sents(self, doc):
        return [self.tokenize(sent) for sent in doc]

    def tokenize(self, doc):
        return self.get_sentences(doc, self.k)

    def set_snippet_size(self, k):
        self.k = k

    def __str__(self):
        return self.__class__.__name__

    def get_combinations(self, n, k):
        return it.combinations(range(n), k)

    def get_combination_pairs(self, sentences):
        all_sents = []
        k = self.k
        n = min(len(sentences[:30])+1, k[1]+1)

        if min(n, k[0]) > len(sentences[:30]):
            return sentences
        sentences = np.array(sentences)
        for i in range(k[0],n):
            pairs = self.get_combinations(len(sentences[:30]), i)
            all_sents.extend([p for p in pairs])
        return all_sents

    def get_sentences_k(self, sentences, k):
        all_sents = []
        n = min(len(sentences[:30])+1, k[1]+1)

        if min(n, k[0]) > len(sentences[:30]):
            return sentences
        sentences = np.array(sentences)
        for i in range(k[0],n):
            pairs = self.get_combinations(len(sentences[:30]), i)
            all_sents.extend([self.separator.join(sentences[list(p)]) for p in pairs])

        return all_sents

    def get_sentences_k_v0(self, sentences, k):

        all_sents = []
        n = min(len(sentences[:30])+1, k[1]+1)
        if min(n, k[0]) > len(sentences[:30]):
            return sentences

        for i in range(k[0],n):
            pairs = it.combinations(sentences[:30], i)
            all_sents.extend([self.separator.join(p) for p in pairs])

        return all_sents

    def get_sentences(self, doc, k):

        d_sent = self.sent_tk.tokenize_sents([doc])
        text_min = []
        for sentences in d_sent[0]:
            if len(sentences.strip()) > 2:
                text_min.append(sentences)  # at least 2 characters

        return self.get_sentences_k(text_min, k)


class First1SnippetTokenizer(SnippetTokenizer):

    def __init__(self, k=(1,1)):
        super(First1SnippetTokenizer, self).__init__(k=k)

    def get_sentences_k(self, sentences, k):

        all_sents = []
        sentences= np.array(sentences)
        pairs = self.get_combination_pairs(sentences[:30])
        all_sents.extend([self.separator.join(sentences[p]) for p in pairs])
        return all_sents

    def get_combination_pairs(self, sents):
        return [range(0, min(self.k[1], len(sents)))]


class Random1SnippetTokenizer(SnippetTokenizer):

    def __init__(self, k=(1,1), seed=5432):
        super(Random1SnippetTokenizer,self).__init__(k)
        self.rnd = np.random.RandomState(seed)

    def get_sentences_k(self, sentences, k):

        n = len(sentences[:30])
        all_sents = []

        pairs = it.combinations(sentences[:30], min(k[1], n))
        for p in pairs:
            all_sents.append(self.separator.join(p))

        pick = self.rnd.randint(0,len(all_sents),1)
        return [all_sents[pick]]

    def get_combination_pairs(self, sents):
        all_sents = []
        k = self.k
        n = min(len(sents[:30])+1, k[1]+1)

        if min(n, k[0]) > len(sents[:30]):
            return range(0, len(sents[:30]))
        sentences = np.array(sents)
        for i in range(k[0],n):
            pairs = self.get_combinations(len(sentences[:30]), i)
            all_sents.extend([p for p in pairs])

        pick = self.rnd.randint(0,len(all_sents),1)
        return [all_sents[pick]]


class WindowSnippetTokenizer(SnippetTokenizer):

    def __init__(self, k=(1,1)):
        super(WindowSnippetTokenizer, self).__init__(k=k)

    def get_sentences_k(self, sentences, k):

        n = len(sentences[:30])
        all_sents = []
        sentences = np.array(sentences)
        pairs = self.get_combination_pairs(sentences)
        for p in pairs:
            all_sents.append(self.separator.join(sentences[p]))

        return all_sents

    def get_combinations(self, n, k):
        ch = range(n)
        ws = min(n, k)
        all_pairs =[]
        for c in range(n-ws+1):
            all_pairs.append(ch[c:c+ws])
        return all_pairs

    def get_combination_pairs(self, sents):
        all_sents = []
        k = self.k
        n = min(len(sents[:30])+1, k[1]+1)

        if min(n, k[0]) > len(sents[:30]):
            return range(0, len(sents[:30]))

        sentences = np.array(sents)
        for i in range(k[0],n):
            pairs = self.get_combinations(len(sentences[:30]), i)
            all_sents.extend([p for p in pairs])

        return all_sents


class FirstWindowSnippetTokenizer(SnippetTokenizer):

    def __init__(self, k=(1,1)):
        super(FirstWindowSnippetTokenizer, self).__init__(k=k)

    def get_sentences_k(self, sentences, k):

        n = len(sentences[:30])
        all_sents = []
        sentences = np.array(sentences)
        pairs = self.get_combination_pairs(sentences)
        for p in pairs:
            all_sents.append(self.separator.join(sentences[p]))

        return all_sents

    def get_combinations(self, n, k):
        ws = min(n, k)
        all_pairs =[]

        for c in range(ws):
            all_pairs.append(range(c+1))

        return all_pairs

    def get_combination_pairs(self, sents):
        all_sents = []
        k = self.k
        n = min(len(sents[:30])+1, k[1]+1)

        if min(n, k[0]) > len(sents[:30]):
            return self.get_combinations(len(sents[:30]),k[0])

        sentences = np.array(sents)
        for i in range(k[0],n):
            pairs = self.get_combinations(len(sentences[:30]), i)
            all_sents.extend([p for p in pairs])

        return all_sents
