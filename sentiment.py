import re
import nltk, nltk.classify.util
from nltk.classify import NaiveBayesClassifier
# search patterns for features
class TwittElection:
    def __init__(self):
        self.testFeatures = \
            [('hasAddict',     (' addict',)), \
            ('hasAwesome',    ('awesome',)), \
             ('hasBroken',     ('broke',)), \
             ('hasBad',        (' bad',)), \
             ('hasBug',        (' bug',)), \
             ('hasCant',       ('cant','can\'t')), \
             ('hasCrash',      ('crash',)), \
             ('hasCool',       ('cool',)), \
             ('hasDifficult',  ('difficult',)), \
             ('hasDisaster',   ('disaster',)), \
             ('hasDown',       (' down',)), \
             ('hasDont',       ('dont','don\'t','do not','does not','doesn\'t')), \
             ('hasEasy',       (' easy',)), \
             ('hasExclaim',    ('!',)), \
             ('hasExcite',     (' excite',)), \
             ('hasExpense',    ('expense','expensive')), \
             ('hasFail',       (' fail',)), \
             ('hasFast',       (' fast',)), \
             ('hasFix',        (' fix',)), \
             ('hasFree',       (' free',)), \
             ('hasFrowny',     (':(', '):')), \
             ('hasFuck',       ('fuck',)), \
             ('hasGood',       ('good','great')), \
             ('hasHappy',      (' happy',' happi')), \
             ('hasHate',       ('hate',)), \
             ('hasHeart',      ('heart', '<3')), \
             ('hasIssue',      (' issue',)), \
             ('hasIncredible', ('incredible',)), \
             ('hasInterest',   ('interest',)), \
             ('hasLike',       (' like',)), \
             ('hasLol',        (' lol',)), \
             ('hasLove',       ('love','loving')), \
             ('hasLose',       (' lose',)), \
             ('hasNeat',       ('neat',)), \
             ('hasNever',      (' never',)), \
             ('hasNice',       (' nice',)), \
             ('hasPoor',       ('poor',)), \
             ('hasPerfect',    ('perfect',)), \
             ('hasPlease',     ('please',)), \
             ('hasSerious',    ('serious',)), \
             ('hasShit',       ('shit',)), \
             ('hasSlow',       (' slow',)), \
             ('hasSmiley',     (':)', ':d', '(:')), \
             ('hasSuck',       ('suck',)), \
             ('hasTerrible',   ('terrible',)), \
             ('hasThanks',     ('thank',)), \
             ('hasTrouble',    ('trouble',)), \
             ('hasUnhappy',    ('unhapp',)), \
             ('hasWin',        (' win ','winner','winning')), \
             ('hasWinky',      (';)',)), \
             ('hasWow',        ('wow','omg')) ]
        self.stopwords = list()
        with open("stopwords.txt") as f:
            for line in f:
                word = line.strip()
                self.stopwords.append(word)


    def featureExtract(self,words):
        featureList = {}
        for test in self.testFeatures:
            key = test[0]
            featureList[key] = False
            for value in test[1]:
                if (value in words):
                    featureList[key] = True
                    words.remove(value)
        for word in words:
            if len(word)>2 and not word in self.stopwords:
                featureList[word] = True
        print featureList
        return featureList

    def evaluate(self):
        with open('temp','r') as document:
            posTweets = []
            for sentence in document:
                lower = sentence.lower()
                text = re.sub( '\s+', ' ', lower ).strip()
                words = text.split()
                tweet = [self.featureExtract(words), 'pos']
                posTweets.append(tweet)
        classifier = NaiveBayesClassifier.train(posTweets)

        # testing
        referenceSets = dict()
        testSets = dict()
        i=1
        for j, (features, label) in enumerate(posTweets):
            referenceSets[i] = label
            predicted = classifier.classify(features)
            testSets[i] = predicted
            i += 1
        print referenceSets
        print testSets

c = TwittElection()
c.evaluate()

