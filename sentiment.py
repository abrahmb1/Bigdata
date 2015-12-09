import re
import pymongo
from pymongo import MongoClient
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

    # Connect to annotated collection
    def connect_to_annotated_tweets(self):
        client = MongoClient('localhost', 27017)
        db = client['tweet_database']
        annotated_tweet_collection = db['annotated']
        return annotated_tweet_collection


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
        return featureList

    def evaluate(self):
        annotated = self.connect_to_annotated_tweets()
	trainData = []
	testData = []
	count = 0
	for tweet in annotated.find():
	    count += 1
	    if count <= 1000:
		trainData.append((tweet['text'].encode('ascii', 'ignore'),tweet['sentiment']))
	    else:
		testData.append((tweet['text'].encode('ascii','ignore'),tweet['sentiment'])) 
        print count
	tweets = []
        for (tweet, sentiment) in trainData:
	    lower = tweet.lower()
            text = re.sub( '\s+', ' ', lower ).strip()
            words = text.split()
            features = [self.featureExtract(words), sentiment]
	    tweets.append(features)
        classifier = NaiveBayesClassifier.train(tweets)

        # testing
        #referenceSets = dict()
        #testSets = dict()
        i=0
	tweets = []
	for (tweet, sentiment) in testData:
            lower = tweet.lower()
            text = re.sub( '\s+', ' ', lower ).strip()
            words = text.split()
            features = [self.featureExtract(words), sentiment]
            tweets.append(features)
	correct = 0
        for j, (features, label) in enumerate(tweets):
            predicted = classifier.classify(features)
            if predicted == label:
		correct += 1
            i += 1
	print "Correct = " + str(correct)
	print "Total = " + str(i)
	print "Accuracy = " + str(float(correct)/i)
c = TwittElection()
c.evaluate()

