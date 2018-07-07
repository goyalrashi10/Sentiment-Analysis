'''
Created on Oct 30, 2016

@author:         Rashi Goyal  (Student Id: 1668230) 
@summary:    Python program to use  Naive Bayes, Logistic Regression, Support Vector Matrix classifiers on Amazon Review Data
@summary:    Sentiment Analysis of Amazon products Review Data.
'''
import csv;
import re, random;
import os;
import pathlib;
from nltk.stem.snowball import SnowballStemmer;
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC

from nltk.metrics import precision;
from nltk.metrics import recall;
import nltk, nltk.classify.util;
from sklearn.metrics import classification_report;
from sklearn.metrics import confusion_matrix;

import collections;

class ReadFiles(object):

    def loadCleanseList(self):
                
        global cleanseList;
        cleanseList=[];

        path=pathlib.Path(os.path.dirname(os.path.abspath(__file__)));
        
        print("Uploading cleanse list.........");
        for word in open(str(path.parent)+"/determiners.txt","r").readlines():
                if word not in cleanseList:
                    word=str(word.replace("\n",""));
                    cleanseList.append(word.lower());
        for word in open(str(path.parent)+"/preposition.txt","r").readlines():               
                if word not in cleanseList:
                    word=str(word.replace("\n",""));
                    cleanseList.append(word.lower());
        for word in open(str(path.parent)+"/conjuctions.txt","r").readlines():               
                if word not in cleanseList:
                    word=str(word.replace("\n",""));
                    cleanseList.append(word.lower());            
        
    def readPositiveReviews(self):
        
        global posSentences;
        posSentences=[];
    
        print("Reading positive Reviews file.........")

        path=pathlib.Path(os.path.dirname(os.path.abspath(__file__)));
        with open(str(path.parent)+"/positiveReviews.csv",encoding="utf-8",errors='ignore') as csvfile:
               

            reader = csv.DictReader(csvfile)
        
            rowCounter=0
            
            for row in reader:
                if rowCounter<=20000:
                    posSentences.append(row["Text"]);
                else:
                    break;        
                rowCounter=rowCounter+1;
                        
    def readNegativeReviews(self):
        
        global negSentences;
        negSentences=[];
                
        print("Reading Negative Reviews file.........")
        path=pathlib.Path(os.path.dirname(os.path.abspath(__file__)));
        with open(str(path.parent)+"/NegativeReviews.csv",encoding="utf-8",errors='ignore') as csvfile:
            reader = csv.DictReader(csvfile)
        
            rowCounter=0
            
            for row in reader:
                if rowCounter<=20000:
                    negSentences.append(row["Text"]);
                else:
                    break;        
                rowCounter=rowCounter+1;
            
    def extractFeatures(self):
        
        global posFeatures;
        posFeatures=[];
        global negFeatures;
        negFeatures=[];

        stemmer = SnowballStemmer("english");

        print("Extracting Features.........");
        for sentence in posSentences:
            sentence=re.sub('[^a-zA-Z]', ' ', sentence.lower());
            posWords = re.findall(r"[\w']+|[.,!?;]", sentence.rstrip());
            posWords = [dict([(stemmer.stem(word), True) for word in posWords]), 'Positive'];
            posFeatures.append(posWords);

        for sentence in negSentences:
            sentence=re.sub('[^a-zA-Z]', ' ', sentence.lower());
            negWords = re.findall(r"[\w']+|[.,!?;]", sentence.rstrip());
            negWords = [dict([(stemmer.stem(word), True) for word in negWords]), 'Negative'];
            negFeatures.append(negWords);

    def splitFeatures(self):
        
        print("Splitting Features into Training & Test Features.........");
        posFtrCount=int((3/4)*len(posFeatures));
        negFtrCount=int((3/4)*len(negFeatures));
        
        global trainFeatures;
        trainFeatures=posFeatures[:posFtrCount] + negFeatures[:negFtrCount];
        
        random.shuffle(trainFeatures);
        
        global testFeatures;
        testFeatures=posFeatures[posFtrCount:] + negFeatures[negFtrCount:];

        random.shuffle(testFeatures);

    def printTrainFeatures(self):
        
        for feature,label in trainFeatures:
            print(feature+"--->"+label);
        
    def ImplementMNB(self):
        
#         classifier = NaiveBayesClassifier.train(trainFeatures)    
        print("~~~~~~~~~~~~~~~ MultinomialNB Classifier ~~~~~~~~~~~~~~~\n");
        classifier = SklearnClassifier(MultinomialNB());
        classifier.train(trainFeatures);
    
        print("MultinomialNB Classifier Training Completed");

        #initiates referenceSets and testSets
        referenceSets = collections.defaultdict(set)
        testSets = collections.defaultdict(set)    
        expected_array=[];
        predicted_array=[];
    
        #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
        for i, (features, label) in enumerate(testFeatures):
            referenceSets[label].add(i)
            expected_array.append(label);
            predicted = classifier.classify(features)
            predicted_array.append(predicted);
            testSets[predicted].add(i)    
    
        print("MultinomialNB Classifier Test Results ");
        print("");
        #prints metrics to show how well the feature selection did
        print ("Length of Training Features" +str(len(trainFeatures)));
        print ("Length of Test Features" +str(len(testFeatures)));
        print ('Accuracy:' + str(nltk.classify.util.accuracy(classifier, testFeatures)));
        print ('Positive precision:', str(precision(referenceSets['Positive'], testSets['Positive'])));
        print ('Positive recall:', str(recall(referenceSets['Positive'], testSets['Positive'])));
        print ('Negative precision:', str(precision(referenceSets['Negative'], testSets['Negative'])));
        print ('Negative recall:', str(recall(referenceSets['Negative'], testSets['Negative'])));
        print ("~~~~~~~~~~~~~~~Classification report~~~~~~~~~~~~~~~\n", classification_report(expected_array, predicted_array));
        print ("~~~~~~~~~~~~~~~Confusion matrix~~~~~~~~~~~~~~~\n",confusion_matrix(expected_array, predicted_array));
        print("");
#         classifier.show_most_informative_features(30);

    def ImplementBNB(self):
        
        print("~~~~~~~~~~~~~~~ BernoulliNB Classifier ~~~~~~~~~~~~~~~\n");
#         classifier = NaiveBayesClassifier.train(trainFeatures)    
        classifier = SklearnClassifier(BernoulliNB());
        classifier.train(trainFeatures);
    
        print("BernoulliNB Classifier Training Completed");

        #initiates referenceSets and testSets
        referenceSets = collections.defaultdict(set)
        testSets = collections.defaultdict(set)    
        expected_array=[];
        predicted_array=[];
    
        #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
        for i, (features, label) in enumerate(testFeatures):
            referenceSets[label].add(i)
            expected_array.append(label);
            predicted = classifier.classify(features)
            predicted_array.append(predicted);
            testSets[predicted].add(i)    
    
        #prints metrics to show how well the feature selection did
        print("BernoulliNB Classifier Test Results ");
        print("");
        print ("Length of Training Features" +str(len(trainFeatures)));
        print ("Length of Test Features" +str(len(testFeatures)));
        print ('Accuracy:' + str(nltk.classify.util.accuracy(classifier, testFeatures)));
        print ('Positive precision:', str(precision(referenceSets['Positive'], testSets['Positive'])));
        print ('Positive recall:', str(recall(referenceSets['Positive'], testSets['Positive'])));
        print ('Negative precision:', str(precision(referenceSets['Negative'], testSets['Negative'])));
        print ('Negative recall:', str(recall(referenceSets['Negative'], testSets['Negative'])));
        print ("~~~~~~~~~~~~~~~Classification report~~~~~~~~~~~~~~~\n", classification_report(expected_array, predicted_array));
        print ("~~~~~~~~~~~~~~~Confusion matrix~~~~~~~~~~~~~~~\n",confusion_matrix(expected_array, predicted_array));
        print("");

    def ImplementSVC(self):
        
        print("~~~~~~~~~~~~~~~  SVC Classifier ~~~~~~~~~~~~~~~\n");

        classifier = nltk.classify.SklearnClassifier(SVC())
        classifier.train(trainFeatures);
    
        print("SVC Classifier Training Completed");

        #initiates referenceSets and testSets
        referenceSets = collections.defaultdict(set)
        testSets = collections.defaultdict(set)    
        expected_array=[];
        predicted_array=[];
   
        #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
        for i, (features, label) in enumerate(testFeatures):
            referenceSets[label].add(i)
            expected_array.append(label);
            predicted = classifier.classify(features)
            predicted_array.append(predicted);
            testSets[predicted].add(i)    
    
        #prints metrics to show how well the feature selection did
        print("SVC Classifier Test Results ");
        print("");
        print ("Length of Training Features" +str(len(trainFeatures)));
        print ("Length of Test Features" +str(len(testFeatures)));
        print ('Accuracy:' + str(nltk.classify.util.accuracy(classifier, testFeatures)));
        print ('Positive precision:', str(precision(referenceSets['Positive'], testSets['Positive'])));
        print ('Positive recall:', str(recall(referenceSets['Positive'], testSets['Positive'])));
        print ('Negative precision:', str(precision(referenceSets['Negative'], testSets['Negative'])));
        print ('Negative recall:', str(recall(referenceSets['Negative'], testSets['Negative'])));
        print ("~~~~~~~~~~~~~~~Classification report~~~~~~~~~~~~~~~\n", classification_report(expected_array, predicted_array));
        print ("~~~~~~~~~~~~~~~Confusion matrix~~~~~~~~~~~~~~~\n",confusion_matrix(expected_array, predicted_array));
        print("");

    def ImplementLSVC(self):
        
        print("~~~~~~~~~~~~~~~ Linear SVC Classifier ~~~~~~~~~~~~~~~\n");

        classifier = nltk.classify.SklearnClassifier(LinearSVC())
        classifier.train(trainFeatures);
    
        print("Linear SVC Classifier Training Completed");

        #initiates referenceSets and testSets
        referenceSets = collections.defaultdict(set)
        testSets = collections.defaultdict(set)    
        expected_array=[];
        predicted_array=[];
    
        #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
        for i, (features, label) in enumerate(testFeatures):
            referenceSets[label].add(i)
            expected_array.append(label);
            predicted = classifier.classify(features)
            predicted_array.append(predicted);
            testSets[predicted].add(i)    
    
        #prints metrics to show how well the feature selection did
        print("Linear SVC Classifier Test Results ");
        print("");
        print ("Length of Training Features" +str(len(trainFeatures)));
        print ("Length of Test Features" +str(len(testFeatures)));
        print ('Accuracy:' + str(nltk.classify.util.accuracy(classifier, testFeatures)));
        print ('Positive precision:', str(precision(referenceSets['Positive'], testSets['Positive'])));
        print ('Positive recall:', str(recall(referenceSets['Positive'], testSets['Positive'])));
        print ('Negative precision:', str(precision(referenceSets['Negative'], testSets['Negative'])));
        print ('Negative recall:', str(recall(referenceSets['Negative'], testSets['Negative'])));
        print ("~~~~~~~~~~~~~~~Classification report~~~~~~~~~~~~~~~\n", classification_report(expected_array, predicted_array));
        print ("~~~~~~~~~~~~~~~Confusion matrix~~~~~~~~~~~~~~~\n",confusion_matrix(expected_array, predicted_array));
        print("");

    def ImplementLR(self):
        
        print("~~~~~~~~~~~~~~~ Logistic Regression Classifier ~~~~~~~~~~~~~~~\n");
        
        classifier = nltk.classify.SklearnClassifier(LogisticRegression())
        classifier.train(trainFeatures);
    
        print("Logistic Regression Classifier Training Completed");

        #initiating referenceSets and testSets, Expected Array & Predicted Array for confusion matrix
        referenceSets = collections.defaultdict(set)
        testSets = collections.defaultdict(set)    
        expected_array=[];
        predicted_array=[];
    
        #assigning referenceSets & expected Array with correct labels & testsSets and predicted_array with predicted labels
        for i, (features, label) in enumerate(testFeatures):
            referenceSets[label].add(i)
            expected_array.append(label);
            predicted = classifier.classify(features)
            predicted_array.append(predicted);
            testSets[predicted].add(i)    
#             print ("\nLabel---->"+label+" i ----> "+str(i)+" referenceSet Label---->"+str(referenceSets[label])+"\n\n\n\n\n");
    
        #prints metrics to show how well the feature selection did
        print("Logistic Regression Classifier Test Results ");
        print("");
        print ("Length of Training Features" +str(len(trainFeatures)));
        print ("Length of Test Features" +str(len(testFeatures)));
        print ('Accuracy:' + str(nltk.classify.util.accuracy(classifier, testFeatures)));
        print ('Positive precision:', str(precision(referenceSets['Positive'], testSets['Positive'])));
        print ('Positive recall:', str(recall(referenceSets['Positive'], testSets['Positive'])));
        print ('Negative precision:', str(precision(referenceSets['Negative'], testSets['Negative'])));
        print ('Negative recall:', str(recall(referenceSets['Negative'], testSets['Negative'])));
        print ("~~~~~~~~~~~~~~~Classification report~~~~~~~~~~~~~~~\n", classification_report(expected_array, predicted_array));
        print ("~~~~~~~~~~~~~~~Confusion matrix~~~~~~~~~~~~~~~\n",confusion_matrix(expected_array, predicted_array));
        print("plotting confusion Matrix");


def main(self):
    
    print("running main method.........")
    
    classObject=ReadFiles();
    classObject.loadCleanseList();
    classObject.readPositiveReviews();
    classObject.readNegativeReviews();
    classObject.extractFeatures();    
    classObject.splitFeatures();
    classObject.ImplementMNB();
#     classObject.ImplementBNB();
    classObject.ImplementLSVC();
    classObject.ImplementLR();
    
main(ReadFiles);