
"""
Created on Sun Feb 17 21:00 2019

lab1.py: a script to do the text classification

Notice: I use the average weight for unigram to test , so I use vector to get the weight
       While I use the last weight for bigram and trigram to test, soo I use dictionary to get the weight


@author: Lei Zhao
"""
import sys,re,getopt
import glob,random,time
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
# indicate the output of plotting function is printed to the notebook

#get the CommandLine input
class CommandLine:

    def __init__(self):
        opts,args=getopt.getopt(sys.argv[1:],'')
        root=args[0]
        print(root)
        self.rootPos=root+'/txt_sentoken/pos/*.txt'
        self.rootNeg=root+'/txt_sentoken/neg/*.txt'

    #return the complete path for Pos and Neg
    def getPath(self):
        return self.rootPos,self.rootNeg


class TextClassification:

    def __init__(self,rootPos,rootNeg,BOWtype):

        self.rootPos=rootPos
        self.rootNeg=rootNeg
        self.BOWtype=BOWtype

        #download from google
        self.stopWord=("film","a", "about", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "did", "do", "does", "doing", "down", "during", "each", "for", "from", "had", "has", "have", "having", "he", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",  "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "my", "myself", "nor", "of", "on", "once", "or", "other", "ought", "our", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they're", "they've", "this", "those", "through", "to", "until", "up", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "who", "who's", "why", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves")

        self.reviewLabel={}

        self.maxIter=20

        self.totalIndex={} #record the index of word in acurracy
        self.switchIdex={} #switch the key and value in self.totalIndex, in order to get top features

        self.weightSum={} #sum the weight in order to get the average

        self.allTrainReviews=[]

        self.allTestReviews=[]

        self.weight={} #the weight for each word in trainning set

        self.error=np.zeros(self.maxIter) #the error for the prediction in trainning set in order to get learning progress

#==============================================================================================================================
#this part is for unigram
    def tokeniForTestUni(self,path,leftRange,rightRange,label):
        try:
            for file in glob.glob(path)[leftRange:rightRange]:
                with open(file,'r') as f:
                    listWords=re.sub("[^\w']"," ",f.read()).split()
                    listWithoutStop=[item for item in listWords if item not in self.stopWord]
                    self.reviewLabel[str(listWithoutStop)]=label
                self.allTestReviews.append(listWithoutStop)
        except FileNotFoundError:
            print('Fail to find file')


    def tokeniForTrainUni(self,path,leftRange,rightRange,label):
        try:
            for file in glob.glob(path)[leftRange:rightRange]: #read the file in the path
                with open(file,'r') as f:
                    listWords=re.sub("[^\w']"," ",f.read()).split()
                    listWithoutStop=[item for item in listWords if item not in self.stopWord] #get rid of stop word
                    for word in listWithoutStop:
                        if word not in self.totalIndex:
                            self.totalIndex[word]=len(self.totalIndex)
                self.reviewLabel[str(listWithoutStop)]=label
                self.allTrainReviews.append(listWithoutStop)

                self.switchIdex={y:x for x,y in self.totalIndex.items()}

        except FileNotFoundError:
            print('Fail to find file')


    def trainForUni(self):
        result=lambda s: -1 if s<0 else 1  #classify the score to -1 and 1
        weightSum=np.zeros(self.trainLen)

        c=0 #the counter

        for i in range(self.maxIter):
            random.Random(i).shuffle(self.allTrainReviews)
            for reviews in self.allTrainReviews:
                tf=np.zeros(self.trainLen) #initialise the vector for each review
                score=0
                Count=Counter(reviews) #get the count of each word in every review
                for word,counts in Count.items():
                    wordIndex=self.totalIndex[word]
                    tf[wordIndex]=counts
                score=np.dot(self.weight,tf)

                if self.reviewLabel[str(reviews)]!=result(score):
                    self.error[i]+=1#when prediction different from label, plus 1. In order to get the learning progress
                    self.weight+=self.reviewLabel[str(reviews)]*tf #

                weightSum+=self.weight  #sum the weight
                c+=1  #counter+1 every time

            if self.error[i]==0: #when the model cannot learn from training set,break
                break

        weightSum=weightSum/c #average the weight, instead of using last one
        self.weight=weightSum

    def getTopFeatureUni(self):
        top=np.argsort(self.weight)[-10:] #get the index of positively 10 in weight vector
        pos10=[self.switchIdex[item] for item in top]

        top=np.argsort(self.weight)[:10] #get the index of negetively 10 in weight vector
        neg10=[self.switchIdex[item] for item in top]

        print 'pos10 for '+self.BOWtype+' is',pos10
        print 'neg10 for '+self.BOWtype+' is',neg10
        print '\n'



    def testForUni(self):
        result=lambda s: -1 if s<0 else 1 #classify the score to -1 and 1
        precision=0
        for reviews in self.allTestReviews:
            tf=np.zeros(self.trainLen)
            score=0
            Count=Counter(reviews) #get the count of each word in every review
            for word in Count:
                if word in self.totalIndex.keys():
                    wordIndex=self.totalIndex[word]
                    tf[wordIndex]=Count[word]
            score=np.dot(tf,self.weight)
            if result(score)==self.reviewLabel[str(reviews)]:
                precision+=1
        print 'acurracy for '+self.BOWtype+' is',precision/400.0

#==================================================================================================
#tokenization for Ngram
    def tokeniForTrainNGRAM(self,path,leftRange,rightRange,label,nGram):

        try:
            for file in glob.glob(path)[leftRange:rightRange]:
                with open(file,'r') as f:
                    listWords=re.sub("[^\w']"," ",f.read()).split()
                    ngrams = zip(*[listWords[i:] for i in range(nGram)])
                    listWords=[" ".join(ngram) for ngram in ngrams]
                    for word in listWords:
                        self.weight[word]=0
                self.reviewLabel[str(listWords)]=label
                self.allTrainReviews.append(listWords)
        except FileNotFoundError:
            print('Fail to find file')

    def tokeniForTestNGRAM(self,path,leftRange,rightRange,label,nGram):
        try:
            for file in glob.glob(path)[leftRange:rightRange]:
                with open(file,'r') as f:
                    #listWords=toc.findall(f.read())
                    listWords=re.sub("[^\w']"," ",f.read()).split()
                    ngrams = zip(*[listWords[i:] for i in range(nGram)])
                    listWords=[" ".join(ngram) for ngram in ngrams]
                    self.reviewLabel[str(listWords)]=label
                self.allTestReviews.append(listWords)
        except FileNotFoundError:
            print('Fail to find file')


#==================================================================================================
#these function is for bigram and trigram
    def train(self):
        result=lambda s: -1 if s<0 else 1 #classify the score to -1 and 1
        for i in range(self.maxIter):  #iterate twenty times
            random.Random(i).shuffle(self.allTrainReviews) #shuffle the order every time
            for reviews in self.allTrainReviews:
                score=0
                Count=Counter(reviews) #get the count of each word in every review
                for word,counts in Count.items():
                    score+=self.weight[word]*counts
                if self.reviewLabel[str(reviews)]!=result(score):
                    self.error[i]+=1 #when prediction different from label, plus 1. In order to get the learning progress
                    if self.reviewLabel[str(reviews)]==1: #change the weight, when the label is 1, plus count
                        for word in Count:
                            self.weight[word]=self.weight[word]+Count[word]
                    else: #when the label is -1 subtract count
                        for word in Count:
                            self.weight[word]=self.weight[word]-Count[word]

            if self.error[i]==0: #when the model cannot learn from training set,break
                break

    def test(self):
        result=lambda s: -1 if s<0 else 1 #classify the score to -1 and 1
        precision=0
        for reviews in self.allTestReviews: #traverse the test review set
            score=0
            Count=Counter(reviews) #get the count of each word in every review
            for word in Count:
                if word in self.weight:
                    score+=Count[word]*self.weight[word]
            if result(score)==self.reviewLabel[str(reviews)]:
                precision+=1
        print 'acurracy for '+self.BOWtype+' is',precision/400.0


#==================================================================================================
#implement the text classification
    def perceptron(self):

        #unigram
        if(self.BOWtype=='unigram'):

            #tokenization as unigram
            self.tokeniForTrainUni(self.rootPos,0,800,1)
            self.tokeniForTrainUni(self.rootNeg,0,800,-1)
            self.tokeniForTestUni(self.rootPos,800,1000,1)
            self.tokeniForTestUni(self.rootNeg,800,1000,-1)


            self.trainLen=len(self.totalIndex)  #get the lenth of whole word in trainning set
            self.weight=np.zeros(self.trainLen) #initialise the weight vector with zero
            self.trainForUni()  #train for the weight
            self.testForUni()   #test the weight

        #bigram
        elif(self.BOWtype=='bigram'):
            self.tokeniForTrainNGRAM(self.rootPos,0,800,1,2)
            self.tokeniForTrainNGRAM(self.rootNeg,0,800,-1,2)
            self.tokeniForTestNGRAM(self.rootPos,800,1000,1,2)
            self.tokeniForTestNGRAM(self.rootNeg,800,1000,-1,2)

            self.train()
            self.test()

        #trigram
        else:
            self.tokeniForTrainNGRAM(self.rootPos,0,800,1,3)
            self.tokeniForTrainNGRAM(self.rootNeg,0,800,-1,3)
            self.tokeniForTestNGRAM(self.rootPos,800,1000,1,3)
            self.tokeniForTestNGRAM(self.rootNeg,800,1000,-1,3)

            self.train()
            self.test()



    #get the wrong predition number in trainning set
    def getError(self):
        return self.error

    def getTopFeature(self):
        top=sorted(self.weight.items(),key=lambda x:x[1])
        neg10=[item[0] for item in top[:10]]

        top=sorted(self.weight.items(),reverse=True,key=lambda x:x[1])
        pos10=[item[0] for item in top[:10]]

        print 'pos10 for '+self.BOWtype+' is',pos10
        print 'neg10 for '+self.BOWtype+' is',neg10
        print '\n'


#=========================================================================
#MAIN

if __name__=='__main__':

    timeStart=time.time() #get the start time

    config=CommandLine() #read from CommandLine
    rootPos,rootNeg=config.getPath()

    #classification for the unigram
    unigram=TextClassification(rootPos,rootNeg,'unigram')
    unigram.perceptron()
    unigram.getTopFeatureUni()
    errorUni=unigram.getError()

    #classification for the bigram
    bigram=TextClassification(rootPos,rootNeg,'bigram')
    bigram.perceptron()
    bigram.getTopFeature()
    errorBi=bigram.getError()

    #classification for trigram
    trigram=TextClassification(rootPos,rootNeg,'trigram')
    trigram.perceptron()
    trigram.getTopFeature()
    errorTri=trigram.getError()

    timeEnd=time.time()#get the end time
    print 'The total duration is',timeEnd-timeStart,'s'

    #plot the graph for learning progress

    plt.plot(errorUni, label="unigram")
    plt.plot(errorBi, label="bigram")
    plt.plot(errorTri, label="trigram")
    plt.legend(loc='upper right')
    plt.xlabel('iterate times')
    plt.ylabel('wrong prediction number in trainning set') #the wrong prediciton in trainning set
                                                     #can show the learning progress more directly
    plt.show()
