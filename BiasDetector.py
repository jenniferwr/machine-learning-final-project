# Bias detection in nyc jobs dataset
import csv
import nltk, string
import numpy
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import itertools
import matplotlib.pyplot as plt 


warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec

def loadData():
    with open('nyc-jobs.csv', 'rt', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)

    # 2d list to contain all the information about a job
    # jobInfo[1] will contain all info about job at index 1
    # 'Business Title', 'Level', 'Job Category', 'Full-Time/Part-Time indicator', 'Salary Range From', 'Salary Range To', 'Salary Frequency'
    w, h = 7, len(data);
    jobInfo = [[0 for x in range(w)] for y in range(h)]

    # 2d list, stores 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills', 'Additional Information'
    # jobDesc[1] will contain info about job at index 1
    w, h = 4, len(data);
    jobDesc = [[0 for x in range(w)] for y in range(h)]

    # loop to put the data from the larger dataset into our lists
    for x in range(1, len(data)):
        jobInfo[x][0] = data[x][4]
        jobInfo[x][1] = data[x][7]
        jobInfo[x][2] = data[x][8]
        jobInfo[x][3] = data[x][9]
        jobInfo[x][4] = data[x][10]
        jobInfo[x][5] = data[x][11]
        jobInfo[x][6] = data[x][12]


        jobDesc[x][0] = data[x][16]
        jobDesc[x][1] = data[x][17]
        jobDesc[x][2] = data[x][18]
        jobDesc[x][3] = data[x][19]

    return jobInfo, jobDesc

def getBiasScorePerJobDescription(jobDesc):
    # female words
    femaleWords = ["Amy" , "Joan" , "Lisa" , "Sarah" , "Diana" , "Kate" , "Ann" , "Donna", "sister" , "female" , "woman" , "girl" , "daughter" , "she" , "hers" , "her"]
    maleWords = ["brother" , "male" , "man" , "boy" , "son" , "he" , "his" , "him", "John" , "Paul" , "Mike" , "Kevin" , "Steve" , "Greg" , "Jeff" , "Bill"]
    femaleCosineSim = []
    maleCosineSim = []

    # go through job descriptions
    for desc0, desc1, desc2, desc3 in jobDesc[1:500,:]:
        f = desc0.replace("\n", " ")
        f+=desc1.replace("\n", " ")
        f+=desc2.replace("\n", " ")
        f+=desc3.replace("\n", " ")

        data = []
        # iterate through each sentence in the file
        for i in sent_tokenize(f):
            temp = []
            # tokenize the sentence into words
            for j in word_tokenize(i):
                temp.append(j.lower())
            data.append(temp)

        # get cosine similarity per word
        cosineSimilarityFemale = []
        cosineSimilarityMale = []

        # append female and male words
        data.append(femaleWords)
        data.append(maleWords)

        # arr = numpy.array(vocab)
        model1 = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5)

        for word in data:
            for i in range(len(femaleWords)):
                cosineSimilarityFemale.append(model1.similarity(word, femaleWords[i]))
                cosineSimilarityMale.append(model1.similarity(word, maleWords[i]))

        # flatmap cosine values
        cosineSimFem =list(itertools.chain(*cosineSimilarityFemale))

        cosineSimMal =list(itertools.chain(*cosineSimilarityMale))

        # average
        femaleAvg = sum(cosineSimFem)/len(cosineSimFem)
        maleAvg = sum(cosineSimMal)/len(cosineSimMal)

        # add to an array that cooresponds to to the same index
        femaleCosineSim.append(femaleAvg)
        maleCosineSim.append(maleAvg)
       # print(".")


    # return an array of the bias scores
    return femaleCosineSim, maleCosineSim

def biasDistribution(jobs,femaleBias,maleBias):
    #x-axis values 
    x = jobs
    # y-axis values 
    y1 = femaleBias
    y2 = maleBias
  
    # plotting points as a scatter plot 
    plt.scatter(x, y1, label= "femaleBias", color= "red",  
            marker= "*", s=30) 
    plt.ylim(-0.002,0.004)
    plt.scatter(x, y2, label= "maleBias", color= "green",  
            marker= "*", s=30) 
    plt.ylim(-0.002,0.004)
  
    # x-axis label 
    plt.xlabel('Jobs') 
    # frequency label 
    plt.ylabel('Bias') 
    # plot title 
    plt.title('Distribution of Bias') 
    # showing legend 
    plt.legend() 
  
    # function to show the plot 
    plt.show() 
    return

def biasAndSalaryFreq(jobInfo2, femaleBias, maleBias):
    x1 = numpy.array(femaleBias)
    x2 = numpy.array(maleBias)

    unique, rev = numpy.unique(jobInfo2[1:500,6], return_inverse=True)
    #print (unique)
    #print (rev)
    #print (unique[rev])

    fig,ax=plt.subplots()
    ax.scatter(x1, rev, label= "femaleBias", color= "red", marker= '*')
    plt.xlim(-0.001,0.004)
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels(unique)
    plt.plot()
     # x-axis label 
    plt.xlabel('Female Bias') 
    # frequency label 
    plt.ylabel('Salary Frequency') 
    # plot title 
    plt.title('Distribution of Female Bias with reference to Salary Frequency') 

    
    fig,ax2=plt.subplots()
    ax2.scatter(x2, rev, label= "maleBias", color= "green", marker= '*')
    plt.xlim(-0.001,0.004)
    ax2.set_yticks(range(len(unique)))
    ax2.set_yticklabels(unique)
    plt.plot()
    # x-axis label 
    plt.xlabel('Male Bias') 
    # frequency label 
    plt.ylabel('Salary Frequency') 
    # plot title 
    plt.title('Distribution of Male Bias with reference to Salary Frequency') 
    plt.show()
    return 

#def biasAndsalary():
def biasAndFP(jobInfo2, femaleBias, maleBias):
    x1 = numpy.array(femaleBias)
    x2 = numpy.array(maleBias)

    unique, rev = numpy.unique(jobInfo2[1:500,3], return_inverse=True)
    
    fig,ax=plt.subplots()
    ax.scatter(x1, rev, label= "femaleBias", color= "red", marker= '*')
    plt.xlim(-0.001,0.004)
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels(unique)
    plt.plot()
     # x-axis label 
    plt.xlabel('Female Bias') 
    # frequency label 
    plt.ylabel('Full-Time/Part-Time') 
    # plot title 
    plt.title('Distribution of Female Bias with reference toFull/Part -Time indicator') 

    
    fig,ax2=plt.subplots()
    ax2.scatter(x2, rev, label= "maleBias", color= "green", marker= '*')
    plt.xlim(-0.001,0.004)
    ax2.set_yticks(range(len(unique)))
    ax2.set_yticklabels(unique)
    plt.plot()
    # x-axis label 
    plt.xlabel('Male Bias') 
    # frequency label 
    plt.ylabel('Full-Time/Part-Time') 
    # plot title 
    plt.title('Distribution of Male Bias with reference to Full/Part -Time indicator') 
    plt.show()
    return 
    
def biasAndJobCat(jobInfo2, femaleBias, maleBias):
    x1 = numpy.array(femaleBias)
    x2 = numpy.array(maleBias)

    unique, rev = numpy.unique(jobInfo2[1:500,2], return_inverse=True)
    
    fig,ax=plt.subplots()
    ax.scatter(x1, rev, label= "femaleBias", color= "red", marker= '*')
    plt.xlim(-0.001,0.004)
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels(unique)
    plt.plot()
     # x-axis label 
    plt.xlabel('Female Bias') 
    # frequency label 
    plt.ylabel('Job Category') 
    # plot title 
    plt.title('Distribution of Female Bias with reference to Job Category') 

    fig,ax2=plt.subplots()
    ax2.scatter(x2, rev, label= "maleBias", color= "green", marker= '*')
    #plt.xlim(-0.001,0.004)
    ax2.set_yticks(range(len(unique)))
    ax2.set_yticklabels(unique)
    plt.plot()
    # x-axis label 
    plt.xlabel('Male Bias') 
    # frequency label 
    plt.ylabel('Job Category') 
    # plot title 
    plt.title('Distribution of Male Bias with reference to Job Category') 
    plt.show()
    return


def main():
    jobInfo, jobDesc = loadData()
    jobDescArr = numpy.array(jobDesc)
    femaleBias, maleBias = getBiasScorePerJobDescription(jobDescArr)
    h=499
    jobs = [ 0 for x in range(h)]
    for i in range(len(jobs)):
        jobs[i] = i
    biasDistribution(jobs,femaleBias,maleBias)
    jobInfo2 = numpy.array(jobInfo)  
    biasAndSalaryFreq(jobInfo2, femaleBias, maleBias)
    biasAndFP(jobInfo2, femaleBias, maleBias)
    salaryFrom =jobInfo2[1:500,4]
    salaryTo = jobInfo2[1:500,5]
    biasAndJobCat(jobInfo2, femaleBias, maleBias)

    '''avgSalary= []
    for i in range(len(jobInfo2)):
        avgSalary[i] = (salaryFrom[i] + salaryTo[i])/2'''
    

    
    



    
    
main()
