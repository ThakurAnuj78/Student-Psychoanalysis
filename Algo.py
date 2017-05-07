import csv
import random
import math

from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
def getVal(res):
    str = ["Strongly disagree", "Disagree", "Neutral" , "Agree", "Strongly agree"]
    return str.index(res);
def getBoolVal(res):
    str1 = ["No","Yes"]
    return str1.index(res)
def loadCsv(filename):
    lines = csv.reader(open(filename, "rt"))
    dataset = list(lines)
    bl=["No","Yes"]
    for i in range(1,len(dataset)):
        for j in range(10,36):
            if (j<20):
                dataset[i][j] = getVal(dataset[i][j])
            elif((j>=22 and j<=31) or j==33):
                dataset[i][j] = getVal(dataset[i][j])
            elif(dataset[i][j] in bl):
                 dataset[i][j] = getBoolVal(dataset[i][j])
    for i in range(1,len(dataset)):
        for j in range(10,36):
            dataset[i][j] = int(dataset[i][j])
            dataset[i][7] = float(dataset[i][7])
            dataset[i][3] = float(dataset[i][3])
            dataset[i][4] = float(dataset[i][4])
    for i in range(1,len(dataset)):
        if (dataset[i][3]>10.0):
            dataset[i][3]=dataset[i][3]/10.0
        if (dataset[i][4]>10.0):
            dataset[i][4]=dataset[i][4]/10.0
        if (dataset[i][7]>10.0):
            dataset[i][7]=dataset[i][7]/10.0
        if(dataset[i][7]>=8.0):
            dataset[i].append(2)
        elif(dataset[i][7]<8.0 and dataset[i][7]>=6.5):
            dataset[i].append(1)
        else:
            dataset[i].append(0)

                
    '''print(dataset)'''
    
    return editSet(dataset)

def editSet(dataset):
    dataset1 = [[]]
    dataset1.pop(0)
    
    for i in range (1,len(dataset)):
        csv1 = []
        csv1.append((dataset[i][10]+dataset[i][25])/2.0)
        csv1.append((dataset[i][12]+dataset[i][14]+dataset[i][16]+dataset[i][17]+dataset[i][18]+dataset[i][19])/6.0)
        csv1.append(((dataset[i][3]+dataset[i][4])/2*0.3)+((dataset[i][7])*0.7))
        csv1.append((dataset[i][11]+dataset[i][26]+dataset[i][27]+dataset[i][28])/4.0)
        csv1.append((dataset[i][22]+dataset[i][23]+dataset[i][24]+dataset[i][29])/4.0)
        csv1.append((dataset[i][13]+dataset[i][28]+dataset[i][31]+dataset[i][32])/4.0)
        csv1.append((dataset[i][33]+dataset[i][35]+dataset[i][34])/3.0)
        csv1.append((dataset[i][20]+dataset[i][31]+dataset[i][15])/3.0)
        csv1.append((dataset[i][36]))
        dataset1.append(csv1)
    '''print(csv1)'''
    #print(dataset1)
    return dataset1


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]
 
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
    
        if (vector[-1] not in separated):    
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)        
    return separated
 
def mean(numbers):
    return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)
 
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries
 
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries
 
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]            
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities
            
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel
 
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(1,len(testSet)):
        
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions
 
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(1,len(testSet)):
        if testSet[i][-1] == predictions[i-1]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
def checkPredict(summ,tr):
    
    '''Kmeans'''
    
    trcheck=[]
    #for j in range(len(tr)):
     #   trcheck[j]=tr[j][-1]
    clf = KMeans(n_clusters=3)
    clf.fit(tr)
    crr=0
    trcheck2=clf.predict(tr)
    #print(trcheck2)
    for k in range(len(tr)):
        if(trcheck2[k]==tr[k][-1]):
            crr=crr+1
        #print(tr[k][-1])
    accur=(crr/len(tr))*100
    print("Accuracy by KMeans:",accur,"%")
    
    '''Entry Prediction'''
    
    line = csv.reader(open("Res.csv", "rt"))
    res=[[]]            
    res = list(line)
    for i in range(28):
        res[0][i]=float(res[0][i])
    
    dataset2=[[]]
    #for i in range (len(res)):
    csv2 = list()
    csv2.append((res[0][1]+res[0][16])/2.0)
    csv2.append((res[0][3]+res[0][5]+res[0][7]+res[0][8]+res[0][9]+res[0][10])/6.0)
    csv2.append(((res[0][26]+res[0][27])/2*0.3)+((7.0*0.7)))
    csv2.append((res[0][2]+res[0][17]+res[0][18]+res[0][19])/4.0)
    csv2.append((res[0][13]+res[0][14]+res[0][15]+res[0][20])/4.0)
    csv2.append((res[0][4]+res[0][19]+res[0][22]+res[0][23])/4.0)
    csv2.append((res[0][24]+res[0][26]+res[0][25])/3.0)
    csv2.append((res[0][11]+res[0][22]+res[0][6])/3.0)
    print("Entry details ",csv2)
    predictions2=predict(summ,csv2)
    print("Entry prediction class",predictions2)
    
            
    

def main():
    filename = 'student psychology.csv'
    splitRatio = 0.67
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    # prepare model
    
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy by Gaussian Naive Bayes: {0}%'.format(accuracy))
    checkPredict(summaries,trainingSet)
    
 
main()
