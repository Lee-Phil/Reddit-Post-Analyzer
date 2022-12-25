import gzip
import json

import matplotlib.pyplot as plt


def getCol(data, index):
    column = []
    for i in range(len(data)):
        column.append(data[i][index])
    return column

def countOccurrence(array, label):
    return array.count(label)

def generatePieChart(data, labels, filename):
    occurrence = []

    for label in labels:
        occurrence.append(countOccurrence(data, label))
    
    plt.pie(occurrence, labels = labels)
    plt.savefig(filename+".pdf")
    plt.clf()


# If flag is true if we want to generate the charts, false if we do not.
def q1(flag):      
    # 1.1, 1.2
    file = gzip.open('goemotions.json.gz', 'rb')
    data = json.loads(file.read())

    # 1.3
    posts = getCol(data, 0)

    emotions = getCol(data, 1)
    emotionLabels = [
        'admiration',
        'amusement',
        'anger',
        'annoyance',
        'approval',
        'caring',
        'confusion',
        'curiosity',
        'desire',
        'disappointment',
        'disapproval',
        'disgust',
        'embarrassment',
        'excitement',
        'fear',
        'gratitude',
        'grief',
        'joy',
        'love',
        'nervousness',
        'optimism',
        'pride',
        'realization',
        'relief',
        'remorse',
        'sadness',
        'surprise',
        'neutral',
    ]

    if flag:
        generatePieChart(emotions, emotionLabels, 'emotions-pie.png')
        print("Generated pie chart for emotions.")

    sentiments = getCol(data, 2)
    sentimentLabels = ["neutral", "positive", "negative", "ambiguous"]

    if flag:
        generatePieChart(sentiments, sentimentLabels, "sentiments-pie.png")
        print("Generated pie chart for sentiments.")
        
    return posts, emotions, sentiments

q1(True)
