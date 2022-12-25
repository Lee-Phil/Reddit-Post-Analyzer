import gzip
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier


def getCol(data, index):
    column = []
    for i in range(len(data)):
        column.append(data[i][index])
    return column


def padMatrix(post, maxLength):
    difference = maxLength - len(post)
    if difference > 0:
        x.extend([0] * difference)


def MLP(postsTrain, postsTest, dataTrain, dataTest, file):
    classifier = MLPClassifier(verbose=True)
    dataModel = classifier.fit(postsTrain, dataTrain)

    # Getting the Hyper Parameters
    file.write("Hyper Parameters:" + str(classifier.get_params()) + "\n\n")
    # Using the data to get the confusion matrix
    confusionMatrixAndMetrics(dataTest, classifier.predict(postsTest), file)

    accuracyScore = classifier.score(postsTest, dataTest)
    print("The accuracy of the model is " + str(accuracyScore) + "\n")


def MLPwithGridEmotions(postsTrain, postsTest, dataTrain, dataTest, file):
    params = {'activation': ['tanh'],
              'hidden_layer_sizes': [(3, 30)],
              'solver': ['sgd']}
    classifier = GridSearchCV(MLPClassifier(verbose=True, early_stopping=True, max_iter=15), params)
    dataModel = classifier.fit(postsTrain, dataTrain)

    # Getting the Hyper Parameters
    file.write("Hyper Parameters:" + str(classifier.get_params()) + "\n\n")
    # Using the data to get the confusion matrix
    confusionMatrixAndMetrics(dataTest, classifier.predict(postsTest), file)

    accuracyScore = classifier.score(postsTest, dataTest)
    print("The accuracy of the model is " + str(accuracyScore) + "\n")


def MLPwithGridSentiments(postsTrain, postsTest, dataTrain, dataTest, file):
    params = {'activation': ['identity'],
              'hidden_layer_sizes': [(3, 30)],
              'solver': ['adam']}
    classifier = GridSearchCV(MLPClassifier(verbose=True, early_stopping=True, max_iter=15), params)
    dataModel = classifier.fit(postsTrain, dataTrain)

    # Getting the Hyper Parameters
    file.write("Hyper Parameters:" + str(classifier.get_params()) + "\n\n")
    # Using the data to get the confusion matrix
    confusionMatrixAndMetrics(dataTest, classifier.predict(postsTest), file)

    accuracyScore = classifier.score(postsTest, dataTest)
    print("The accuracy of the model is " + str(accuracyScore) + "\n")


def confusionMatrixAndMetrics(expected, predicted, file):
    cMatrix = confusion_matrix(expected, predicted)
    file.write("Confusion Matrix: \n" + str(cMatrix) + "\n\n")
    file.write("Classification Report \n" + classification_report(expected, predicted) + "\n\n")


def splitData(postMatrix, data):
    return train_test_split(postMatrix, data, test_size=0.2)


def embeddingsFailedWordCounter(sentences, model):
    failedWordCounter = 0
    averageEmbeddings = []
    averageEmbeddingsAllPosts = []
    totalTokens = 0

    for index, lists in enumerate(sentences):
        for x in lists:
            try:
                totalTokens += 1
                singularWordEmbedding = model[x]
                averageEmbeddings.append(singularWordEmbedding)
            except KeyError:
                failedWordCounter += 1
        if len(averageEmbeddings) != 0:
            average = np.average(averageEmbeddings, axis=0)
            averageEmbeddingsAllPosts.append(average)
            averageEmbeddings.clear()
        else:
            averageEmbeddingsAllPosts.append([0] * 300)

    return averageEmbeddingsAllPosts, failedWordCounter, totalTokens


file = gzip.open('goemotions.json.gz', 'rb')
data = json.loads(file.read())
posts = getCol(data, 0)
emotions = getCol(data, 1)
sentiments = getCol(data, 2)

# 3 Embeddings as Features
# 3.1
print('3.1 Starting')
import gensim.downloader as api

wv = api.load('word2vec-google-news-300')
print('word2vec loaded')
print('3.1 Ending')

# 3.2
print('3.2 Starting')
from nltk.tokenize import word_tokenize

counter = 0
tokensPerPost = []

for x in posts:
    tokens = word_tokenize(x)
    tokensPerPost.append(tokens)
    counter = counter + len(tokens)

postsTrainEmotions, postsTestEmotions, emotionsTrain, emotionsTest = splitData(tokensPerPost, data=emotions)
postsTrainSentiments, postsTestSentiments, sentimentsTrain, sentimentsTest = splitData(tokensPerPost,
                                                                                       data=sentiments)

postsTrainEmotions1, failedWordCounterPostsTrain1, totalTokensTrain1 = embeddingsFailedWordCounter(postsTrainEmotions,
                                                                                                   wv)
postsTestEmotions1, failedWordCounterPostsTest1, totalTokensTest1 = embeddingsFailedWordCounter(postsTestEmotions, wv)

postsTrainSentiments1, failedWordCounterPostsTrain1, totalTokensTrain1 = embeddingsFailedWordCounter(
    postsTrainSentiments, wv)
postsTestSentiments1, failedWordCounterPostsTest1, totalTokensTest1 = embeddingsFailedWordCounter(postsTestSentiments,
                                                                                                  wv)

print('The number of tokens in the entirety of the dataset is ', counter)
print('3.2 Ending')

# 3.3
print('3.3 Starting')
print('The total amount of failed words in the training set: ', failedWordCounterPostsTrain1)
print('The total amount of failed words in the test set: ', failedWordCounterPostsTest1)
print('The total amount of failed words is: ', (failedWordCounterPostsTrain1 + failedWordCounterPostsTest1))
print('3.3 Ending')

# 3.4
print('3.4 Starting')

hitRateTraining = ((totalTokensTrain1 - failedWordCounterPostsTrain1) / totalTokensTrain1)
hitRateTest = ((totalTokensTest1 - failedWordCounterPostsTest1) / totalTokensTest1)
hitRateTotal = (counter - (failedWordCounterPostsTrain1 + failedWordCounterPostsTest1)) / counter

print('The hit rate of the training set from the Reddit posts is: ',
      (totalTokensTrain1 - failedWordCounterPostsTrain1), ' / ', totalTokensTrain1, ' = ', hitRateTraining)
print('The hit rate of the test set from the Reddit posts is: ',
      (totalTokensTest1 - failedWordCounterPostsTest1), ' / ', totalTokensTest1, ' = ', hitRateTest)
print('The hit rate of both the training and test set from the Reddit posts is: ', hitRateTotal)

print('3.4 Ending')

# 3.5
print('3.5 Starting Base MLP')

performanceQ3File = open("performanceQ3.txt", "w")  # write mode

performanceQ3File.write("------------------ BASE-MLP Emotions Model ------------------" + "\n")
MLP(postsTrainEmotions1, postsTestEmotions1, emotionsTrain, emotionsTest, performanceQ3File)
performanceQ3File.write("------------------ BASE-MLP Sentiments Model ------------------" + "\n")
MLP(postsTrainSentiments1, postsTestSentiments1, sentimentsTrain, sentimentsTest, performanceQ3File)

print('Base-MLP has completed')
print('3.5 Ending')

# 3.6
print('3.6 Starting Top Model')

performanceQ3File.write("------------------ TOP-MLP Emotions Model ------------------" + "\n")
MLPwithGridEmotions(postsTrainEmotions1, postsTestEmotions1, emotionsTrain, emotionsTest, performanceQ3File)
performanceQ3File.write("------------------ TOP-MLP Sentiments Model ------------------" + "\n")
MLPwithGridSentiments(postsTrainSentiments1, postsTestSentiments1, sentimentsTrain, sentimentsTest, performanceQ3File)

print('Top-MLP has completed')
print('3.6 Ending')


# 3.7
print('3.7 Starting')
print('Performance file has been updated on performanceQ3.txt')
print('3.7 Ending')

# 3.8
print('3.8 Starting')
print('')
model2 = api.load('glove-wiki-gigaword-300')
postsTrainEmotions1, failed1, totalTokensTrain1 = embeddingsFailedWordCounter(postsTrainEmotions, model2)
postsTestEmotions1, failed2, totalTokensTest1 = embeddingsFailedWordCounter(postsTestEmotions, model2)
postsTrainSentiments1, failed1, totalTokensTrain1 = embeddingsFailedWordCounter(postsTrainSentiments, model2)
postsTestSentiments1, failed2, totalTokensTest1 = embeddingsFailedWordCounter(postsTestSentiments, model2)

performanceQ3File.write("-----------------Wikipedia 2014 + Gigaword 5 ---------------" + "\n")
performanceQ3File.write("------------------ TOP-MLP Emotions Model ------------------" + "\n")
MLPwithGridEmotions(postsTrainEmotions1, postsTestEmotions1, emotionsTrain, emotionsTest, performanceQ3File)
performanceQ3File.write("------------------ TOP-MLP Sentiments Model ------------------" + "\n")
MLPwithGridSentiments(postsTrainSentiments1, postsTestSentiments1, sentimentsTrain, sentimentsTest, performanceQ3File)

model3 = api.load('conceptnet-numberbatch-17-06-300')
postsTrainEmotions1, failed3, totalTokensTrain1 = embeddingsFailedWordCounter(postsTrainEmotions, model3)
postsTestEmotions1, failed4, totalTokensTest1 = embeddingsFailedWordCounter(postsTestEmotions, model3)
postsTrainSentiments1, failed3, totalTokensTrain1 = embeddingsFailedWordCounter(postsTrainSentiments, model3)
postsTestSentiments1, failed4, totalTokensTest1 = embeddingsFailedWordCounter(postsTestSentiments, model3)

performanceQ3File.write("-----ConceptNet, word2vec, GloVe, and OpenSubtitles 2016----" + "\n")
performanceQ3File.write("------------------ TOP-MLP Emotions Model ------------------" + "\n")
MLPwithGridEmotions(postsTrainEmotions1, postsTestEmotions1, emotionsTrain, emotionsTest, performanceQ3File)
performanceQ3File.write("------------------ TOP-MLP Sentiments Model ------------------" + "\n")
MLPwithGridSentiments(postsTrainSentiments1, postsTestSentiments1, sentimentsTrain, sentimentsTest, performanceQ3File)
print('For comparisons on both models, please view the Analysis document')
print('3.8 Ending')

performanceQ3File.close()
