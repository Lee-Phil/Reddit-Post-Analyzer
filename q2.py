from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import q1
# 2.1
def q2_1(posts):
    vectorizer = CountVectorizer()
    postMatrix = vectorizer.fit_transform(posts)

    # Getting the tokens/words
    extractedWords = vectorizer.get_feature_names_out()

    # Display the number of tokens (the size of the vocabulary) in the dataset.
    print("There are " + str(len(extractedWords)) + " tokens in this dataset")
    print("")
    return postMatrix


# 2.2
# Review this
from sklearn.model_selection import train_test_split


def splitData(postMatrix, data):
    return train_test_split(postMatrix, data, test_size=0.2)


# 2.3
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


#Creating the confusion Matrix
def confusionMatrixAndMetrics(expected,predicted,file):
    cMatrix = confusion_matrix(expected, predicted)
    file.write("Confusion Matrix: \n" + str(cMatrix)+"\n\n")
    file.write("Classification Report \n"+ classification_report(expected, predicted)+"\n\n")

def MNBclassifier(postsTrain, postsTest, dataTrain, dataTest, file):
    classifier = MultinomialNB()
    dataModel = classifier.fit(postsTrain, dataTrain)

    #Getting the Hyper Parameters
    file.write("Hyper Parameters:" + str(classifier.get_params()) + "\n\n")
    #Using the data to get the confusion matrix
    confusionMatrixAndMetrics(dataTest,classifier.predict(postsTest),file)

    accuracyScore = classifier.score(postsTest, dataTest)
    print("The accuracy of the model is " + str(accuracyScore) + "\n")

from sklearn import tree


def DecisionClassifier(postsTrain, postsTest, dataTrain, dataTest, file):
    classifier = tree.DecisionTreeClassifier()
    dataModel = classifier.fit(postsTrain, dataTrain)

    #Getting the Hyper Parameters
    file.write("Hyper Parameters:" + str(classifier.get_params()) + "\n\n")
    #Using the data to get the confusion matrix
    confusionMatrixAndMetrics(dataTest,classifier.predict(postsTest),file)

    accuracyScore = classifier.score(postsTest, dataTest)
    print("The accuracy of the model is " + str(accuracyScore) + "\n")


from sklearn.neural_network import MLPClassifier


def MLP(postsTrain, postsTest, dataTrain, dataTest, file):
    classifier = MLPClassifier(verbose=True,early_stopping=True)
    dataModel = classifier.fit(postsTrain, dataTrain)

    #Getting the Hyper Parameters
    file.write("Hyper Parameters:" + str(classifier.get_params()) + "\n\n")
    #Using the data to get the confusion matrix
    confusionMatrixAndMetrics(dataTest,classifier.predict(postsTest),file)

    accuracyScore = classifier.score(postsTest, dataTest)
    print("The accuracy of the model is " + str(accuracyScore) + "\n")


from sklearn.model_selection import GridSearchCV


def MNBclassifierWithGrid(postsTrain, postsTest, dataTrain, dataTest,file):
    # the return of the bayes
    params = {'alpha': [0.5, 0, 2]}
    classifier = GridSearchCV(MultinomialNB(), params)
    dataModel = classifier.fit(postsTrain, dataTrain)

    #Getting the Hyper Parameters
    file.write("Hyper Parameters:" + str(classifier.get_params()) + "\n\n")
    #Using the data to get the confusion matrix
    confusionMatrixAndMetrics(dataTest,classifier.predict(postsTest),file)

    accuracyScore = dataModel.score(postsTest, dataTest)
    print("The accuracy of the model is " + str(accuracyScore) + "\n")


def DecisionTreeClassifierWithGrid(postsTrain, postsTest, dataTrain, dataTest,file):
    params = {'criterion': ('gini', 'entropy'),
              'max_depth': [30, 40],
              'min_samples_split': [3, 4, 5]}
    classifier = GridSearchCV(tree.DecisionTreeClassifier(), params)
    dataModel = classifier.fit(postsTrain, dataTrain)

    #Getting the Hyper Parameters
    file.write("Hyper Parameters:" + str(classifier.get_params()) + "\n\n")
    #Using the data to get the confusion matrix
    confusionMatrixAndMetrics(dataTest,classifier.predict(postsTest),file)

    accuracyScore = classifier.score(postsTest, dataTest)
    print("The accuracy of the model is " + str(accuracyScore) + "\n")


def MLPwithGrid(postsTrain, postsTest, dataTrain, dataTest,file):
    params = {'activation': ('sigmoid', 'tanh', 'relu', 'identity'),
              'hidden_layer_sizes': [(2, 30), (2, 50), (3, 10), (3, 20), (3, 30)],
              'solver': ['sgd', 'adam']}
    classifier = GridSearchCV(MLPClassifier(verbose=True, early_stopping=True, max_iter=15), params)
    dataModel = classifier.fit(postsTrain, dataTrain)

    #Getting the Hyper Parameters
    file.write("Hyper Parameters:" + str(classifier.get_params()) + "\n\n")
    #Using the data to get the confusion matrix
    confusionMatrixAndMetrics(dataTest,classifier.predict(postsTest),file)

    accuracyScore = classifier.score(postsTest, dataTest)
    print("The accuracy of the model is " + str(accuracyScore) + "\n")


def main():
    #Opening file to write
    performanceFile = open("performance.txt", "w")  # write mode

    posts, emotions, sentiments = q1.q1(False)

    # 2.1
    postMatrix = q2_1(posts)

    # 2.2
    postsTrainEmotions, postsTestEmotions, emotionsTrain, emotionsTest = splitData(postMatrix, data=emotions)
    postsTrainSentiments, postsTestSentiments, sentimentsTrain, sentimentsTest = splitData(postMatrix, data=sentiments)

    # 2.3.1
    performanceFile.write("---------------------- BASE-MNB Emotions Model ----------------------"+"\n")
    MNBclassifier(postsTrainEmotions, postsTestEmotions, emotionsTrain, emotionsTest,performanceFile)

    performanceFile.write("---------------------- BASE-MNB Sentiments Model ----------------------"+"\n")
    MNBclassifier(postsTrainSentiments, postsTestSentiments, sentimentsTrain, sentimentsTest,performanceFile)

    # 2.3.2
    performanceFile.write("------------------ BASE-Emotions Decision Tree Model ------------------" + "\n")
    print("Emotions Decision Tree Model")
    DecisionClassifier(postsTrainEmotions, postsTestEmotions, emotionsTrain, emotionsTest, performanceFile)

    performanceFile.write("------------------ BASE-Sentiment Decision Tree Model ------------------" + "\n")
    print("Sentiment Decision Tree Model")
    DecisionClassifier(postsTrainSentiments, postsTestSentiments, sentimentsTrain, sentimentsTest, performanceFile)

    # 2.3.3
    performanceFile.write("------------------ BASE-MLP Sentiments Model ------------------" + "\n")
    print("MLP Sentiments Model")
    MLP(postsTrainSentiments, postsTestSentiments, sentimentsTrain, sentimentsTest, performanceFile)
    performanceFile.write("------------------ BASE-MLP Emotions Model ------------------" + "\n")
    print("MLP Emotions Model:")
    MLP(postsTrainEmotions, postsTestEmotions, emotionsTrain, emotionsTest, performanceFile)

    # 2.3.4
    performanceFile.write("------------------ MNB Emotions Model with GridSearch ------------------" + "\n")
    print("MNB Emotions Model with GridSearch:")
    MNBclassifierWithGrid(postsTrainEmotions, postsTestEmotions, emotionsTrain, emotionsTest, performanceFile)

    performanceFile.write("------------------ MNB Sentiments Model with GridSearch ------------------" + "\n")
    print("MNB Sentiments Model with GridSearch")
    MNBclassifierWithGrid(postsTrainSentiments, postsTestSentiments, sentimentsTrain, sentimentsTest, performanceFile)

    # 2.3.5
    performanceFile.write("--------------- Emotions Decision Tree Model with GridSearch ---------------" + "\n")
    print("Emotions Decision Tree Model with GridSearch:")
    DecisionTreeClassifierWithGrid(postsTrainEmotions, postsTestEmotions, emotionsTrain, emotionsTest, performanceFile)

    performanceFile.write("-------------- Sentiments Decision Tree Model with GridSearch --------------" + "\n")
    print("Sentiments Decision Tree Model with GridSearch")
    DecisionTreeClassifierWithGrid(postsTrainSentiments, postsTestSentiments, sentimentsTrain, sentimentsTest, performanceFile)

    # 2.3.6
    performanceFile.write("------------------ MLP Emotions Model with GridSearch ------------------" + "\n")
    print("MLP Emotions Model with GridSearch:")
    MLPwithGrid(postsTrainEmotions, postsTestEmotions, emotionsTrain, emotionsTest, performanceFile)

    performanceFile.write("------------------ MLP Sentiments Model with GridSearch ------------------" + "\n")
    print("MLP Sentiments Model with GridSearch")
    MLPwithGrid(postsTrainSentiments, postsTestSentiments, sentimentsTrain, sentimentsTest, performanceFile)
    performanceFile.close()

main()
