import nltk, csv, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import feature_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

tableHeader = ['', 'urlDrugName', 'rating', 'effectiveness', 'sideEffects', 'condition', 'benefitsReview',
               'sideEffectsReview', 'commentsReview']
siteEffect = ['No ', 'Mild ', 'Moderate', 'Severe', 'Extremely Severe']
def loaddata():
    test_raw = []
    train_raw = []
    '''Read data from .tsv file'''
    with open("drugLib_raw/drugLibTest_raw.tsv") as drugTest:
        reader = csv.DictReader(drugTest, dialect='excel-tab')
        for row in reader:
            test_raw.append(row)

    with open("drugLib_raw/drugLibTrain_raw.tsv") as drugTrain:
        for row in csv.DictReader(drugTrain, dialect='excel-tab'):
            train_raw.append(row)

    print(len(test_raw))  # 1036
    print(len(train_raw))  # 3107

    '''preprocessing text function: lower case remove numbers stopwords, lemmatization POS'''


    def preprocess_text(text):
        # Convert text to lowercase and remove multiple spaces, number
        text = re.sub(r'[^A-Za-z]+', ' ', text.lower()).strip()

        # remove stopwords
        document = [word for word in nltk.word_tokenize(text) if word not in nltk.corpus.stopwords.words('english')]

        # Lemmatization
        document = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in document]

        # POS remove noun and verb
        # document = [word for (word, pos) in nltk.pos_tag(document) if pos != 'NN' and pos != 'VB']

        text = ' '.join(document)
        return text


    '''Pre-processing the data'''
    testSet = []
    trainSet = []
    for data in test_raw:
        testSet.append(
            [preprocess_text(data['benefitsReview'] + ' ' + data['sideEffectsReview'] + ' ' + data['commentsReview']),
             (data['sideEffects'])])

    for data in train_raw:
        trainSet.append(
            [preprocess_text(data['benefitsReview'] + ' ' + data['sideEffectsReview'] + ' ' + data['commentsReview']),
             (data['sideEffects'])])

    '''store and write data to csv file so when run multiple classifier, does not have to wait for preprocess'''
    dtf_test = pd.DataFrame(testSet)
    dtf_train = pd.DataFrame(trainSet)
    dtf_test.to_csv('test.csv')
    dtf_train.to_csv('train.csv')


'''read preprocessed data'''
testSet = pd.read_csv('test.csv', index_col=0)
trainSet = pd.read_csv('train.csv', index_col=0)
# combine test and train set
dataSet = pd.concat([trainSet, testSet])

'''print the bar graph'''
a = [0, 0, 0, 0, 0]
for data in testSet['1']:
    if data == 'No Side Effects':
        a[0] = a[0] + 1
    elif data == 'Mild Side Effects':
        a[1] = a[1] + 1
    elif data == 'Moderate Side Effects':
        a[2] = a[2] + 1
    elif data == 'Severe Side Effects':
        a[3] = a[3] + 1
    elif data == 'Extremely Severe Side Effects':
        a[4] = a[4] + 1

print(a)

plt.bar(siteEffect, a, color='green')
plt.show()

'''machine learning'''
y_test = testSet['1']
y_train = trainSet['1']

'''function to run machine learning algorithm'''


def classifier(classifier, x_train, x_test):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))  # print matrix
    print(classification_report(y_test, y_pred))  # print recall precision and f-score
    print('accuracy: ', accuracy_score(y_test, y_pred))  # print accuracy

def produceResult():
    print()
    '''covert word into vector using BOW model
       also try to capture unigrams and bigrams(2 word)
       and ignore some of high and low frequency '''
    vectorizer = CountVectorizer(min_df=5, max_df=0.7, ngram_range=(1, 2))
    X = vectorizer.fit_transform(dataSet['0'])
    x_train = vectorizer.transform(trainSet['0'])
    x_test = vectorizer.transform(testSet['0'])

    print('Convert the word to a vector using BOW model')
    print('KNeighborsClassifier')
    classifier(KNeighborsClassifier(4), x_train, x_test)
    print('----------------------------------------------------------------------------')
    print('MLPClassifier')
    classifier(MLPClassifier(alpha=1, max_iter=1000), x_train, x_test)  # 1000 iterations
    print('----------------------------------------------------------------------------')

    '''Using TF-IDF instead of BOW'''
    tfidfVectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidfVectorizer.fit(dataSet['0'])
    x_train = tfidfVectorizer.transform(trainSet['0'])
    x_test = tfidfVectorizer.transform(testSet['0'])

    print('Convert the word to a vector using TF-IDF')
    print('KNeighborsClassifier')
    classifier(KNeighborsClassifier(4), x_train, x_test)
    print('----------------------------------------------------------------------------')
    print('MLPClassifier')
    classifier(MLPClassifier(alpha=1, max_iter=1000), x_train, x_test)
    print('----------------------------------------------------------------------------')

    print('Feature Selection to select some word')
    y = dataSet['1']
    X_names = vectorizer.get_feature_names()
    p_value_limit = 0.95
    dtf_features = pd.DataFrame()
    for cat in np.unique(y):
        chi2, p = feature_selection.chi2(X, y == cat)
        dtf_features = dtf_features.append(pd.DataFrame({"feature": X_names, "score": 1 - p, "y": cat}))
        dtf_features = dtf_features.sort_values(["y", "score"],ascending=[True, False])
        dtf_features = dtf_features[dtf_features["score"] > p_value_limit]
    X_names = dtf_features["feature"].unique().tolist()

    for cat in np.unique(y):
       print("# {}:".format(cat))
       print("  . selected features:",
             len(dtf_features[dtf_features["y"]==cat]))
       print("  . top features:", ",".join(
    dtf_features[dtf_features["y"]==cat]["feature"].values[:10]))
       print(" ")

    tfidfVectorizer = TfidfVectorizer(vocabulary=X_names)
    tfidfVectorizer.fit(dataSet['0'])
    x_train = tfidfVectorizer.transform(trainSet['0'])
    x_test = tfidfVectorizer.transform(testSet['0'])

    print('Convert the word to a vector using Feature Selection')
    print('KNeighborsClassifier')
    classifier(KNeighborsClassifier(4), x_train, x_test)
    print('----------------------------------------------------------------------------')
    print('MLPClassifier')
    classifier(MLPClassifier(alpha=1, max_iter=1000), x_train, x_test)
    print('----------------------------------------------------------------------------')

print("END")
