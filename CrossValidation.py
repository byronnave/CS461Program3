from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(8, activation='sigmoid', kernel_initializer='random_uniform', input_dim=13))
    classifier.add(Dense(8, activation='sigmoid', kernel_initializer='random_uniform'))
    classifier.add(Dense(4, activation='softmax', kernel_initializer='random_uniform'))
    classifier.compile(optimizer ='SGD',loss='categorical_crossentropy', metrics =['accuracy'])
    return classifier

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=200)
accuracies =  cross_val_score(estimator=classifier, X= X, y=output_category,cv=10, n_jobs=-1)