from keras.activations import relu, softmax
from numpy import genfromtxt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Conv1D, Dropout, MaxPooling1D, GlobalAveragePooling1D
from scipy.signal import butter, lfilter
from sample import (butter_bandpass_filter)
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.signal import butter, lfilter, iirnotch
import matplotlib.pyplot as plt

# Import EEG Data from CSV
pathname = '../../Documents/BCI/pyOpenBCI-master/Examples/data/'
filename_01 = '2a.csv'
filename_02 = '2b.csv'
filename_03 = '2c.csv'
filename_04 = '2d.csv'
filename_05 = '2e.csv'
filename_11 = '1a.csv'
filename_12 = '1b.csv'
my_data_01 = genfromtxt(pathname + filename_01, delimiter=',')
my_data_02 = genfromtxt(pathname + filename_02, delimiter=',')
my_data_03 = genfromtxt(pathname + filename_02, delimiter=',')
my_data_04 = genfromtxt(pathname + filename_02, delimiter=',')
my_data_05 = genfromtxt(pathname + filename_02, delimiter=',')
my_data_11 = genfromtxt(pathname + filename_11, delimiter=',')
my_data_12 = genfromtxt(pathname + filename_12, delimiter=',')
my_data = np.vstack((my_data_01, my_data_02, my_data_03, my_data_04, my_data_05))
my_data_old = np.vstack((my_data_11, my_data_12))

# Remove Noise
nsamples = my_data[:, 1].shape[0]
T = nsamples/400
t = np.linspace(0, T, nsamples, endpoint=False)
fs = 400.0
lowcut = 2.0
highcut = 45.0
my_data[:, 2] = butter_bandpass_filter(my_data[:, 2], lowcut, highcut, fs, order=6)
my_data[:, 3] = butter_bandpass_filter(my_data[:, 3], lowcut, highcut, fs, order=6)
my_data[:, 4] = butter_bandpass_filter(my_data[:, 4], lowcut, highcut, fs, order=6)
my_data[:, 5] = butter_bandpass_filter(my_data[:, 5], lowcut, highcut, fs, order=6)


# Downsize Data
lineIndex = 0
shrinkData = np.zeros(6)
while lineIndex < my_data.shape[0]:
    if lineIndex % 3 == 0:
        currentLine = np.array(my_data[lineIndex])
        shrinkData = np.vstack((shrinkData, currentLine))
    lineIndex += 1
shrinkData = shrinkData[1:, :]
my_data = shrinkData

# Isolate Y
Y = my_data[:, 1]

# Separate words
lineIndex = 0
currentWord = 2
imageLength = 110
currentImage = np.zeros(4)
imageDimensions = (imageLength, 4)
imageDirectory = np.zeros(imageDimensions)
answerDirectory = np.zeros(1)

while lineIndex < my_data.shape[0]:
    currentLine = np.array(my_data[lineIndex])
    if int(currentLine[0]) == currentWord:
        currentImage = np.vstack((currentImage, currentLine[2:]))
    else:
        currentImageTrimmed = np.delete(currentImage, 0, 0)
        currentImageTrimmed = np.vsplit(currentImageTrimmed, ([imageLength]))[0]
        if currentImageTrimmed.shape[0] < imageLength:
            print("ERROR: Invalid Image at currentWord = " + str(currentWord))
            exit(1)
        imageDirectory = np.dstack((imageDirectory, currentImageTrimmed))
        answerDirectory = np.vstack((answerDirectory, currentLine[1]))
        print(str(imageDirectory.shape) + "\n")
        currentImage = np.zeros(4)
        currentWord = currentLine[0]
    lineIndex += 1

imageDirectory = np.transpose(imageDirectory, (2, 0, 1))
imageDirectory = np.delete(imageDirectory, 0, 0)
answerDirectory = np.delete(answerDirectory, 0, 0)
answerDirectory = np_utils.to_categorical(answerDirectory)
# imageDirectory = imageDirectory.reshape(imageDirectory.shape[0], imageDirectory.shape[1], imageDirectory.shape[2], 1)
print(imageDirectory.shape)
print(answerDirectory.shape)



# Split to Training and Testing Set
X_train, X_test, y_train, y_test = train_test_split(imageDirectory, answerDirectory, test_size=0.3)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Generate Noise Added Images
numImages, time, ch = X_train.shape
mean = 0
var = 0.12
sigma = var**0.5
gauss = np.random.normal(mean, sigma, (numImages, time, ch))
gauss = gauss.reshape(numImages, time, ch)
noisySet1 = X_train + gauss
noisyY1 = y_train
gauss = np.random.normal(mean, sigma, (numImages, time, ch))
noisySet2 = X_train + gauss
noisyY2 = y_train
gauss = np.random.normal(mean, sigma, (numImages, time, ch))
noisySet3 = X_train + gauss
noisyY3 = y_train

X_train = np.vstack((X_train, noisySet1, noisySet2, noisySet3))
y_train = np.vstack((y_train, noisyY1, noisyY2, noisyY3))








# Build Model
model = Sequential()
model.add(Conv1D(40, 10, strides=2, padding='same', activation='relu', input_shape=(imageLength, 4)))
model.add(Dropout(0.2))
model.add(MaxPooling1D(3))
# model.add(Conv1D(40, 5, strides=2, padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(MaxPooling1D(3))
# model.add(Conv1D(40, 4, strides=1, padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(MaxPooling1D(3))
model.add(GlobalAveragePooling1D())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from keras.utils import plot_model
plot_model(model, show_shapes=True, show_layer_names= True, to_file='model.svg')


# Train Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=100, epochs=500)

# CONSOLE TEST --------------------------------

# Test Model
y_predicted = model.predict(X_test)
classPredictions = np.zeros(y_predicted.shape[0])
classActual = np.zeros(y_test.shape[0])
index = 0
for prediction in y_predicted:
    if prediction[0] > prediction[1]:
        classPredictions[index] = 0
    else:
        classPredictions[index] = 1
    index += 1
index = 0
for answer in y_test:
    if answer[0] > answer[1]:
        classActual[index] = 0
    else:
        classActual[index] = 1
    index += 1

# print("\nPredicted: \n" + str(classPredictions))
# print("\nActual: \n" + str(classActual))
resultsChart = np.vstack((classActual, classPredictions)).transpose()






# Plot History
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("acc.png")
plt.clf()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("loss.png")
plt.clf()






# Calculate ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predicted[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_predicted.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC
plt.figure()
lw = 2
plt.plot(fpr[1], tpr[1], lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig("roc.png")
plt.clf()


# Plot PR
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(2):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_predicted[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_predicted[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
    y_predicted.ravel())
average_precision["micro"] = average_precision_score(y_test, y_predicted,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

# Make PR PLOT
plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))
plt.savefig("pr.png")




print("hello")
exit(0)


# Query
import webbrowser
qID = 9
baseString = "https://www.google.com/search?query="
queryString = ""
if classPredictions[qID] == 0:
    queryString = "directions+to+Starbucks+near+me"
elif classPredictions[qID] == 1:
    queryString = "directions+to+gas+station+near+me"
urlString = baseString + queryString
webbrowser.open(urlString, new=2)


# Print an image
X = imageDirectory[4, :, :]  # sample 2D array
print(X.shape)
plt.imshow(X, cmap="gray")
plt.savefig('demo.png', bbox_inches='tight')


