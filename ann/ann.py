""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
ann.py
Sheet A03, exercise 2 "Machine Learning"
CO-Physics 1 2021WS
By Clemens Wager, 01635477
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This script can be run by using Python: python3.7

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.neural_network import MLPRegressor as MLNN

#----------------------------------------------------------------------------------------------#
### Function for plotting ###
def plotter(X, Y, title: str):
    """ Plot coordinate data """
    # print("Plot of the dataset we're working with:")
    plt.figure(figsize=(8, 4))
    plt.plot(X, Y, '.')
    if "vs." not in title:
        plt.xlabel("Input values")
        plt.ylabel("Output values")
    # print("X:",X.shape)
    # print("Y:",Y.shape)
    plt.title(title)
    plt.show()


def plotFile(file: str, title: str):
    """ Plot X,Y data from output file """
    data = pd.read_csv(file, sep=' ', header=None)
    data = pd.DataFrame(data)
    x = data[0]
    y = data[1]
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, '+r')
    plt.xlabel("Input values")
    plt.ylabel("Output values")
    plt.title(title)
    plt.show()


def plotFileValidation(file: str, validx, validy, title: str):
    """ Plot X,Y data from output file """
    data = pd.read_csv(file, sep=' ', header=None)
    data = pd.DataFrame(data)
    xfile = data[0]
    yfile = data[1]
    plt.figure(figsize=(8, 4))
    plt.plot(xfile, yfile, '+r')
    plt.plot(validx, validy, marker='.')
    plt.xlabel("Input values")
    plt.ylabel("Output values")
    plt.title(title)
    plt.show()


def allPlotter(plotList):
    """ Plot all datasets of final result"""
    plt.figure(figsize=(8, 4))
    m = 0
    markers = ['.', '+', '+', '+']
    colors =  ['tab:blue', 'red', 'green', 'orange']
    labels = ["Validation data",
              "Prediction model",
              "Test 1",
              "Test 2"]
    for data in plotList:
        data = pd.DataFrame(data)
        x = data[0]
        y = data[1]
        plt.plot(x, y,
                 marker=markers[m],
                 color=colors[m],
                 label=labels[m])
        m += 1
    plt.xlabel("Input values")
    plt.ylabel("Output values")
    plt.title("Final results")
    plt.legend()
    plt.show()
#----------------------------------------------------------------------------------------------#

#################################
### Enable all test
testOne = True  # include test 1
testTwo = True  # include test 2

#################################
### Data Collection & Preparation

allPlots = []  # collect computed datasets for final resul plot
# load the data from the file 'data-train.dat'
# into trainx and trainy (input and output data, respectively).
train = np.loadtxt('data/data-train.dat')
# X: (715, 1)
# Y: (715,)
trainx = train[:, 0].reshape(-1, 1)
trainy = train[:, 1]
print("--Training data")  # DEBUG
#plotter(trainx, trainy, "Training data")  # DEBUG plot train data

### load the data from the file 'data-validation.dat'
### into validx and validy
valid = np.loadtxt('data/data-validation.dat')
# X: (209, 1)
# Y: (209,)
validx = valid[:, 0].reshape(-1, 1)
validy = valid[:, 1]
print("--Validation data")  # DEBUG
validation_data = np.column_stack((validx, validy))  # stack output data
allPlots.append(validation_data)  # add validation dataset to list for plotting
#plotter(validx, validy, "Validation data")  # DEBUG plot validation data

####################################
### Define the ANN hyper-parameters.
start_time = time.time()  # measure time of function

# Define the ANN hyper-parameters: the ANN architecture.
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
model = MLNN(random_state=42,  # set fix random state to make results reproducible
             hidden_layer_sizes=(60, 40, 40, 40),  # deeper is better # 50, 40, 30, 20, 10, 5
             activation='relu',  # Activation function (‘tanh’, ‘relu’)
             solver='adam',  # weight optimization (adam, sgd, lbfgs)
             tol=1e-6,  # tolerance
             alpha=1e-5,  # L2 penalty (regularization term)
             learning_rate_init=1e-3,  # on Adam or SGD
             max_iter=500)  # max. number of epochs
"""
input: 60
hidden: 40, 40
output: 40
activation: relu
solver: adam
tol=1e-6
alpha=1e-5
learning_rate: 1e-3
--------------------
MAE:	 0.3229
MSE:	 0.1641
Score:	 0.5854
Epochs:  232
training+validation in 0.893 s

Data is just super noisy. Higher score not possible.
"""

#######################
### Training the ANN with trainx and trainy
model.fit(trainx, trainy)

#######################
### Validation phase
# Apply the just-trained ANN model to the validation data.

# Plot the results from the 'results.dat' file.
# Compare graphically the results calculated during the validation phase,
# with the expected target values in "validy" ('data/data-validation.dat').
# Adjust the ANN hyper-parameters in order to obtain satisfying accuracy on the validation data set.

Y_pred_val = model.predict(validx)  # Y_pred_val is the prediction of Y values in the validation phase

stop_time = time.time()  # measure time of function

validation_results = np.column_stack((validx, Y_pred_val))  # stack output data
allPlots.append(validation_results)  # add prediction model to list for plotting
np.savetxt('results.dat', validation_results)  # save data (validx, Y_pred_val)

# Print out calculated outputs of the validation phase
print("\n--Validation results")  # DEBUG
MAE = np.mean(np.abs(Y_pred_val - validy))
print("MAE:\t", round(MAE, 4))  # TODO explain MAE ->
MSE = np.mean(np.abs(Y_pred_val - validy) ** 2)
print("MSE:\t", round(MSE, 4))  # TODO explain MSE ->mean_squared_error
score = model.score(validx, validy)
print("Score:\t", round(score, 4))  # TODO explain score ->
print(f"Epochs:  {model.n_iter_}")
print("Duration (train+predict) %s s" % (round((stop_time - start_time), 3)))

# plotFile('results.dat', "Prediction model (red)")  # plot output data
plotFileValidation('results.dat', validx, validy,
                   "Prediction model (red) and Validation data (blue)")  # plot (validx, Y_pred_val) and true validation

#################################
### Test Phase:
### Prepare test sets
### load the data from the file 'data-test.*.dat' into validx and and validy.
# Once your ANN is able to describe the validation data with good accuracy,
# you can adapt this script to run also over the test data in 'data/data-test.1.dat' and 'data/data-test.2.dat'.
# Show the results for the test data, and comment on the limits of applicability of your ANN model.

if testOne:
    # test 1
    # range: x-axis 0-2.1
    test1 = np.loadtxt('data/data-test.1.dat')
    test1x = test1[:, 0].reshape(-1, 1)  # X: (96,1)
    test1y = test1[:, 1]  # Y: (96,)
    print("--Test 1 data--")  # DEBUG
    # plotter(test1x, test1y, "Test 1 data")  # DEBUG plot test1 data

    ### Test phase for TEST 1
    Y_pred_test1 = model.predict(test1x)  # Y_pred_test1 is the prediction of Y values in the test phase test1
    test1_results = np.column_stack((test1x, Y_pred_test1))  # stack output data
    allPlots.append(test1_results)  # add test1 prediction dataset to list for plotting
    np.savetxt('results.test.1.dat', test1_results)  # save output data

    # Print out calculated outputs of the validation phase
    MAE = np.mean(np.abs(Y_pred_test1 - test1y))
    print("MAE: ", round(MAE, 4))
    MSE = np.mean(np.abs(Y_pred_test1 - test1y) ** 2)
    print("MSE: ", round(MSE, 4))
    score = model.score(test1x, test1y)
    print("Score: ", round(score, 4))
    print(f"Epochs:  {model.n_iter_}")

    plotFileValidation('results.test.1.dat', test1x, test1y,
                       "Test 1 prediction (red) and true test data (blue)")  # plot both
    #plotter(test1y, Y_pred_test1, "Test 1: prediction vs. true values")

if testTwo:
    # test 2
    # range: x-axis 9.9-12.5
    test2 = np.loadtxt('data/data-test.2.dat')
    test2x = test2[:, 0].reshape(-1, 1)  # X: (212,1)
    test2y = test2[:, 1]  # Y: (212,)
    print("--Test 2 data--")  # DEBUG
    # plotter(test2x, test2y, "Test 2 data")  # DEBUG plot test2 data

    ### Test phase for TEST 2
    Y_pred_test2 = model.predict(test2x)  # Y_pred_test2 is the prediction of Y values in the test phase in test2
    test2_results = np.column_stack((test2x, Y_pred_test2))  # stack output data
    allPlots.append(test2_results)  # add test2 prediction dataset to list for plotting
    np.savetxt('results.test.2.dat', test2_results)  # save output data

    # Print out calculated outputs of the validation phase
    MAE = np.mean(np.abs(Y_pred_test2 - test2y))
    print("MAE: ", round(MAE, 4))
    MSE = np.mean(np.abs(Y_pred_test2 - test2y) ** 2)
    print("MSE: ", round(MSE, 4))
    score = model.score(test2x, test2y)
    print("Score: ", round(score, 4))
    print(f"Epochs:  {model.n_iter_}")

    plotFileValidation('results.test.2.dat', test2x, test2y,
                       "Test 2 prediction (red) and true test data (blue)")  # plot both
    #plotter(test2y, Y_pred_test2, "Test 2: prediction vs. true values")

allPlotter(allPlots)  # plot all computed datasets
