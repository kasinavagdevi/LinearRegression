#####################################################################################################################
#   CS 6375.003 - Assignment 1, Linear Regression using Gradient Descent
#   This is a simple starter code in Python 3.6 for linear regression using the notation shown in class.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   test - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self, train):
        np.random.seed(1)
        # train refers to the training dataset
        # stepSize refers to the step size of gradient descent
        self.df = pd.read_csv(train)
        self.df.insert(0, 'X0', 1)
        self.preProcess()
        self.nrows, self.ncols = self.df.shape[0], self.df.shape[1]
        self.X =  self.df.iloc[:, 0:(self.ncols -1)].values.reshape(self.nrows, self.ncols-1)
        self.y = self.df.iloc[:, (self.ncols-1)].values.reshape(self.nrows, 1)
        self.W = np.random.rand(self.ncols-1).reshape(self.ncols-1, 1)

    # TODO: Perform pre-processing for your dataset. It may include doing the following:
    #   - getting rid of null values
    #   - converting categorical to numerical values
    #   - scaling and standardizing attributes
    #   - anything else that you think could increase model performance
    # Below is the pre-process function
    def preProcess(self):
        self.df = LinearRegression.preProcessCsv(self.df)

    @staticmethod
    def preProcessCsv(df):
        # print(self.df.isnull().sum())
        # removing null values in the data. This data has no null values.
        df = df.dropna()
        # There are no categorical variables
        # Removing duplicate values from the data
        df = df.drop_duplicates()

        # Scaling the features between 0 and 1.
        for column in df.columns[1:12]:
            df[column] = df[column] - df[column].min()
            df[column] = df[column] / df[column].max()
        # print(self.df.isnull().sum())
        return df

    # Below is the training function
    def train(self, epochs = 10000, learning_rate = 0.1):
        # Perform Gradient Descent
        for i in range(epochs):
            # Make prediction with current weights
            h = np.dot(self.X, self.W)
            # Find error
            error = h - self.y
            self.W = self.W - (1 / self.nrows) * learning_rate * np.dot(self.X.T, error)

        return self.W, error

    # predict on test dataset
    def predict(self, file):
        testDF = pd.read_csv(file)
        testDF = LinearRegression.preProcessCsv(testDF)
        testDF.insert(0, "X0", 1)
        nrows, ncols = testDF.shape[0], testDF.shape[1]
        testX = testDF.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        testY = testDF.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        pred = np.dot(testX, self.W)
        error = pred - testY
        mse = 1/(2*nrows) * np.dot(error.T, error)
        return mse


if __name__ == "__main__":
    model = LinearRegression("train.csv")
    W, e = model.train()
    mse = model.predict("train.csv")
    print(mse)
    mse = model.predict("test.csv")
    print(mse)
