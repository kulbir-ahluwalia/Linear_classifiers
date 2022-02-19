"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        #N, D = X_train.shape
        #print(N,D)
        #gradients = np.zeros((N,D))
        x = X_train

        y_hat_list = np.dot(self.reg_const + self.w, x)  # get the dot product of weight and feature
        #print(y_hat_list)
        #exp_y = np.exp(y_hat_list)
        #print(exp_y)
        log_k = -np.max(y_hat_list)
        exp_y = np.exp(y_hat_list + log_k)
        sum_exp_y = np.sum(exp_y)
        gradients = exp_y / sum_exp_y

        return gradients

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape
        #self.w = np.random.uniform(low=0.1, high=0.8,size=(N,D))
        #self.w = np.zeros((N,D))
        self.w = np.random.rand(self.n_class,D)
        #print(self.w.shape)

        for iter in range(self.epochs):
          #if iter > 4:
          self.lr -= iter*self.lr/5

          #if self.lr > 6:
          self.reg_const /= 0.9

          for example_num in range(N):
            x = X_train[example_num]
            y_label = y_train[example_num]
            #print(y_label)
            #print(x.shape)
            gradients = self.calc_gradient(x,y_label)
            #print(gradients)
            #break
    
            for class_num in range(self.n_class):
              if class_num == y_label:
                self.w[y_label] = self.w[y_label] + (self.lr*(1 - gradients[y_label]))*x
              else:
                self.w[class_num] = self.w[class_num] - (self.lr*(gradients[class_num]))*x


        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        N, D = X_test.shape
        labels = np.zeros(N)

        for image_num in range(N):
          x = X_test[image_num]
          y_hat_list = np.dot(self.w, x)
          labels[image_num] = np.argmax(y_hat_list)
          if self.n_class == 2:
            labels[image_num] = np.where(labels[image_num] == -1, 0, labels[image_num])
 
        return labels