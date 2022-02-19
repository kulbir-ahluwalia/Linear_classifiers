import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape

        #self.w = np.random.rand(self.n_class,D)  # create a weight matrix of shape (1,D)
        self.w = np.zeros((self.n_class,D))
        #print(self.w)
        #print(self.w.shape)
        #print(y_train[0:20])
        for iter in range(self.epochs):
          #if iter > 5:
          #  self.lr = 0.5
          for example_num in range(N):
            x = X_train[example_num]
            y_label = y_train[example_num]
            y_hat_list = np.dot(self.w, x)  # get the dot product of weight and feature
            #print(y_label,y_hat_list)
            y_hat_max = np.argmax(y_hat_list)

            if y_label == y_hat_max:
              pass
            else:     # update weight
              y_yi = y_hat_list[y_label]      # correct label w^T_yi*xi
              #y_c = np.argwhere(y_hat_list > y_yi).reshape(1,-1)  # all labels higher than y_yi

              coef_x = (self.lr)*x

              for class_num in range(self.n_class):
                if iter == 0:
                  #if class_num == y_label:
                  self.w[y_label] = self.w[y_label] + coef_x
                  #else:
                  self.w[class_num] = self.w[class_num] - coef_x

                if y_hat_list[class_num] > y_yi:
                  self.w[y_label] = self.w[y_label] + coef_x
                  self.w[class_num] = self.w[class_num] - coef_x

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
        labels = np.zeros((N))
        #print(self.w.shape)
        for example_num in range(N):
          x = X_test[example_num]
          y_hat = np.dot(self.w,x)
          labels[example_num] = np.argmax(y_hat)


        return labels