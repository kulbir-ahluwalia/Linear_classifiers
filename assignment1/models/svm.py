class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float,batch_size:int):
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
        self.batch_size = batch_size
        self.learning_rate_exponent = learning_rate_exponent

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        x = X_train

        y_hat_list = self.reg_const + np.dot(self.reg_const + self.w, x)  # get the dot product of weight and feature

        return y_hat_list


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape
        batch_size = self.batch_size
        #self.w = np.random.uniform(low=0.1, high=0.8,size=(N,D))
        #self.w = np.zeros((N,D))
        self.w = np.random.rand(self.n_class,D)
        #print(self.w.shape)

        for iter in range(self.epochs):
          #if iter > 4:
          # self.lr -= iter*self.lr/9
          self.lr *= (self.learning_rate_exponent ** iter)
          # self.lr = self.lr * math.exp(-1*(self.learning_rate_exponent)*iter)
          print("lr: ",self.lr)

          #if self.lr > 6:
          # self.reg_const /= 0.9
          print("reg constant: ",self.reg_const)

          for example_num in range(0,N,batch_size):
            # print("example_num is: ",example_num)
            x = X_train[example_num]
            y_label = y_train[example_num]
            #print(y_label)
            #print(x.shape)
            y_hat_list = self.calc_gradient(x,y_label)
            y_correct = y_hat_list[y_label]
            #print(y_correct)
            #break
    
            for class_num in range(self.n_class):
              if y_correct != y_hat_list[class_num]:
                if y_correct - y_hat_list[class_num] < 1: 
                  self.w[y_label] = self.w[y_label] + self.lr*(x)
                  self.w[class_num] = self.w[class_num] - self.lr*(x)

              self.w[class_num] = (1 - self.lr*(self.reg_const/self.n_class))*self.w[class_num]
          print("weights are: ",self.w)
          print("Epoch number finished: ",iter)
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