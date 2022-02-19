def scalar_value_of_sigmoid(sigmoid_input):
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        
        sigmoid_value = 1/(1+math.exp(-1*sigmoid_input))
        # print("sigmoid function returns: ",sigmoid_value)
        return sigmoid_value


# find the gradient of loss at a point
def sgd_gradient_of_loss_for_a_point(weight_vec,y_sub_i,x_sub_i,learning_rate,sigmoid_input,x_cols):


  # print("y_sub_i is: ",y_sub_i)

  sigmoid_input_for_gradient = -1*y_sub_i*(np.dot(x_sub_i,weight_vec))
  # print("sigmoid input of gradient is: ",sigmoid_input_for_gradient)
  # print("shape of sigmoid input of gradient is: ",sigmoid_input_for_gradient.shape)

  # print("x_sub_i is: ",x_sub_i)
  # print("x_sub_i shape is: ",x_sub_i.shape)

  # print("weight_vec is: ",weight_vec)
  # print("weight_vec shape is: ",weight_vec.shape)

  output_of_sigmoid_function = scalar_value_of_sigmoid(sigmoid_input_for_gradient)
  # print("output_of_sigmoid_function is: ",output_of_sigmoid_function)  


  gradient_of_loss_multiplied_by_eta = (x_sub_i)*learning_rate*(output_of_sigmoid_function)*y_sub_i
  # print("gradient_of_loss_multiplied_by_eta is: ",gradient_of_loss_multiplied_by_eta)
  # print("shape of gradient_of_loss_multiplied_by_eta is: ",gradient_of_loss_multiplied_by_eta.shape)
  gradient_of_loss_multiplied_by_eta = gradient_of_loss_multiplied_by_eta.reshape(x_cols,1)

  return gradient_of_loss_multiplied_by_eta


def get_acc(pred, y_test):
    return np.sum(y_test == pred) / len(y_test) * 100


"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.weight_vec = None  # TODO: change this
        self.lr = lr
        self.epoch_number = epochs
        self.threshold = threshold
        self.logistic_loss = []

    #for sigmoid, see function outside
    # def sigmoid(self, z: np.ndarray) -> np.ndarray:
    #     """Sigmoid function.

    #     Parameters:
    #         z: the input

    #     Returns:
    #         the sigmoid of the input
    #     """
    #     exp_z = np.exp(-z)
    #     # print("exp_z is: ",exp_z)
    #     # ones_array = np.ones(len(z))
    #     sum = 1+exp_z 
    #     print("sum is: ",sum)
    #     sigmoid_value = 1/(1+exp_z)
    #     print("sigmoid function returns: ",sigmoid_value)
    #     return sigmoid_value

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the *logistic regression update rule* as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        #in class notes, x_rows=n and x_cols=d
        x_rows,x_cols = X_train.shape
        self.weight_vec = np.zeros((x_cols,1))
        # print("self.weight_vec is: ",self.weight_vec)
        # print("self.weight_vec.shape is: ",self.weight_vec.shape)

        #reshape y_train to a column vector that is n by 1, 
        y_train = y_train.reshape(x_rows,1)
  

        #loop for each epoch
        for epoch_number in range(self.epoch_number):

            #we need to iterate over the weight matrix and take each row as input
            for x_row_index in range(x_rows):
                  x_row_for_example = X_train[x_row_index]
                  # print("x_row_for_example:",x_row_for_example)
                  y_label = y_train[x_row_index]
                  # print("y_label:",y_label)

                  sigmoid_input = y_label*np.dot(x_row_for_example,self.weight_vec)
                  sigmoid_output = scalar_value_of_sigmoid(sigmoid_input)
                  delta_weight_vector = sgd_gradient_of_loss_for_a_point(self.weight_vec,y_label,x_row_for_example,self.lr,sigmoid_input,x_cols)
                  # print("delta_weight_vector is: ",delta_weight_vector)
                  
                  # Updating the weight vector.
                  # print("weight vector before update is: ",self.weight_vec)
                  self.weight_vec = self.weight_vec + delta_weight_vector
                  # print("weight vector after update is: ",self.weight_vec)
              
            
        return self.weight_vec


        

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
          y_hat = np.dot(x,self.weight_vec)
          if y_hat>=self.threshold:
            labels[example_num] = 1
          else:
            labels[example_num] = -1


        return labels


