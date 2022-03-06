from tqdm import tqdm
from features import FeatureVector
from copy import deepcopy

def sgd_optimizer(training_size, epochs, gradient, parameters, training_observer):
    """
    Stochastic gradient descent
    :param training_size: int. Number of examples in the training set
    :param epochs: int. Number of epochs to run SGD for
    :param gradient: func from index (int) in range(training_size) to a FeatureVector of the gradient
    :param parameters: FeatureVector.  Initial parameters.  Should be updated while training
    :param training_observer: func that takes epoch and parameters.  You can call this function at the end of each
           epoch to evaluate on a dev set and write out the model parameters for early stopping.
    :return: final parameters
    """
    # Look at the FeatureVector object.  You'll want to use the function times_plus_equal to update the
    # parameters.
    # To implement early stopping you can call the function training_observer at the end of each epoch.
    max_patience = 3
    best_f1 = float('-inf')
    best_parameters = None
    patience = 0

    for epoch in range(epochs):
        # print ('Hello')
        print ('-'*20,'Epoch:',epoch+1,'-'*20)
        for i in tqdm(range(training_size)):
            parameters.times_plus_equal(-1,gradient(i))

        cur_f1 = training_observer(epoch, parameters)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_parameters = parameters
            patience = 0
        else:
            patience +=1
            if patience > max_patience:
                return best_parameters

    return best_parameters

def svm_optimizer(training_size, epochs, gradient, parameters, training_observer, alpha=1.0, lamda=0.0):
    """
    SVM
    :param training_size: int. Number of examples in the training set
    :param epochs: int. Number of epochs 
    :param gradient: func from index (int) in range(training_size) to a FeatureVector of the gradient
    :param parameters: FeatureVector.  Initial parameters.  Should be updated while training
    :param training_observer: func that takes epoch and parameters.  You can call this function at the end of each
           epoch to evaluate on a dev set and write out the model parameters for early stopping.

    :param alpha: step size
    :param lamda: regularization strength
    :return: final parameters
    """

    max_patience = 3
    best_f1 = float('-inf')
    best_parameters = None
    patience = 0
    print (f'--------- Stepsize: {alpha}, lambda: {lamda} ---------')

    for epoch in range(epochs):
        print ('-'*20,'Epoch:',epoch+1,'-'*20)
        
        for i in tqdm(range(training_size)):

            parameters.times_plus_equal(-alpha, gradient(i)) # w - α g
            parameters.times_plus_equal(-alpha*lamda, deepcopy(parameters)) # # w - αg - αλw

        cur_f1 = training_observer(epoch, parameters)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_parameters = parameters
            patience = 0
        else:
            patience +=1
            if patience > max_patience:
                return best_parameters

    return best_parameters

def adagrad_optimizer(training_size, epochs, gradient, parameters, training_observer):
    """
    Adagrad
    :param training_size: int. Number of examples in the training set
    :param epochs: int. Number of epochs to run SGD for
    :param gradient: func from index (int) in range(training_size) to a FeatureVector of the gradient
    :param parameters: FeatureVector.  Initial parameters.  Should be updated while training
    :param training_observer: func that takes epoch and parameters.  You can call this function at the end of each
           epoch to evaluate on a dev set and write out the model parameters for early stopping.
    :return: final parameters
    """
    # Look at the FeatureVector object.  You'll want to use the function times_plus_equal to update the
    # parameters.
    # To implement early stopping you can call the function training_observer at the end of each epoch.
    max_patience = 3
    best_f1 = float('-inf')
    best_parameters = None
    patience = 0
    gradient_matrix = FeatureVector({}) # s

    for epoch in range(epochs):
        print ('-'*20,'Epoch:',epoch+1,'-'*20)

        for i in tqdm(range(training_size)):
            
            gradient_t = gradient(i) #g_{t,i}
            gradient_matrix.times_plus_equal(1,gradient_t.square()) # s_{t,i} = s_{t-1,i} + g_{t,i}^2
            gradient_t_square_running = gradient_matrix.current_params(gradient_t) # get gradient history of only current parameters
            parameters.times_plus_equal(-1,gradient_t.divide(gradient_t_square_running.square_root()))
            # print ('PF',parameters.fdict)

        cur_f1 = training_observer(epoch, parameters)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_parameters = parameters
            patience = 0
        else:
            patience +=1
            if patience > max_patience:
                return best_parameters

    return best_parameters
