from optimizers import sgd_optimizer, svm_optimizer, adagrad_optimizer
from features import FeatureVector, Features
from helpers import compute_features,read_data, evaluate, write_predictions
from decode import decode

def hamming_loss(gold,predicted):
    return (10*int(gold!=predicted))

def hamming_loss_modified(gold,predicted):
    if gold!='O' and predicted=='O':
        return 30*int(gold!=predicted)
    return 10*int(gold!=predicted)

def train(data, feature_names, tagset, epochs, method, optimizer, step_size=1.0, l2=None):
    """
    Trains the model on the data and returns the parameters
    :param data: Array of dictionaries representing the data.  One dictionary for each data point (as created by the
        make_data_point function).
    :param feature_names: Array of Strings.  The list of feature names.
    :param tagset: Array of Strings.  The list of tags.
    :param epochs: Int. The number of epochs to train
    :return: FeatureVector. The learned parameters.
    """
    parameters = FeatureVector({})   # creates a zero vector

    def perceptron_gradient(i):
        """
        Computes the gradient of the Perceptron loss for example i
        :param i: Int
        :return: FeatureVector
        """
        inputs = data[i]
        input_len = len(inputs['tokens'])
        gold_labels = inputs['gold_tags']
        features = Features(inputs, feature_names)

        def score(cur_tag, pre_tag, i):
            return parameters.dot_product(features.compute_features(cur_tag, pre_tag, i))

        tags = decode(input_len, tagset, score)
        fvector = compute_features(tags, input_len, features)           # Add the predicted features
        fvector.times_plus_equal(-1, compute_features(gold_labels, input_len, features))    # Subtract the features for the gold labels
        return fvector

    def svm_gradient(i):
        """
        Computes the gradient of the SVM loss for example i
        :param i: Int
        :return: FeatureVector
        """
        inputs = data[i]
        input_len = len(inputs['tokens'])
        gold_labels = inputs['gold_tags']
        features = Features(inputs, feature_names)

        def score_svm(cur_tag, pre_tag, i):
            return parameters.dot_product(features.compute_features(cur_tag, pre_tag, i)) + hamming_loss(gold_labels[i],cur_tag)

        tags = decode(input_len, tagset, score_svm)
        fvector = compute_features(tags, input_len, features)           # Add the predicted features
        fvector.times_plus_equal(-1, compute_features(gold_labels, input_len, features))    # Subtract the features for the gold labels
        return fvector
    
    def svm_modified_gradient(i):
        """
        Computes the gradient of the SVM loss for example i
        :param i: Int
        :return: FeatureVector
        """
        inputs = data[i]
        input_len = len(inputs['tokens'])
        gold_labels = inputs['gold_tags']
        features = Features(inputs, feature_names)

        def score_svm(cur_tag, pre_tag, i):
            return parameters.dot_product(features.compute_features(cur_tag, pre_tag, i)) + hamming_loss_modified(gold_labels[i-1],pre_tag)

        tags = decode(input_len, tagset, score_svm)
        fvector = compute_features(tags, input_len, features)           # Add the predicted features
        fvector.times_plus_equal(-1, compute_features(gold_labels, input_len, features))    # Subtract the features for the gold labels
        return fvector
    
    def training_observer(epoch, parameters):
        """
        Evaluates the parameters on the development data, and writes out the parameters to a 'model.iter'+epoch and
        the predictions to 'ner.dev.out'+epoch.
        :param epoch: int.  The epoch
        :param parameters: Feature Vector.  The current parameters
        :return: Double. F1 on the development data
        """
        print ('---- Dev Data -----')
        dev_data = read_data('ner.dev')
        # dev_data = read_data('ner.train')[:2]
        (_, _, f1) = evaluate(dev_data, parameters, feature_names, tagset)
        write_predictions('outputs/ner.dev.out'+str(epoch), dev_data, parameters, feature_names, tagset)
        parameters.write_to_file('outputs/model.iter'+str(epoch))
        
        print ('---- Test Data -----')
        test_data = read_data('ner.test')
        # test_data = read_data('ner.train')[:2]
        (_, _, f1) = evaluate(test_data, parameters, feature_names, tagset)
        write_predictions('outputs/ner.test.out'+str(epoch), test_data, parameters, feature_names, tagset)
        return f1

    print (f'------------Method: {method} optimizer: {optimizer}--------------')
    print (f'-------- Feature Names: {feature_names}')

    if method=='svm':
        return svm_optimizer(len(data), epochs, svm_gradient, parameters, training_observer, alpha=step_size, lamda=l2)
    if method=='svm_modified':
        return svm_optimizer(len(data), epochs, svm_modified_gradient, parameters, training_observer)
    if method=='structured_perceptron' and optimizer=='sgd':
        return sgd_optimizer(len(data), epochs, perceptron_gradient, parameters, training_observer)
    if method=='structured_perceptron' and optimizer=='adagrad':
        return adagrad_optimizer(len(data), epochs, perceptron_gradient, parameters, training_observer)
