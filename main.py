
import os
import sys
import random
import math

from train import train
from helpers import read_data
from features import FeatureVector
random.seed(1234)

def main_predict(data_filename, model_filename):
    """
    Main function to make predictions.
    Loads the model file and runs the NER tagger on the data, writing the output in CoNLL 2003 evaluation format to data_filename.out
    :param data_filename: String
    :param model_filename: String
    :return: None
    """
    data = read_data(data_filename)
    parameters = FeatureVector({})
    parameters.read_from_file(model_filename)

    tagset = ['B-PER', 'B-LOC', 'B-ORG', 'B-MISC', 'I-PER', 'I-LOC', 'I-ORG', 'I-MISC', 'O']

    feature_names = ['current_word', 'prev_tag', 'lowercase','current_pos_tag','shape',
        'prev_next_word_features','word_lower_pos','length_k','gazetteer','uppercase','position' ]
    
    write_predictions(data_filename+'.out', data, parameters, feature_names, tagset)
    evaluate(data, parameters, feature_names, tagset)

    return


def main_train(method='structured_perceptron',optimizer='sgd',step_size=1.0, l2=None, epochs=20, is_only_four_features=False):
    """
    Main function to train the model
    :return: None
    """
    print('Reading training data')
    train_data = read_data('ner.train')
    # train_data = read_data('ner.train')[:2]

    print ('Size of training data: ', len(train_data))
    #train_data = read_data('ner.train')[1:1] # if you want to train on just one example

    tagset = ['B-PER', 'B-LOC', 'B-ORG', 'B-MISC', 'I-PER', 'I-LOC', 'I-ORG', 'I-MISC', 'O']
    feature_names = ['current_word', 'prev_tag', 'lowercase','current_pos_tag','shape',
        'prev_next_word_features','word_lower_pos','length_k','gazetteer','uppercase','position' ]
    
    print('Training...')

    if is_only_four_features:
        feature_names = feature_names[:4]

    if method=='structured_perceptron':
        parameters = train(train_data, feature_names, tagset, epochs=20, method='structured_perceptron', optimizer=optimizer, step_size=step_size, l2=l2)
    if method=='svm':
        parameters = train(train_data, feature_names, tagset, epochs=20,  method='svm', optimizer='svm', step_size=step_size, l2=l2)
    if method=='svm_modified':
        parameters = train(train_data, feature_names, tagset, epochs=20,  method='svm_modified', optimizer='svm', step_size=step_size, l2=l2)

    print('Training done')

    # dev_data = read_data('ner.dev')
    # evaluate(dev_data, parameters, feature_names, tagset)
    # parameters.write_to_file('model')

    return


if __name__ == '__main__':

    os.mkdir('outputs/')
    # test_decoder()

    # main_train(method='structured_perceptron',optimizer='sgd',step_size=1.0, l2=None, epochs=20, is_only_four_features=True)
    # print ('\n\n','='*20)

    # main_train(method='structured_perceptron',optimizer='sgd',step_size=1.0, l2=None, epochs=20, is_only_four_features=False)
    # print ('\n\n','='*20)

    # main_train(method='structured_perceptron',optimizer='adagrad',step_size=1.0, l2=None, epochs=20, is_only_four_features=False)

    # print ('\n\n','='*20)
    main_train(method='svm',optimizer='svm',step_size=1.0, l2=0.0001, epochs=10, is_only_four_features=False)

    print ('\n\n','='*20)
    # main_train(method='svm_modified',optimizer='svm',step_size=1.0, l2=0.0001, epochs=20, is_only_four_features=False,)