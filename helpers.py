from features import FeatureVector, Features
from decode import decode
from conlleval import evaluate as conllevaluate


def predict(inputs, input_len, parameters, feature_names, tagset):
    """
    
    :param inputs:
    :param input_len:
    :param parameters:
    :param feature_names:
    :param tagset:
    :return:
    """
    features = Features(inputs, feature_names)

    def score(cur_tag, pre_tag, i):
        return parameters.dot_product(features.compute_features(cur_tag, pre_tag, i))
    return decode(input_len, tagset, score)


def make_data_point(sent):
    """
        Creates a dictionary from String to an Array of Strings representing the data.  The dictionary items are:
        dic['tokens'] = Tokens padded with <START> and <STOP>
        dic['pos'] = POS tags padded with <START> and <STOP>
        dic['NP_chunk'] = Tags indicating noun phrase chunks, padded with <START> and <STOP>
        dic['gold_tags'] = The gold tags padded with <START> and <STOP>
    :param sent: String.  The input CoNLL format string
    :return: Dict from String to Array of Strings.
    """
    dic = {}
    sent = [s.strip().split() for s in sent]
    dic['tokens'] = ['<START>'] + [s[0] for s in sent] + ['<STOP>']
    dic['pos'] = ['<START>'] + [s[1] for s in sent] + ['<STOP>']
    dic['NP_chunk'] = ['<START>'] + [s[2] for s in sent] + ['<STOP>']
    dic['gold_tags'] = ['<START>'] + [s[3] for s in sent] + ['<STOP>']
    return dic

def read_data(filename):
    """
    Reads the CoNLL 2003 data into an array of dictionaries (a dictionary for each data point).
    :param filename: String
    :return: Array of dictionaries.  Each dictionary has the format returned by the make_data_point function.
    """
    data = []
    with open(filename, 'r') as f:
        sent = []
        for line in f.readlines():
            if line.strip():
                sent.append(line)
            else:
                data.append(make_data_point(sent))
                sent = []
        data.append(make_data_point(sent))

    return data

def write_predictions(out_filename, all_inputs, parameters, feature_names, tagset):
    """
    Writes the predictions on all_inputs to out_filename, in CoNLL 2003 evaluation format.
    Each line is token, pos, NP_chuck_tag, gold_tag, predicted_tag (separated by spaces)
    Sentences are separated by a newline
    The file can be evaluated using the command: python conlleval.py < out_file
    :param out_filename: filename of the output
    :param all_inputs:
    :param parameters:
    :param feature_names:
    :param tagset:
    :return:
    """
    with open(out_filename, 'w', encoding='utf-8') as f:
        for inputs in all_inputs:
            input_len = len(inputs['tokens'])
            tag_seq = predict(inputs, input_len, parameters, feature_names, tagset)
            for i, tag in enumerate(tag_seq[1:-1]):  # deletes <START> and <STOP>
                f.write(' '.join([inputs['tokens'][i+1], inputs['pos'][i+1], inputs['NP_chunk'][i+1], inputs['gold_tags'][i+1], tag])+'\n') # i + 1 because of <START>
            f.write('\n')

def evaluate(data, parameters, feature_names, tagset):
    """
    Evaluates precision, recall, and F1 of the tagger compared to the gold standard in the data
    :param data: Array of dictionaries representing the data.  One dictionary for each data point (as created by the
        make_data_point function)
    :param parameters: FeatureVector.  The model parameters
    :param feature_names: Array of Strings.  The list of features.
    :param tagset: Array of Strings.  The list of tags.
    :return: Tuple of (prec, rec, f1)
    """
    all_gold_tags = [ ]
    all_predicted_tags = [ ]
    for inputs in data:
        all_gold_tags.extend(inputs['gold_tags'][1:-1])  # deletes <START> and <STOP>
        input_len = len(inputs['tokens'])
        all_predicted_tags.extend(predict(inputs, input_len, parameters, feature_names, tagset)[1:-1]) # deletes <START> and <STOP>
    return conllevaluate(all_gold_tags, all_predicted_tags)


def compute_score(tag_seq, input_length, score):
    """
    Computes the total score of a tag sequence
    :param tag_seq: Array of String of length input_length. The tag sequence including <START> and <STOP>
    :param input_length: Int. input length including the padding <START> and <STOP>
    :param score: function from current_tag (string), previous_tag (string), i (int) to the score.  i=0 points to
        <START> and i=1 points to the first token. i=input_length-1 points to <STOP>
    :return:
    """
    total_score = 0
    for i in range(1, input_length):
        total_score += score(tag_seq[i], tag_seq[i - 1], i)
    return total_score


def compute_features(tag_seq, input_length, features):
    """
    Compute f(xi, yi)
    :param tag_seq: [tags] already padded with <START> and <STOP>
    :param input_length: input length including the padding <START> and <STOP>
    :param features: func from token index to FeatureVector
    :return:
    """
    feats = FeatureVector({})
    for i in range(1, input_length):
        feats.times_plus_equal(1, features.compute_features(tag_seq[i], tag_seq[i - 1], i))
    return feats

