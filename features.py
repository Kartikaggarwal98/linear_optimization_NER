
from collections import defaultdict
import math

def load_gazetteer_dict():
    with open('./gazetteer.txt') as f:
        lines = f.readlines()
        lines = [i[:-1] for i in lines]
        g_dict = defaultdict(set)
        for line in lines:
            tag, word = line.split()[0], (' ').join(line.split()[1:])
            g_dict[tag].add(word) #stores a set of words for each tag
    
    # print ('gazetteer dict sample: ',g_dict.keys())
    return g_dict
g_dict = load_gazetteer_dict()


class Features(object):
    def __init__(self, inputs, feature_names):
        """
        Creates a Features object
        :param inputs: Dictionary from String to an Array of Strings.
            Created in the make_data_point function.
            inputs['tokens'] = Tokens padded with <START> and <STOP>
            inputs['pos'] = POS tags padded with <START> and <STOP>
            inputs['NP_chunk'] = Tags indicating noun phrase chunks, padded with <START> and <STOP>
            inputs['gold_tags'] = DON'T USE! The gold tags padded with <START> and <STOP>
        :param feature_names: Array of Strings.  The list of features to compute.
        """
        self.feature_names = feature_names
        self.inputs = inputs
        

    def compute_features(self, cur_tag, pre_tag, i):
        """
        Computes the local features for the current tag, the previous tag, and position i
        :param cur_tag: String.  The current tag.
        :param pre_tag: String.  The previous tag.
        :param i: Int. The position
        :return: FeatureVector
        """
        feats = FeatureVector({})
        # print (self.feature_names)
        cur_word = self.inputs['tokens'][i]

        if 'current_word' in self.feature_names:#Feature 1 :  Wi=France+Ti=I-LOC 1.0
            feats.times_plus_equal(1, FeatureVector({'Wi='+cur_word+'+Ti='+cur_tag: 1}))

        if 'prev_tag' in self.feature_names:#Feature 2 : Ti-1=<START>+Ti=I-LOC 1.0
            feats.times_plus_equal(1, FeatureVector({'Ti='+cur_tag+"+Ti-1="+pre_tag: 1}))

        if 'lowercase' in self.feature_names:#Feature 3 : Oi=france+Ti=I-LOC 1.0
                feats.times_plus_equal(1, FeatureVector({'Oi='+cur_word.lower()+'+Ti='+cur_tag:1}))

        if 'current_pos_tag' in self.feature_names: #Feature 4 : Pi=NNP+Ti=I-LOC 1.0
            feats.times_plus_equal(1, FeatureVector({'Pi='+self.inputs['pos'][i]+'+Ti='+cur_tag:1}))
            
        if 'shape' in self.feature_names: #Feature 5 : Si=Aaaaaa+Ti=I-LOC 1.0
            new_word = ('').join(['A' if i.isupper() else 'a' if i.islower() else 'd' if i in '0123456789' else i for i in cur_word])
            feats.times_plus_equal(1, FeatureVector({'Si='+new_word+'+Ti='+cur_tag: 1}))

        if 'prev_next_word_features' in self.feature_names: #Feature 6
            prev_word  = self.inputs['tokens'][i-1]
            feats.times_plus_equal(1, FeatureVector({'Wi-1='+prev_word+'+Ti='+cur_tag: 1})) #Wi-1=<START>+Ti=I-LOC 1.0
            feats.times_plus_equal(1, FeatureVector({'Oi-1='+prev_word.lower()+'+Ti='+cur_tag: 1})) #Oi-1=france+Ti=I-LOC 1.0
            feats.times_plus_equal(1, FeatureVector({"Pi-1="+self.inputs['pos'][i-1]+'+Ti='+cur_tag: 1})) # Pi-1=<START>+Ti=I-LOC 1.0
            
            if cur_word!='<STOP>':
                next_word = self.inputs['tokens'][i+1]
                feats.times_plus_equal(1, FeatureVector({'Wi+1='+next_word+'+Ti='+cur_tag: 1}))
                feats.times_plus_equal(1, FeatureVector({'Oi+1='+next_word.lower()+'+Ti='+cur_tag: 1}))
                # if next_word!='<STOP>':
                    # feats.times_plus_equal(1, FeatureVector({'Ti='+cur_tag+"+Ti+1="+self.inputs['NP_chunk'][i+1]: 1}))
                feats.times_plus_equal(1, FeatureVector({"Pi+1="+self.inputs['pos'][i+1]+'+Ti='+cur_tag: 1}))

        if 'word_lower_pos' in self.feature_names: #Feature 7 : Wi=France+Oi=france+Pi=NNP+Ti-1=<START>+Ti=I-LOC 1.0
            feats.times_plus_equal(1, FeatureVector({'Wi='+cur_word+'+Oi='+cur_word.lower()+'+P_i='+self.inputs['pos'][i]+'+Ti-1='+pre_tag+'+Ti='+cur_tag: 1}))

        if 'length_k' in self.feature_names: #Feature 8 : PREi=Fr+Ti=I-LOC 1.0
            for i in range(min(4,len(cur_word))):
                feats.times_plus_equal(1, FeatureVector({'PREi='+cur_word[:i+1]+'Ti='+cur_tag: 1}))

        if 'gazetteer' in self.feature_names: #Feature 9 : GAZi=True+Ti=I-LOC 1.0
            if cur_word!='<STOP>':
                if cur_tag!='O' and cur_word in g_dict[cur_tag.split('-')[1]]:
                    feats.times_plus_equal(1, FeatureVector({'GAZi=True'+'Ti='+cur_tag: 1}))
                else:
                    feats.times_plus_equal(1, FeatureVector({'GAZi=False'+'Ti='+cur_tag: 1}))
            else:
                feats.times_plus_equal(1, FeatureVector({'GAZi=False'+'Ti='+cur_tag: 1}))

        if 'uppercase' in self.feature_names: #Feature 10 : CAPi=True+Ti=I-LOC 1.0
            if cur_word[0].isupper(): feats.times_plus_equal(1, FeatureVector({'CAPi=True+Ti='+cur_tag:1}))
            else: feats.times_plus_equal(1, FeatureVector({'CAPi=False+Ti='+cur_tag:1}))

        if 'position' in self.feature_names: #Feature 11 : POSi=1+Ti=I-LOC 1.0
            feats.times_plus_equal(1, FeatureVector({'POSi='+str(i)+'+Ti='+cur_tag: 1}))

        return feats


class FeatureVector(object):

    def __init__(self, fdict):
        self.fdict = fdict

    def times_plus_equal(self, scalar, v2):
        """
        self += scalar * v2
        :param scalar: Double
        :param v2: FeatureVector
        :return: None
        """
        for key, value in v2.fdict.items():
            self.fdict[key] = scalar * value + self.fdict.get(key, 0)


    def dot_product(self, v2):
        """
        Computes the dot product between self and v2.  It is more efficient for v2 to be the smaller vector (fewer
        non-zero entries).
        :param v2: FeatureVector
        :return: Int
        """
        # print (self.fdict)
        # print (v2.fdict)
        # print ('-'*10)
        # raise
        retval = 0
        for key, value in v2.fdict.items():
            retval += value * self.fdict.get(key, 0)
        return retval

    def square(self):
        feat = FeatureVector({})
        for key, value in self.fdict.items():
            feat.fdict[key] = self.fdict.get(key,0)**2
        return feat
    
    def current_params(self,v2):
        feat = FeatureVector({})
        for key, value in v2.fdict.items():
            feat.fdict[key] = self.fdict.get(key,0)
        return feat
    
    def square_root(self):
        feat = FeatureVector({})
        for key, value in self.fdict.items():
            feat.fdict[key] = math.sqrt(self.fdict.get(key,0))
        return feat
    
    def divide(self, v2):
        feat = FeatureVector({})
        for key, value in self.fdict.items():
            if v2.fdict.get(key,0)==0:
                feat.fdict[key]=0
            else:
                feat.fdict[key]=self.fdict.get(key,0)/v2.fdict.get(key,0)
        return feat

    def write_to_file(self, filename):
        """
        Writes the feature vector to a file.
        :param filename: String
        :return: None
        """
        print('Writing to ' + filename)
        with open(filename, 'w', encoding='utf-8') as f:
            for key, value in self.fdict.items():
                f.write('{} {}\n'.format(key, value))


    def read_from_file(self, filename):
        """
        Reads a feature vector from a file.
        :param filename: String
        :return: None
        """
        self.fdict = {}
        with open(filename, 'r') as f:
            for line in f.readlines():
                txt = line.split()
                self.fdict[txt[0]] = float(txt[1])
