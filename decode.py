
def decode(input_length, tagset, score):
    """
    Compute the highest scoring sequence according to the scoring function.
    :param input_length: int. number of tokens in the input including <START> and <STOP>
    :param tagset: Array of strings, which are the possible tags.  Does not have <START>, <STOP>
    :param score: function from current_tag (string), previous_tag (string), i (int) to the score.  i=0 points to
        <START> and i=1 points to the first token. i=input_length-1 points to <STOP>
    :return: Array strings of length input_length, which is the highest scoring tag sequence including <START> and <STOP>
    """
    # Look at the function compute_score for an example of how the tag sequence should be scored
    n_tags, n_words = len(tagset), input_length-2

    best_path_scores = [[float('-inf')]*n_words for i in range(n_tags)]
    best_path_pointers = [[None]*n_words for i in range(n_tags)]

    #initialization
    for i in range(n_tags):
        best_path_scores[i][0] = score(tagset[i], "<START>", 1) # best score for start word given each tag
    # print ([best_path_scores[i][0] for i in range(n_tags)])
    # print (best_path_scores)

    for i in range(1,n_words):

        for j in range(n_tags):
            best_score = float('-inf')
            best_path  = None

            cur_tag = tagset[j] #for each tag (run)

            for k in range(n_tags):

                prev_tag = tagset[k] #cur tag
                cur_score = best_path_scores[k][i-1] + score(cur_tag, prev_tag, i+1) 
                if cur_score > best_score:
                    best_score = cur_score
                    best_path = k

            best_path_scores[j][i] = best_score
            best_path_pointers[j][i] = best_path
    
    #backtrack towards the best path
    best_path = [None]*n_words
    best_score_last_word = float('-inf')
    for i in range(n_tags):
        cur_score = best_path_scores[i][n_words-1]
        if cur_score > best_score_last_word:
            best_score_last_word = cur_score
            best_path[n_words-1] = i
        
    for i in reversed(range(1,n_words)):
        best_path[i-1] = best_path_pointers[best_path[i]][i]
    
    best_path =  (['<START>']+[tagset[i] for i in best_path]+['<STOP>'])
    # print ('bp', best_path)

    return best_path
