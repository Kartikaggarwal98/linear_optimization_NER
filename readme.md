# Structured Prediction using Linear Models

Named Entity Recognition on [CoNLL 2003 Shared Task](https://www.aclweb.org/anthology/W03-0419) using Linear models such as Perceptron using SGD, Adagrad and SVM  with hamming loss + l2 regularization. Implementation contains BIO tagging for four kinds of entities: people, locations, organizations, and names of other miscellaneous entities.

Run the code using: `python main.py`

Features are manually created in `features.py`. The following features are considered:

1. Current word wi. Example: Wi=France+Ti=I-LOC 1.0

2. Previous tag ti−1. Example: Ti-1=<START>+Ti=I-LOC 1.0

3. Lowercased word oi. Example: Oi=france+Ti=I-LOC 1.0

4. Current POS tag pi. Example: Pi=NNP+Ti=I-LOC 1.0

5. Shape of current word si. Just replace all letters with a or A depending on capitalization, and replace digits with d. Example: Si=Aaaaaa+Ti=I-LOC 1.0

6. The features 1-4 for the previous word, and the next word.

7. Features 1, 3 and 4 conjoined with the previous tag. 

8. Length k prefix for the current word, for k = 1, 2, 3, 4. Examples: PREi=Fr+Ti=I-LOC 1.0, PREi=Fra+Ti=I-LOC 1.0

9. Is the current word in the gazetteer for the current tag? Example: GAZi=True+Ti=I-LOC 1.0

10. Does the current word start with a capital letter? Example: CAPi=True+Ti=I-LOC 1.0

11. Position of the current word (indexed starting from 1). Example: POSi=1+Ti=I-LOC 1.0

Viterbi Decoder is used for decoding all algorithms. 

## Structured Perceptron 

### SSGD

Training with the structured perceptron loss function is done using stochastic subgradient descent (SSGD) and early stopping for this model. Because the perceptron loss function scales linearly with the weight vector, neither the regularizer nor the step size have a mean- ingful effect on the perceptron loss for a linear model. For this reason, use stepsize 1 and do not include a regularizer.

See `sgd_optimizer` function in `optimizers.py` for details.

<img src="https://latex.codecogs.com/svg.latex?\Large&space;w = w - \alpha g(x,y)" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> 


### Adagrad

Adagrad is implemented the same as SSGD, but with a modified equation for updating the parameters. Like SSGD, each gradient update is called a time step. Let t count the number of gradient updates that have been performed (and does not reset after each epoch.) Let gt,i be the ith component of the gradient at time step t. We keep a running total of the sum of squares of all the components of all the previous gradients: st,i. This includes gradients from all previous epochs. At time step t, the ith parameter θt,i is updated like so:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta_{t,i}=\theta_{t-1,i}-\frac{\alpha}{\sqrt{s_{t,i}}}g_{t,i}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> 

where, 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;s_{t,i} = \sum^t_{T=1} g^2_{T,i}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />

See `adagrad_optimizer` function in `optimizers.py` for details.

## Structured SVM

Cost augmented decoding for this cost function is implemented as another feature during Viterbi decoding. Hamming Loss is used as the cost function for decoding. L2 regularizer is used and stepsize is tuned.



<img src="https://latex.codecogs.com/svg.latex?\Large&space;L(w,D)=\sum_{i=1}^N((\text{max}\;w f(x_i,y_i') + cost(y_i,y')) - w f(x_i,y_i)) + \frac{\lambda}{2}|w|^2" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> 

See `svm_gradient` function in `train.py` for details.

Hence,
<img src="https://latex.codecogs.com/svg.latex?\Large&space;y'=\text{argmax}_{y'\in Y} \; w f(x_i,y_i') + cost(y_i,y'))" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> 

See `svm_optimizer` function in `optimizers.py` for details.

<img src="https://latex.codecogs.com/svg.latex?\Large&space;w = w - \alpha g(x,y) - \alpha\lambda w" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> 



## Structured SVM with modified cost function

An important property of the SVM loss is you can penalize some errors more than others during training. Modify the cost function to penalize mistakes three times more (penalty of 30) if the gold standard has a tag that is not O but the candidate tag is O.
