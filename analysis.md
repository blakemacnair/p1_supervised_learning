# Final Analysis Contents

1. A description of your classification problems, and why you feel that they are interesting. 
Think hard about this. 
    1. To be at all interesting the problems should be non-trivial on the one hand, but capable of admitting comparisons 
       and analysis of the various algorithms on the other.

2. The **training and testing error rates** you obtained running the various learning algorithms on your problems. 
    1. At the very least you should include graphs that show performance on both training and test data as a function 
       of:
        1. **training size** (note that this implies that you need to design a classification problem that has more than a 
      trivial amount of data) 
        2. and--for the algorithms that are iterative--**training times/iterations**. 
       
    2. Both of these kinds of graphs are referred to as learning curves, BTW.


3. Analyses of your results.
    1. Why did you get the results you did? 
    2. Compare and contrast the different algorithms.
    3. What sort of changes might you make to each of those algorithms to improve performance? 
    4. How fast were they in terms of:
        1. wall clock time? 
        2. Iterations? 
    5. Would cross validation help (and if it would, why didn't you implement it?)? 
    6. How much performance was due to the problems you chose? 
    7. How about the values you choose for learning rates, stopping criteria, pruning methods, and so forth? 
        1. (and why doesn't your analysis show results for the different values you chose? 
           Please do look at more than one. 
           And please make sure you understand it, it only counts if the results are meaningful)
    8. Which algorithm performed best? 
        1. How do you define best?
    9. Be creative and think of as many questions you can, and as many answers as you can.


# Classification Problems

- For the purposes of this assignment, a classification problem is just a set of training examples and a set of test 
  examples. 
- I don't care where you get the data. 
  You can download some, take some from your own research, or make some up on your own.
- **You'll have to explain why they are interesting, use them in later assignments, and come to really care about them.**

## 1. Credit Card Fraud

[Data source here](https://www.kaggle.com/mlg-ulb/creditcardfraud)
``` bash
kaggle datasets download -d mlg-ulb/creditcardfraud
```


## 2. Heart Disease Cleveland UCI

[Data source here](https://www.kaggle.com/cherngs/heart-disease-cleveland-uci)
``` bash
kaggle datasets download -d cherngs/heart-disease-cleveland-uci
```


# Training Algorithms

## 1. Decision Tree

- Be sure to use some form of pruning.
- You are not required to use information gain (for example, there is something called the GINI index that is sometimes 
  used) to split attributes, but you should describe whatever it is that you do use.
  
## 2. Neural Network

- Choose any kind and complexity of NN as you like, just be able to describe it!

## 3. Boosting

- Implement or steal a boosted version of your decision trees.

## 4. SVM

- This should be done in such a way that you can swap out kernel functions. 
  I'd like to see at least two.
  
## 5. KNN

- Use different values of k.