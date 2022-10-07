# Naive-Bayes Assignment

This is an implementation of a Naive-Bayes Classifier, using the data in the HamSpam folder. I first implemented my Naive-Bayes model, and then further tried to optimize it by removing stop-words (see below). 

This classifier aims to classify emails as Ham (normal email, not spam) or Spam (a spam email) - binary classification.

## (1) Naive-Bayes Classifier

The first iteration of my classifier implements a Naive Bayes Classifier. It considers each word contained in an email to be independent opf the other (which in real life is a big assumption but we maintain it for this model). In this case, a Naive Bayes model predicts/estimates the probability of a message being ham or spam based on the joint probabilistic distributions of the appreance of certain words in the email. I further optimized the hyperparameters, `v` (total vocabulary words; total number of words in the english language) as well as `alpha` (used for smoothing). I found that the best values for these paramters were as follows:

​	`v = 170000`  

​	`alpha = 0.005` 

### Results/Metrics: 

I set `P` (condition positive) as the SPAM emails, and `R` (condition negative) as the HAM emails. The results for this model (classification on the 100 emails in the test set) were as follows:

`P = 37`

`R = 63`

`TP (SPAM classified as SPAM) = 36`

`FN (SPAM classified as HAM) =  1`

`FP (HAM classified as SPAM) =  13`

`TN (HAM classified as HAM) =  50`

`f-score =  0.8372093023255814`

`precision =  0.7346938775510204`

`recall =  0.972972972972973`

## (2) Naive-Bayes Classifier with removal of Stop-Words

In this second attempt, I removed stop-words (as defined in [Ranks NL](https://www.ranks.nl/stopwords)). In this case, when creating the dictionairies for my model, if a word came up that was in this list, I didn't include it. Similarly, when classifying a word/email, if the word was in the list of stop words (found in the `stop-words.txt` file), I did not include it in the probability calculation.

Contrary to what I anticipated, this did not improve the performance of my classifier. Initially, I believed that it would reduce noise that would come from using words that don't hold any semantic importance as part of the classification (in in turn improve classification). I speculate that the performance of this classifier might have been worse due to a reduced size of training set (now that stop-words are removed), and that perhaps in this case, these stop-words did in fact hold some importance. Further, seeing as I used a pre-defined set of stop-words (not onces that I myself chose), some of these words may have actually been relevant in the classification.

The same hyperparameters as above were used.

### Results/Metrics:

Same as in the previous classifier, I set `P` (condition positive) as the SPAM emails, and `R` (condition negative) as the HAM emails. The results for this model without stop-words (classification on the 100 emails in the test set) were as follows:

`P = 37`

`R = 63`

`TP (SPAM classified as SPAM) = 35`

`FN (SPAM classified as HAM) =  2`

`FP (HAM classified as SPAM) =  13`

`TN (HAM classified as HAM) =  50`

`f-score =  0.8235294117647058`

`precision =  0.7291666666666666`

`recall =  0.9459459459459459`









