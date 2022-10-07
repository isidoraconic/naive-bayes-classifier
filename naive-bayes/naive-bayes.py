import os
import random
import math

ham_dict = dict()
spam_dict = dict()

# get list of all the files in spam and ham, respectively
ham_email_files = os.listdir("../HamSpam/ham")
spam_email_files = os.listdir("../HamSpam/spam")

# getting total number of emails in each category (ham vs. spam)
total_ham_email_count = len(ham_email_files)
total_email_spam_count = len(spam_email_files)

# creating basic dictionaries with just the word counts (no log, smoothing, or probabilities)
# creating the ham dictionary
for ham_filename in ham_email_files:
    with open(os.path.join("../HamSpam/ham", ham_filename), 'r') as h:
        ham_file_lines = h.readlines()
        for ham_line in ham_file_lines:
            ham_line = ham_line.strip()
            if ham_line in ham_dict:
                ham_dict[ham_line] = ham_dict[ham_line] + 1
            else:
                ham_dict[ham_line] = 1
# creating the spam dictionary
for spam_filename in spam_email_files:
    with open(os.path.join("../HamSpam/spam", spam_filename), 'r') as s:
        spam_file_lines = s.readlines()
        for spam_line in spam_file_lines:
            spam_line = spam_line.strip()
            if spam_line in spam_dict:
                spam_dict[spam_line] = spam_dict[spam_line] + 1
            else:
                spam_dict[spam_line] = 1

# wordcount of unique words in ham and spam training sets
ham_train_wordcount = len(ham_dict)
spam_train_wordcount = len(spam_dict)

# n is the number of unique words in our training sets (both ham and spam) --> is this right?
n = ham_train_wordcount + spam_train_wordcount

# n_spam is the total number of words in the spam emails (not unique)
n_spam = 0
for value in spam_dict.values():
    n_spam = n_spam + value

# n_ham is the total number of words in the ham emails (not unique)
n_ham = 0
for value in ham_dict.values():
    n_ham = n_ham + value

# hyperparams for smoothing
alpha = 0.005

# v is the total words in the english language, this is an estimate
v = 170000

# converting the dictionary entries into probabilities, and take the log of each
# note that these are SEEN words (that we got from the training set)
# the formula for prob P(word|HAM) or P(word|SPAM) = word_count/(n+v) where n is n_ham or n_spam
# ham probability (log) dictionary first, making an implicit copy
ham_log_prob = ham_dict.copy()
for key, value in ham_log_prob.items():
    ham_log_prob[key] = math.log(value/(n_ham+v))

# spam log probability dictionary
spam_log_prob = spam_dict.copy()
for key, value in spam_log_prob.items():
    spam_log_prob[key] = math.log(value/(n_spam+v))

# unseen ham log probability (probability of an unseen word being ham)
unseen_ham_log_prob = math.log(alpha/(n_ham+(alpha*v)))

# unseen spam log probability (probability of an unseen word being spam)
unseen_spam_log_prob = math.log(alpha/(n_spam+(alpha*v)))

# classifying the testing emails as ham or spam (they are classified so we know that they should be)
# will check prob of each word being ham or spam, and then multiply all those (independent words)
# if prob of the overall email being ham > prob of spam, then email is ham and vice versa
# if the word is unseen, we use the above unseen word probabilities
# the input is one of the testing file names, so we are pulling from that folder by default

# classifying the emails in the given test file
test_files = os.listdir("../HamSpam/test")
test_email_classification = dict()
for test_email in test_files:
    ham_prob = 0
    spam_prob = 0
    with open(os.path.join("../HamSpam/test", test_email), 'r') as te:
        lines = te.readlines()
        for line in lines:
            line = line.strip()
            # calculate ham log prob of this word
            if line in ham_log_prob:
                ham_prob = ham_prob + ham_log_prob[line]
            else:
                ham_prob = ham_prob + unseen_ham_log_prob
            # calculate spam log prob of the same word
            if line in spam_log_prob:
                spam_prob = spam_prob + spam_log_prob[line]
            else:
                spam_prob = spam_prob + unseen_spam_log_prob

    # check which prob is greater, ham or spam, and then that is the classification for this email
    test_email = test_email.split(".", 1)[0]
    if ham_prob >= spam_prob:
        test_email_classification[test_email] = "HAM"
    else:
        test_email_classification[test_email] = "SPAM"


# validating to see which emails in the test folder were classified correctly based on the truthfile
# P = condition positive is a SPAM email (R) = # of file names in the truthfile
# R = condition negative is a HAM email = # of test files (100) - # of files in the truthfile (R)
# TP = SPAM email that got classified as SPAM
# FN = SPAM email that got classified as HAM (incorrect)
# FP = HAM email that got classified as SPAM (incorrect)
# TN = HAM email that got classified as HAM
# count = sanity check to make sure we count all 100 test emails
P = 0
R = 0
TP = 0
FN = 0
FP = 0
TN = 0
count = 0
with open("../HamSpam/truthfile", 'r') as tf:
    spam_files = tf.readlines()
    P = len(spam_files)
    # removing all newlines from the elements in the list
    spam_files = list(map(lambda s: s.strip(), spam_files))
    for key in test_email_classification.keys():
        count += 1
        # if the key (file number) is in the spam_files list, it means that this email should have been classified as SPAM
        if key in spam_files:
            if test_email_classification[key] == "SPAM":
                TP += 1
            else:
                FN += 1
        # if the key (file num) is not in the spam_files list, then it means it should be classified as HAM
        else:
            if test_email_classification[key] == "HAM":
                TN += 1
            else:
                FP += 1
R = len(os.listdir("../HamSpam/test")) - P

# calculating f-score
# f-score = TP/(TP + 0.5(FP+FN))
f_score = TP/(TP + 0.5*(FP+FN))

# calculating precision --> quantifies num of positive class predictions that actually belong to the positive class
# ie how many HAMS actually got classified as HAM
# precision = TP/(TP+FP)
precision = TP/(TP+FP)

# calculating recall --> quantifies num of positive class predictions made out of all positive examples in the dataset
# recall = TP/(TP+FN)
recall = TP/(TP+FN)

print("P = ", P)
print("R = ", R)
print("TP (SPAM classified as SPAM) = ", TP)
print("FN (SPAM classified as HAM) = ", FN)
print("FP (HAM classified as SPAM) = ", FP)
print("TN (HAM classified as HAM) = ", TN)
print("f-score = ", f_score)
print("precision = ", precision)
print("recall = ", recall)
print("The TEST emails were classified as follows: ")
test_key_list = sorted(list(map(int, test_email_classification.keys())))
for key in test_key_list:
    print(key, " = ", test_email_classification[str(key)])







