import pandas as pd
import re
import numpy as np
import math
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import ShuffleSplit
from collections import Counter, defaultdict

#
data = '/Users/bryankamau/Documents/SPRING 2019/CS5412/testingAzurePython/data/'
data_file = data + "data job posts.csv"


# reading the csv files
# path1 = "/Users/bryankamau/Documents/SPRING 2019/CS5412/testingAzurePython/data/Current_Job_Postings.csv"
# path2 = "/Users/bryankamau/Documents/SPRING 2019/CS5412/testingAzurePython/data/data job posts.csv"
# jobs_df = pd.read_csv(path1)
jobs_df_2 = pd.read_csv(data_file)

# tokenize job descriptions


def tokenize(text):
    """Returns a list of words that make up the text.
    Note: for simplicity, lowercase everything.
    Requirement: Use Regex to satisfy this function
    Params: {text: String}
    Returns: List
    """
    regex = re.compile(r'[a-z]+')
    return regex.findall(text.lower())


# work with numpy arrays instead
jobs_df_2 = jobs_df_2.as_matrix()
r, c = np.shape(jobs_df_2)


# gets all words from the location, job description and requirements
# does this for all job entries
def all_words(jobs, row_num, col_num):
    job_desc = str(jobs[row_num, col_num])
    lst = []
    if job_desc != "nan":
        lst = tokenize(job_desc)
    return lst

# returns the unique words from all entries


def uniq_words(jobs, col_num):
    """Returns a list of unique words in the entire data set
    Params: {text: String, col_num: int}
    Returns: List
    """
    words = []
    r, c = np.shape(jobs)
    for i in range(r):
        # print(jobs_df_2[i, 11])
        job_desc = str(jobs[i, col_num])
        if job_desc != "nan":
            lst = tokenize(job_desc)
        for word in lst:
            if not word in words:
                words.append(word)
    return words


# all unique words. Job Description
all_unique_words = uniq_words(jobs_df_2, 11)

# all unique locations
all_locs = uniq_words(jobs_df_2, 10)

# qualifications
all_quals = uniq_words(jobs_df_2, 13)

# all necessary features
# concat the job des, locations, quals
all_features = all_unique_words + all_locs + all_quals


# remove all words with less than 3 letters
# the do not contribute much to the feature vector
# might mess with locations
relevant_unique_words = list(
    filter(lambda x: len(x) > 3 and len(x) < 10, all_features))

# job_id document matrix
job_doc_matrix = np.zeros((r, len(relevant_unique_words)))


# labels are the companies matched with
companies = []
for i in range(r):
    companies.append(str(jobs_df_2[i, 3]))

# set of all unique companies
uniq_companies = set(companies)

# returns the index of a company
# company inverted index. More efficient

comp_inv_idx = defaultdict(list)
for idx, comp in enumerate(uniq_companies):
    if idx not in comp_inv_idx[comp]:
        comp_inv_idx[comp].append(idx)

# creates a integer indexing (list) for companies
# uses the company inverted index
company_indices = []
for comp in companies:
    company_indices.append(comp_inv_idx[comp])

# builld an inverted index for efficiency
inverted_index = defaultdict(list)
for idx, wrd in enumerate(relevant_unique_words):
    if idx not in inverted_index[wrd]:
        inverted_index[wrd].append(idx)


# fill up the job_document matrix (company document matrix)
# initial job_document matrix is all zeros
for i in range(r):
    tokens = all_words(jobs_df_2, i, 11)+all_words(jobs_df_2,
                                                   i, 10)+all_words(jobs_df_2, i, 13)
    counts = Counter(tokens)
    for key, val in counts.items():
        if key in all_features:
            job_doc_matrix[i][(inverted_index[key])] = val

# Machine Learning Aspect
# Naive Bayes Model
shuffle_split = ShuffleSplit(len(companies), test_size=0.2, random_state=0)
train_idx, test_idx = next(iter(shuffle_split))

train_set = job_doc_matrix[train_idx]
test_set = job_doc_matrix[test_idx]

# could be made more efficient
class_train = [company_indices[i] for i in train_idx]
class_train_flat_list = [idx for sublist in class_train for idx in sublist]
class_test = [company_indices[i] for i in test_idx]
class_test_flat_list = [idx for sublist in class_test for idx in sublist]

# MNNB classifier
classifier = MultinomialNB()
classifier.fit(train_set, class_train_flat_list)

# predict
p = classifier.predict(test_set)
print("Accuracy: {:.2f}%".format(np.mean(p == class_test_flat_list)*100))


# TO DO NEXT
# Automate the path determination process
# Debug the third last part to clean it up
# Parse the questionnaires filled in and use that for prediction/matching
# Reduce the dimensionality of the data as not all the features are important - This would make the model
# efficient, faster, more accurate and cleaner. More relevant to fit it in the start-up context
# Validate/regular expressions for the Google Form for consistency
