import pandas as pd
import re
import numpy as np
import math
import json
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import ShuffleSplit
from collections import Counter, defaultdict

# data = '/Users/bryankamau/Documents/SPRING 2019/CS 5412/testingAzurePython/data/'
# data_file = data + "data job posts.csv"
data = '/Users/bryankamau/Documents/SPRING 2019/CS 5412/testingAzurePython/data/'
data_file = data + "data_file.json"

companies = []
job_descs = []
locs = []
quals = []
num_rows = 0
with open(data_file) as json_file:
    data_vals = json.load(json_file)

    for entry in data_vals:
        companies.append(entry["name"])
        job_descs.append(entry["description"])
        locs.append(entry["location"])
        quals.append(entry["qualification"])

    num_rows = len(data_vals)

# reading the csv files
# path1 = "/Users/bryankamau/Documents/SPRING 2019/CS5412/testingAzurePython/data/Current_Job_Postings.csv"
# path2 = "/Users/bryankamau/Documents/SPRING 2019/CS5412/testingAzurePython/data/data job posts.csv"
# jobs_df = pd.read_csv(path1)
# jobs_df_2 = pd.read_csv(data_file)

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


# gets all words from the location, job description and requirements
# does this for all job entries
def all_words(jobs, i):
    lst = []
    entry = jobs[i]
    if entry != "nan":
        lst.append(tokenize(entry))
    return lst

# returns the unique words from all entries


def uniq_words(jobs):
    """Returns a list of unique words in the entire data set
    Params: {text: String}
    Returns: List
    """
    words = []
    for entry in jobs:
        if entry != "nan":
            lst = tokenize(entry)
        for word in lst:
            if not word in words:
                words.append(word)
    return words


# all unique words. Job Description
all_unique_words = uniq_words(job_descs)

# all unique locations
all_locs = uniq_words(locs)

# qualifications
all_quals = uniq_words(quals)

# all necessary features
# concat the job des, locations, quals
all_features = all_unique_words + all_locs + all_quals


# remove all words with less than 3 letters
# the do not contribute much to the feature vector
# might mess with locations
relevant_unique_words = list(
    filter(lambda x: len(x) > 3 and len(x) < 10, all_features))

# job_id document matrix
job_doc_matrix = np.zeros((num_rows, len(relevant_unique_words)))


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
for i in range(num_rows):
    tokens = all_words(job_descs, i)+all_words(locs, i)+all_words(quals, i)
    token_list = [item for sublist in tokens for item in sublist]
    counts = Counter(token_list)
    # fill up the job_doc_matrix
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
# p = classifier.predict(test_set)
# print("Accuracy: {:.2f}%".format(np.mean(p == class_test_flat_list)*100))


# TO DO NEXT
# Automate the path determination process
# Debug the third last part to clean it up
# Parse the questionnaires filled in and use that for prediction/matching
# Reduce the dimensionality of the data as not all the features are important - This would make the model
# efficient, faster, more accurate and cleaner. More relevant to fit it in the start-up context
#

###### PREDICTION TEST ####
query_file = "/Users/bryankamau/Documents/SPRING 2019/CS 5412/testingAzurePython/data/questionnaire.json"

# name and role
roles = {}
roles_list_names = []

query_lst = []
with open(query_file, "r") as read_file:
    data = json.load(read_file)

    count = 0
    # query_lst = []
    # parse the data to relevant columns
    for entry in data:
        if count > 0:
            query = {}
            query_data = entry["fn4"]+entry["fn5"]+entry["fn6"] + \
                entry["fn7"]+entry["fn8"]+entry["fn9"] + \
                entry["fn10"]+entry["fn11"]+entry["fn12"] + \
                entry["fn13"]+entry["fn14"]
            query["fn1"] = query_data
            query_lst.append(query)

            ## extract name and roles
            name = entry["fn1"] + ' ' + entry["fn2"]
            roles[name] = entry["fn10"]
            roles_list_names.append(name)
        count += 1

# tokenize query
tokens = []
for item in query_lst:
    for key, val in item.items():
        tokens.append(tokenize(val))

query_features = np.zeros((len(tokens), len(relevant_unique_words)))

query_len = len(tokens)
i = 0
for entry in tokens:
    count = Counter(entry)
    for key, val in count.items():
        if key in inverted_index and i <= query_len:
            query_features[i][(inverted_index[key])] = val
    i += 1

p = classifier.predict(query_features)
predicted_comp = []
for i in p:
    predicted_comp.append(list(uniq_companies)[i])


## create a dictionary of company name : list of all titles
## get the 10th entry for roles from the input from the webform
## binary tokenize it and do a binary search on the list
## determine criteria to break ties or when there is nothing returned
## Levenshtein comparison - maybe
## Company and titles data

#Save the company and titles in the dictionary
comp_titles_data = '/Users/bryankamau/Documents/SPRING 2019/CS 5412/testingAzurePython/data/data_file_2.json'
comp_to_titles_dict = {}

with open(comp_titles_data, "r") as read_file:
    comp_title = json.load(read_file)

    for entry in comp_title:
        if not (entry["name"] in comp_to_titles_dict):
            title_list = []
            a = entry["title"]
            title_list.append(a)
            comp_to_titles_dict[(entry["name"])] = title_list
        else:
            a = entry["name"]
            (comp_to_titles_dict[(entry["name"])]).append(a)



# get the roles input
# print(roles)
# use MAP and FILTER

## binary search for matching
## could be improved alot

#tokenize the roles clients are interested in
interested_roles = []
for key in roles:
    roles_token_lst = []
    roles_token_lst = tokenize(roles[key])
    interested_roles.append(list(set(roles_token_lst)))

#tokenize the company titles and store them in a dictionary of
#company and titles tokenized (list of lists)
company_titles = {}
for comp in predicted_comp:
    comp_tits = []
    titles = comp_to_titles_dict[comp]
    for title in titles:
        title_tokens_lst = []
        title_tokens_lst = tokenize(title)
        comp_tits.append(list(set(title_tokens_lst)))
    company_titles[comp]=comp_tits

## find roles from returned companies 
##
def find_title(roles_set, title_list):
    """returns the index of the title
    that a client matches with"""
    title_indx = []
    for i in range(len(title_list)):
        inters = roles_set.intersection(set(title_list[i]))
        if not (list(inters) == []):
            title_indx.append(i)
    return title_indx

### return the client name, company name and title fit
### return some clean understandable output to user
### Name: {company, title}
output = {}
for i in range(len(predicted_comp)):
    titles = find_title(set(interested_roles[i]),company_titles[(predicted_comp[i])])
    if titles == []:
        comp = {}
        comp["name"] = "jobless"
        comp["title"] = "start your own"
        output[(roles_list_names[i])] = comp
    else:
        comp = {}
        t_list = comp_to_titles_dict[(predicted_comp[i])]
        comp["name"] = predicted_comp[i]
        comp["title"] = t_list[(titles[0])]
        output[(roles_list_names[i])] = comp

print(output)





    
    

