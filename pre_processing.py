import pandas as pd
import csv
import json


# pre_process the CSV data into the required fields only
# THIS WOULD BE EASIER TO COMBINE WITH OTHER DATA SETS with those
# relevant fields to expand the data set.
# data = '/Users/bryankamau/Documents/SPRING 2019/CS 5412/testingAzurePython/data/'
# data_file = data + "data job posts.csv"


# # pre_processed_data = {}
# # jobs_df_2 = pd.read_csv(data_file)

# f = open( data_file, 'rU' )
# # Change each fieldname to the appropriate field name. I know, so difficult.
# reader = csv.DictReader( f, fieldnames = ( "fn0","fn1","fn2","fn3","fn4","fn5","fn6","fn7","fn8","fn9","fn10","fn11","fn12","fn13","fn14","fn15","fn16","fn17","fn18","fn19","fn20","fn21","fn22","fn23" ))
# # Parse the CSV into JSON
# out = json.dumps( [ row for row in reader ] )
# print ("JSON parsed!")
# # Save the JSON
# f = open( 'data/parsed.json', 'w')
# f.write(out)
# print ("JSON saved!")

# # simplfying the json file
# pre_processed_data_json = []
# data = '/Users/bryankamau/Documents/SPRING 2019/CS 5412/testingAzurePython/data/parsed.json'

# with open(data) as json_file:
#     data_file = json.load(json_file)
#     for i in range(len(data_file)):
#         if i > 0:
#             # print(data_file[i]["fn0"])
#             entry = {}
#             entry["name"] = data_file[i]["fn3"]
#             entry["location"] = data_file[i]["fn10"]
#             entry["description"] = data_file[i]["fn11"]
#             entry["qualification"] = data_file[i]["fn13"]
#             # entry["title"]=data_file[i]["fn2"]
#             pre_processed_data_json.append(entry)

# # print(len(pre_processed_data_json))
# # with open("./data/data_file_2.json", "w") as write_file:
#     json.dump(pre_processed_data_json, write_file)

# company_data = []
# with open(data) as json_file:
#     loaded_data = json.load(json_file)

#     for entry in loaded_data:
#         # print(entry["name"])
#         # break
#         new_entry = {}
#         new_entry["name"] = entry["name"]
#         company_data.append(new_entry)

# with open("company_data_file.json", "w") as write_file:
#     json.dump(company_data, write_file)

# new_data = []
# with open(data) as json_file:
#     loaded_data = json.load(json_file)

#     # print(len(loaded_data))
#     count = 2000
#     for entry in loaded_data:
#         if count >= 17000 and count < 19002:

#             new_data.append(entry)
#         count += 1

# with open("azure_09.json", "w") as write_file:
#     json.dump(new_data, write_file)

# pre_processed_data = {}
# jobs_df_2 = pd.read_csv(data_file)

# convert the csv to a json file

# data_file = "/Users/bryankamau/Documents/SPRING 2019/CS 5412/testingAzurePython/data/Job Questionnaire (Responses) - Form Responses 1.csv"
# f = open(data_file, 'rU')
# # Change each fieldname to the appropriate field name. I know, so difficult.
# reader = csv.DictReader(f, fieldnames=("time", "f_name", "l_name", "college", "year", "major", "minor", "projects", "work experience", "values",
#                                        "roles", "locations", "skills", "languages", "hobbies"))
# # Parse the CSV into JSON
# out = json.dumps([row for row in reader])
# # Save the JSON
# f = open('data/questionnaire.json', 'w')
# f.write(out)
# data_file = "/Users/bryankamau/Documents/SPRING 2019/CS 5412/testingAzurePython/data/questionnaire.json"


# query_lst = []
# with open(data_file, "r") as read_file:
#     data = json.load(read_file)

#     count = 0
#     # query_lst = []
#     # parse the data to relevant columns
#     for entry in data:
#         if count > 0:
#             query = {}
#             query_data = entry["fn4"]+entry["fn5"]+entry["fn6"] + \
#                 entry["fn7"]+entry["fn8"]+entry["fn9"] + \
#                 entry["fn10"]+entry["fn11"]+entry["fn12"] + \
#                 entry["fn13"]+entry["fn14"]
#             query["fn1"] = query_data
#             print(query_data)
#             query_lst.append(query)
#         count += 1


# tokenize query
# print(len(query_lst))


# query expansion - using SVD in progress
# ML - sentiment analysis
# search process.

# 608

# feedback
