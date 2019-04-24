import requests
from pprint import pprint
from IPython.display import HTML
import azure.cosmos.cosmos_client as cosmos_client
# Need to install requests, azure.cosmos

# Config for Cosmos DB
config = {
    'ENDPOINT': 'https://trial-db.documents.azure.com:443/',
    'PRIMARYKEY': 'jifdRy7m540KAbn7jVq2DWVbRdWQ5G1dB32wpeAw0JAQd84pb8ZCcXoam6EqGD2yptx0MhKXI91Mxf0ObOTJQg==',
    'DATABASE': 'EHhKAA==',
    'CONTAINER': 'EHhKAIL5N7k='
}

# Initialize the Cosmos client
client = cosmos_client.CosmosClient(url_connection=config['ENDPOINT'], auth={
                                    'masterKey': config['PRIMARYKEY']})

database_link = 'dbs/' + config['DATABASE']
collection_link = database_link + '/colls/' + config['CONTAINER']
collection = client.ReadContainer(collection_link)

query = {'query': 'SELECT * FROM col1'}
options = {}
options['enableCrossPartitionQuery'] = True
options['maxItemCount'] = 10

# Key for text analytics API
subscription_key = 'b911917a4b33431eb7659a44d58300a2'
assert subscription_key

text_analytics_base_url = "https://centralus.api.cognitive.microsoft.com/"
key_phrase_api_url = text_analytics_base_url + "text/analytics/v2.0/keyPhrases"

# Make query to Cosmos DB
result_iterable = list(client.QueryItems(collection_link, query, options))[0:100] #limit to 10 results

# Format data for the text analytics API
def format_json(result, id):
    return {'id': result['id'], 'text': result['description'] + result['qualification']}
docs = []
for x in range(len(result_iterable)):
    docs.append(format_json(result_iterable[x], x))
documents = {'documents' : docs}
headers   = {'Ocp-Apim-Subscription-Key': subscription_key}
# Make request to text analytics API
response  = requests.post(key_phrase_api_url, headers=headers, json=documents)
key_phrases = response.json()

# Update the documents in Cosmos DB
for i in range(len(result_iterable)):
    result_iterable[i]['keyPhrases'] = key_phrases['documents'][i]['keyPhrases']
    client.ReplaceItem(result_iterable[i]['_self'], result_iterable[i])
