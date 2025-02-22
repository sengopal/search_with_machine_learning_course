# A simple client for querying driven by user input on the command line.  Has hooks for the various
# weeks (e.g. query understanding).  See the main section at the bottom of the file
from opensearchpy import OpenSearch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import json
import os
from getpass import getpass
from urllib.parse import urljoin
import pandas as pd
import fileinput
import logging
import numpy as np
import fasttext
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s')

cat_model = fasttext.load_model("../week3/cat_classifier_v5.bin")

model = SentenceTransformer('all-MiniLM-L6-v2')
MAX_RETRIEVAL_SIZE = 10000

# expects clicks and impressions to be in the row
def create_prior_queries_from_group(
        click_group):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    if click_group is not None:
        for item in click_group.itertuples():
            try:
                click_prior_query += "%s^%.3f  " % (item.doc_id, item.clicks / item.num_impressions)

            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# expects clicks from the raw click logs, so value_counts() are being passed in
def create_prior_queries(doc_ids, doc_id_weights,
                         query_times_seen):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    click_prior_map = ""  # looks like: '1065813':100, '8371111':809
    if doc_ids is not None and doc_id_weights is not None:
        for idx, doc in enumerate(doc_ids):
            try:
                wgt = doc_id_weights[doc]  # This should be the number of clicks or whatever
                click_prior_query += "%s^%.3f  " % (doc, wgt / query_times_seen)
            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


def create_vector_query(user_query, size=10, source=None,use_cat_filter=False):
    query_embedding = model.encode([user_query])
    query_obj = {
        "size": size,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding[0],
                    "k": size
                }
            }
        }
    }

    if use_cat_filter:
        cats_list = get_cat_preds(user_query)
        if len(cats_list) > 0:
            cats_filter = {
                "terms": {
                    "categoryPathIds": cats_list
                }
            }
            query_obj["post_filter"] = cats_filter

    if source is not None:  # otherwise use the default and retrieve all source
        query_obj["_source"] = source

    return query_obj


def get_cat_preds(user_query):
    pred = cat_model.predict(user_query, k=20)
    scores = pred[1]
    # pred_cats = max(1, np.argmin(scores > 0.1)) ## use only cats with score more thatn 0.25
    # try summing the scores of the top categories returned by the classifier until their sum is above the threshold
    # pred_cats = 5
    num_cats = np.argmin(np.cumsum(scores) < 0.5)
    pred_cats = num_cats if num_cats > 1 else 1
    category = pred[0][0:pred_cats]
    cats_list = [cat.replace('__label__', '') for cat in category]
    print(f"pred cat list: {cats_list}")
    return cats_list


# Hardcoded query here.  Better to use search templates or other query config.
def create_query(user_query, click_prior_query, filters, sort="_score", sortDir="desc", size=10, source=None,use_synonyms=False,use_cat_filter=False):
    cats_list = get_cat_preds(user_query)

    query_obj = {
        'size': size,
        "sort": [
            {sort: {"order": sortDir}}
        ],
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [

                        ],
                        "should": [  #
                            {
                                "match": {
                                    "name": {
                                        "query": user_query,
                                        "fuzziness": "1",
                                        "prefix_length": 2,
                                        # short words are often acronyms or usually not misspelled, so don't edit
                                        "boost": 0.01,
                                        # "auto_generate_synonyms_phrase_query": "true"
                                    }
                                }
                            },
                            {
                                "match_phrase": {  # near exact phrase match
                                    "name.hyphens": {
                                        "query": user_query,
                                        "slop": 1,
                                        "boost": 50
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": user_query,
                                    "type": "phrase",
                                    "slop": "6",
                                    "minimum_should_match": "2<75%",
                                    "fields": [f"name^10", "name.hyphens^10", "shortDescription^5",
                                               "longDescription^5", "department^0.5", "sku", "manufacturer", "features",
                                               "categoryPath"]
                                }
                            },
                            {
                                "terms": {
                                    # Lots of SKUs in the query logs, boost by it, split on whitespace so we get a list
                                    "sku": user_query.split(),
                                    "boost": 50.0
                                }
                            },
                            {  # lots of products have hyphens in them or other weird casing things like iPad
                                "match": {
                                    "name.hyphens": {
                                        "query": user_query,
                                        "operator": "OR",
                                        "minimum_should_match": "2<75%"
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1,
                        "filter": filters  #
                    }
                },
                "boost_mode": "multiply",  # how _score and functions are combined
                "score_mode": "sum",  # how functions are combined
                "functions": [
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankShortTerm"
                            }
                        },
                        "gauss": {
                            "salesRankShortTerm": {
                                "origin": "1.0",
                                "scale": "100"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankMediumTerm"
                            }
                        },
                        "gauss": {
                            "salesRankMediumTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankLongTerm"
                            }
                        },
                        "gauss": {
                            "salesRankLongTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "script_score": {
                            "script": "0.0001"
                        }
                    }
                ]

            }
        }
    }

    if use_cat_filter and len(cats_list) > 0:
        cats_filter = {
            "terms": {
                "categoryPathIds": cats_list
            }
        }
        query_obj["query"]["function_score"]["query"]["bool"]["must"].append(cats_filter)

    if use_synonyms:
        synonym_match = {
                        "match": {
                            "name.synonyms": {
                                "query": user_query,
                                "operator": "OR"
                            }
                        }
                    }

        query_obj["query"]["function_score"]["query"]["bool"]["should"].append(synonym_match)

    if click_prior_query is not None and click_prior_query != "":
        query_obj["query"]["function_score"]["query"]["bool"]["should"].append({
            "query_string": {
                # This may feel like cheating, but it's really not, esp. in ecommerce where you have all this prior data,  You just can't let the test clicks leak in, which is why we split on date
                "query": click_prior_query,
                "fields": ["_id"]
            }
        })
    if user_query == "*" or user_query == "#":
        # replace the bool
        try:
            query_obj["query"] = {"match_all": {}}
        except:
            print("Couldn't replace query for *")
    if source is not None:  # otherwise use the default and retrieve all source
        query_obj["_source"] = source
    return query_obj


def search(client, user_query, index="bbuy_products", sort="_score", sortDir="desc",use_synonyms=False,use_cat_filter=False,use_vector=False):
    #### W3: classify the query
    #### W3: create filters and boosts
    # Note: you may also want to modify the `create_query` method above
    # Current result size expected
    size = 10

    if use_vector:
        query_obj = create_vector_query(user_query,size=size, use_cat_filter=use_cat_filter)
    else:
        query_obj = create_query(user_query, click_prior_query=None, filters=None, sort=sort, sortDir=sortDir, source=["name", "shortDescription"], use_synonyms=use_synonyms, use_cat_filter=use_cat_filter)
    # logging.info(query_obj)
    # print(json.dumps(query_obj, indent=2))
    response = client.search(query_obj, index=index)

    # Filtering Vector Search Results and retrying with a larger size
    response_cnt = len(response['hits']['hits'])
    if response_cnt == 0:
        while response_cnt == 0 and size < MAX_RETRIEVAL_SIZE:
            size = size * 2
            query_obj = create_vector_query(user_query, size=size, use_cat_filter=use_cat_filter)
            response = client.search(query_obj, index=index)
            response_cnt = len(response['hits']['hits'])

    if response and response['hits']['hits'] and len(response['hits']['hits']) > 0:
        hits = response['hits']['hits']
        # print(json.dumps(response, indent=2))
        print(f"result_count:{response['hits']['total']['value']}")
        for hit in hits:
            print(f"{hit['_source']['name'][0]}")
    else:
        print(f"response_len:{len(response['hits']['hits'])}")

if __name__ == "__main__":
    host = 'localhost'
    port = 9200
    auth = ('admin', 'admin')  # For testing only. Don't store credentials in code.
    parser = argparse.ArgumentParser(description='Build LTR.')
    general = parser.add_argument_group("general")
    general.add_argument("-i", '--index', default="bbuy_products",
                         help='The name of the main index to search')
    general.add_argument("-s", '--host', default="localhost",
                         help='The OpenSearch host name')
    general.add_argument("-p", '--port', type=int, default=9200,
                         help='The OpenSearch port')
    general.add_argument('--user',
                         help='The OpenSearch admin.  If this is set, the program will prompt for password too. If not set, use default of admin/admin')
    general.add_argument('--synonyms', default="False",
                         help='If set as True, uses the “name.synonyms” field instead of “name”. All other values will be ignored')
    general.add_argument('--catFilter', default="False",
                         help='If set as True, uses the Category Filter. All other values will be ignored')
    general.add_argument('--vector', default="False",
                         help='If set as True, uses the Vector embedding query. All other values will be ignored')

    args = parser.parse_args()

    if len(vars(args)) == 0:
        parser.print_usage()
        exit()

    host = args.host
    port = args.port
    if args.user:
        password = getpass()
        auth = (args.user, password)

    base_url = "https://{}:{}/".format(host, port)
    opensearch = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        # client_cert = client_cert_path,
        # client_key = client_key_path,
        use_ssl=True,
        verify_certs=False,  # set to true if you have certs
        ssl_assert_hostname=False,
        ssl_show_warn=False,

    )
    index_name = args.index
    use_synonyms = True if args.synonyms=="True" else False
    use_cat_filter = True if args.catFilter=="True" else False
    use_vector = True if args.vector=="True" else False

    query_prompt = "\nEnter your query (type 'Exit' to exit or hit ctrl-c):"
    print(query_prompt)
    query = ""
    while query != "Exit":
        query = input()
        query = query.rstrip()
        if query == "Exit":
            break

        print(f"use_synonyms:{use_synonyms}")
        print(f"use_cat_filter:{use_cat_filter}")
        search(client=opensearch, user_query=query, index=index_name, use_synonyms=use_synonyms, use_cat_filter=use_cat_filter, use_vector=use_vector)

        print(query_prompt)

