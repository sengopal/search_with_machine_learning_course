import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import ltr_utils as lu
import pandas as pd
import query_utils as qu
from opensearchpy import OpenSearch

if __name__ == "__main__":
    train_df = pd.read_csv("/Users/sengopal/build/my-git/search_with_machine_learning_course/ltr_output/train.csv", parse_dates=['click_time', 'query_time'])
    prior_clicks_gb = train_df.groupby(["query"]) #large
    source = ["sku", "name"]

    no_simple = []
    no_ltr_simple = []
    no_hand_tuned = []
    no_ltr_hand_tuned = []
    no_results = {"simple": no_simple, "ltr_simple": no_ltr_simple, "hand_tuned": no_hand_tuned, "ltr_hand_tuned": no_ltr_hand_tuned}

    q = []
    sku = [] #
    rank = []
    type = []
    score = []
    found = [] # boolean indicating whether this result was a match or not
    new = [] # boolean indicating whether this query was in the training set or not
    results = {"query": q, "sku": sku, "rank": rank, "type": type, "found": found, "new": new, "score": score}

    for key in ['Beats']:
        prior_doc_ids = None
        prior_doc_id_weights = None
        query_times_seen = 0 # careful here
        prior_clicks_for_query = None
        seen = False
        try:
            prior_clicks_for_query = prior_clicks_gb.get_group(key)
            if prior_clicks_for_query is not None and len(prior_clicks_for_query) > 0:
                prior_doc_ids = prior_clicks_for_query.sku.drop_duplicates()
                prior_doc_id_weights = prior_clicks_for_query.sku.value_counts() # histogram gives us the click counts for all the doc_ids
                query_times_seen = prior_clicks_for_query.sku.count()
                seen = True
        except KeyError as ke:
            # nothing to do here, we just haven't seen this query before in our training set
            pass

        click_prior_query = qu.create_prior_queries(prior_doc_ids, prior_doc_id_weights, query_times_seen)
        simple_query_obj = qu.create_simple_baseline(key, click_prior_query, filters=None, size=10, highlight=False, include_aggs=False, source=source)
        hand_tuned_query_obj = qu.create_query(key, click_prior_query, filters=None, size=10, highlight=False, include_aggs=False, source=source)

        ltr_simple_query_obj = lu.create_rescore_ltr_query(key, simple_query_obj, click_prior_query, "ltr_model", "week1", rescore_size=500,
                                                           main_query_weight=1, rescore_query_weight=2)

        ltr_hand_query_obj = lu.create_rescore_ltr_query(key, hand_tuned_query_obj, click_prior_query, "ltr_model", "week1", rescore_size=500,
                                                           main_query_weight=1, rescore_query_weight=2)

        opensearch = OpenSearch(
            hosts=[{'host': 'localhost', 'port': 9200}],
            http_compress=True,  # enables gzip compression for request bodies
            http_auth=('admin','admin'),
            # client_cert = client_cert_path,
            # client_key = client_key_path,
            use_ssl=True,
            verify_certs=False,  # set to true if you have certs
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )

        response = opensearch.search(body=ltr_hand_query_obj, index='bbuy_products')
        if response and response['hits']['hits'] and len(response['hits']['hits']) > 0:
            hits = response['hits']['hits']
            limit = len(hits)
            for i in range(limit):
                hit = hits[i]
                sku = int(hit['_source']['sku'][0])
                prod_name = hit['_source']['name'][0]
                rank = i + 1
                score = hit["_score"]
                print(f"{sku}:{prod_name}")
        else:
            print('no hits')