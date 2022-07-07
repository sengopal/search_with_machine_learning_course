import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import nltk

# Useful if you want to perform stemming.
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/Users/sengopal/build/my-git/search_with_machine_learning_course/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/Users/sengopal/build/my-git/search_with_machine_learning_course/datasets/train.csv'
output_file_name = r'/Users/sengopal/build/my-git/search_with_machine_learning_course/datasets/labeled_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

print(f"--min_queries:{min_queries}")

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
# Prepare Parent Dictionary
parents_dict_df = parents_df.set_index("category", drop=True, inplace=False)
parent_dict = parents_dict_df.to_dict()['parent']

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
df['query_norm'] = df['query'].apply(lambda x: stemmer.stem(x))
df_norm = df[['category', 'query_norm']]
df_norm.set_index('category').sort_index()

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
threshold_reached = False
df_cat_count = df_norm.groupby(['category']).count()

# print(df_cat_count[df_cat_count['query_norm'] < 100])
parent_dict[root_category_id] = root_category_id
while not threshold_reached:
    list_to_replace = list(df_cat_count[df_cat_count['query_norm'] < min_queries].index)
    print(f"list_to_replace: {len(list_to_replace)}")
    replace_dict = { k: parent_dict[k] for k in list_to_replace}
    df_norm["category"] = df_norm["category"].replace(replace_dict)
    df_cat_count = df_norm.groupby(['category']).count()
    len_low_count = len(list(df_cat_count[df_cat_count['query_norm'] < min_queries].index))
    print(f"len_low_count: {len_low_count}")
    threshold_reached = (len_low_count <= 0)

print(df_cat_count)

# Create labels in fastText format.
# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df_norm['label'] = '__label__' + df_norm['category']
df_norm['output'] = df_norm['label'] + ' ' + df_norm['query_norm']
df_norm[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
