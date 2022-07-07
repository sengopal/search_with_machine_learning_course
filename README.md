# Search with Machine Learning
The following readme notates the project work and relevant documentation for project submissions. The original README from the forked project is available [here](Original-README.md).

## Week 3
The following are the project assessment questions and responses.


### Project Assessment

To assess your project work, you should be able to answer the following questions:

#### For query classification

1.How many unique categories did you see in your rolled up training data when you set the minimum number of queries per category to 1000?   
**388**

2. How many unique categories did you see in your rolled up training data when you set the minimum number of queries per category to 10000?  
**70**

3. What were the best values you achieved for R@1, R@2, and R@3? You should have tried at least a few different models, varying the minimum number of queries per category, as well as trying different fastText parameters or query normalization. Report at least 2 of your runs.  

|                      Model                       |           Metrics            |
|:------------------------------------------------:|:----------------------------:|
|  threshold=1000; lr=0.4, epoch=25, wordNgrams=1  | R@1=0.52, R@2=0.65, R@3=0.71 |
| threshold=1000; lr=0.5, epoch=100, wordNgrams=3  | R@1=0.52, R@2=0.65, R@3=0.71 |
| threshold=10000; lr=0.6, epoch=100, wordNgrams=3 | R@1=0.58, R@2=0.72, R@3=0.78 |
| threshold=10000; lr=0.5, epoch=100, wordNgrams=3 | R@1=0.60, R@2=0.74, R@3=0.80 |
| threshold=10000; lr=0.3, epoch=25, wordNgrams=3  | R@1=0.61, R@2=0.75, R@3=0.81 |

#### For integrating query classification with search

1. Give 2 or 3 examples of queries where you saw a dramatic positive change in the results because of filtering. Make sure to include the classifier output for those queries.



2. Give 2 or 3 examples of queries where filtering hurt the results, either because the classifier was wrong or for some other reason. Again, include the classifier output for those queries.


## Week 2
The following are the project assessment questions and responses.

#### Level 1: For classifying product names to categories
The notebook [project2_level1_product_names_classification](week2/project2_level1_product_names_classification.ipynb) has all the steps involved. 


1. **What precision (P@1) were you able to achieve?**  
P@1 = 0.747136013330

2. **What fastText parameters did you use?**  
lr=1.0, epoch=25,wordNgrams=3

3. **How did you transform the product names?**  
Normalization using the provided script which removes all non-alphanumeric characters other than underscore, converts all letters to lowercase and trims excess space characters so that tokens are separated by a single space.

4. **How did you prune infrequent category labels, and how did that affect your precision?**  
Pruned the dataset to have only products whose category labels have atleast 500 products associated with them. Also experimented with having atleast 100 products associated. This improved the precision by atleast 12% 
    
5. **How did you prune the category tree, and how did that affect your precision?**  
   Not attempted        


#### Level 2: For deriving synonyms from content
The notebook [project2_level2_derive_synonyms](week2/project2_level2_derive_synonyms.ipynb) has all the steps involved.
    
1. **What were the results for your best model in the tokens used for evaluation?** 

|     query    |              neighbours               |
|:------------:|:-------------------------------------:|
| headphones   |    ['earbud', 'ear', 'headphone']     |
| laptop       |        ['netbook', 'notebook']        |
| freezer      |  ['freezers', 'refrigerator', 'mug']  |
| nintendo     |      ['ds', 'wii', 'nintendogs']      |
| whirlpool    |  ['frigidaire', 'maytag', 'biscuit']  |
| kodak        |     ['easyshare', 'm763', 'c813']     |
| ps2          |    ['playstation', 'xbox', 'ps3']     |
| razr         |     ['motorola', 'krzr', 'droid']     |
| stratocaster |   ['telecaster', 'strat', 'fender']   |
| holiday      |   ['holidays', 'vibes', 'stocking']   |
| plasma       |       ['600hz', '480hz', '42']        |
| leather      | ['leatherskin', 'armless', 'hipcase'] |


2. **What fastText parameters did you use?**  
model='skipgram',epoch=25, minCount=20

3. **How did you transform the product names?**  
Normalized the product titles using the provided script which converts them to lowercase, removing unusual characters and stemming.

#### Level 3: For integrating synonyms with search
The notebook [project2_level3_integrating_synomyns](week2/project2_level3_integrating_synomyns.ipynb) has all the steps involved.

1. **How did you transform the product names (if different than previously)?**  
Just the regular normalization, nothing additional.

2. **What threshold score did you use?**  
**0.65** was used since the value 0.8 was very restrictive leaving multiple words without synonyms. This might be side effect of the training epochs as with more training, the vectors move closer/away from each other with larger magnitudes pushing the thresholds further.

3. **Were you able to find the additional results by matching synonyms?**
    
    | query     | without synonyms | with synonyms |
    |-----------|------------------|---------------|
    | earbuds   | 1205             | 3572          |
    | nespresso | 8                | 420           |
    | dslr      | 2837             | 10000         |

#### Level 4: For classifying reviews
The notebook [project2_level4_reviews_notebook](week2/project2_level4_reviews_notebook.ipynb) has all the steps involved.

1. **What precision (P@1) were you able to achieve?**  
   P@1 = 0.6839

3. **What fastText parameters did you use?**  
   lr=0.65, epoch=300, wordNgrams=3

4. **How did you transform the review content?**  
   Tokenization and stemming of both the title and comment and then concatenated them together.

5. **What else did you try and learn?**  
   Various tokenization and stemming processes. Utilization of hyperparameters for fasttext. Tried to play with hyper parameter optimizations techniques.
       



## Week 1
The analysis of different experiments using varying weights for main and rescoring queries and with different featuresets is available below.

![](week1/week1_experiments.png)

> **The best MRR score for LTR model is 0.724**

### Hand tuned Model
The relevance judgement for the four test queries for the **hand tuned model** is available below.
![](week1/Analysis-Handtuned.png)

### LTR Model
The relevance judgement for the four test queries for the **LTR model** is available below.

![](week1/Analysis-LTR.png)


### Reproducability steps
1. Ensure that the dataset is loaded in the folder `datasets`
2. Run `./ltr-end-to-end.sh -y -m 0 -c quantiles` to generate and run the test against the model.