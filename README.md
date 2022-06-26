# Search with Machine Learning
The following readme notates the project work and relevant documentation for project submissions. The original README from the forked project is available [here](Original-README.md).

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