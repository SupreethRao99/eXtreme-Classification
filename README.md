# Extreme Classification

Extreme classification deals with multi-class and multi-label problems involving an extremely large number of choices.
This repository contains a classifer that has been trained to classify products on Amazon into their Node ID's. The dataset used was obtained from Amazon ML Challenge 2021.

Browse node ID's are numeric codes that identify inside Amazon, a given product category. There are more than 30 thousand product categories on Amazon, each one identified by a unique Node ID. In Amazon's own words
> *Browse Node ID's are positive integers that uniquely identify product
> sets, such as Literature & Fiction: (17), Medicine: (13996), Mystery &
> Thrillers: (18), Nonfiction: (53), Outdoors & Nature: (290060). Amazon
> uses thousands of browse node ID's*

## Approach
The input dataframe is cleaned and a custom BytePairEncoding (BPE) tokenizer from HuggingFace tokenizers is trained on the corpus. The text is then tokenized and the FastText library is used for learn text representation and performing classification. It is observed that entire process takes about 45 minutes. For an in depth explanation, take a look at this [notebook](https://github.com/SupreethRao99/Amazon-ML-Challenge/blob/main/FastTextClassifier.ipynb)

![fasttext-flowchart](https://user-images.githubusercontent.com/55043035/173225770-873cfea0-b8f6-4384-8830-829571602f22.png)

## Demo

Check out the demo on [Streamlit](https://share.streamlit.io/supreethrao99/extreme-classification/main/app.py)
