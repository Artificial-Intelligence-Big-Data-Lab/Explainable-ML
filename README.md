# Explainable Machine Learning Exploiting News and Domain-specific Lexicon for Stock Market Forecasting

## 0. Content

This repository contains the code, the data and the resources to execute the algorithm described in the paper "Explainable Machine Learning Exploiting News and Domain-specific Lexicon for Stock Market Forecasting" (S. Carta, S. Consoli, L. Piras, A.S. Podda, D. Reforgiato Recupero).

Here follows a short description of the content of the repository:
- companies_by_industry: mappings from company codes used in SP500 dataset to codes used in DNA dataset; the companies are grouped by the 3 industries analized in the paper (namely, Information Technology, Financial and Industrials).
- data: limited example news published in 2020 about the 3 industries, respectively.
- lexicons: used when creating new lexicons. Each subfolder inside `2 classes` is named according to the parameters passed to create the lexicons; similarly, a function in `create_lexicons.py` uses the same naming system to retrieve the lexicons based on the parameters. Each subfolder contains a list of csv files, each representing the lexicon created for the day indicated in the file name. Read documentation in "`create_lexicons.py`" for further reference on the saving system
- samples: ready-to-use feature vectors, computed on the whole portion of DNA studied in the paper.
- `create_lexicons.py`: functions used for creating new lexicons and for fetching previously created lexicons
- `evaluation.py`: accuracy metrics and other utility functions used to evaluate the algorithm
- `explainable_ml.py`: main script, containing the backbone of the algorithm.
- `utils.py`: other utility functions.

Please note that, due to licensing constraints, we cannot publish any news document contained in the Dow Jones' Data, News and Analytics dataset, which was employed in the experimental framework of the paper. However, in order to allow the reproducibility of the experiments illustrated in the paper, we hereby make available a set of ready-to-use lexicons and feature vectors, based on the original data.

## 1. MongoDB configuration

Install mongodb community edition: https://www.mongodb.com/try/download/community
We recommend also installing Robo3T, to manage databases and collections from graphic interface: https://robomongo.org/download

Once the installation is complete, create a database called "financial_forecast" (this operation is very easy from Robo3T interface).

Now let's create mongodb collections for news.

- Go to directory `Explainable-ML/data`
- run command >
```
mongoimport --db financial_forecast --collection Information Technology_news --file Information Technology_news_example.json --legacy
```

Do the same for the other two files. Please pay attention to the fact that the names of the mongodb collections must match the strings inserted in the code.


## 2. Python modules installation

In order to run the code, you must have the following Python modules installed and updated:
- sklearn : https://scikit-learn.org/stable/install.html
- numpy : https://numpy.org/install/
- pymongo : https://pymongo.readthedocs.io/en/stable/installation.html


## 3. Executing the code

The main script to run is `explainable_ml.py`. 
You can simply run this command from shell:
`Explainable-ML` > `python event_detector.py`


Please note that this script is configured to run with feature vectors that were pre-computed on the whole portion of dataset analyzed in the paper. If you intend to run the algorithm on personalized lexicons with custom parameters, you will need to use your own news dataset and adapt the code accordingly.
