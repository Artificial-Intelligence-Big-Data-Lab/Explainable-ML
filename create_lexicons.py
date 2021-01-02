# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import pymongo
import os, re
import numpy as np
from gensim.parsing import preprocessing as pproc
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk import ngrams

from utils import get_all_ids, get_companies_by_industry, get_market_per_day

"""
Taken from https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
Please note the operation is in-place
"""
def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


"""
Calculates the average percentage of 'lexicon' words contained in 'documents'.
To obtain this value, we compute the percentage of lexicon words contained in each document and finally apply the average.

It handles also the case with n-grams with n > 1, depending on how the lexicon is composed.
For example, if the longest entry contained by 'lexicon' is a 3-gram, then the function will decompose
each document in 3-grams, 2-grams and 1-grams (simple terms); then, it will start looking for 
matching 3-grams and, subsequently, 2-grams and 1-grams.
Every time a matching 3-gram is found, the terms composing the 3-gram will be removed from the document,
so that they won't be counted when we are looking for 2-grams and 1-grams. Same for different values of n.

Returns the average percentage.
"""
def calc_lexicon_match(documents, lexicon):
    
    max_ngram = 1
    for w in lexicon:
        if len(w.split()) > max_ngram:
            max_ngram = len(w.split())
        
    matches = []
    for i in range(len(documents)):
        document_length = len(documents[i].split())
        words_in_lexicon = 0
        for n in range(max_ngram,0,-1):
            document_ngrams = [' '.join(x) for x in ngrams(documents[i].split(), n)]
            for w in document_ngrams:
                if w in lexicon:
                    words_in_lexicon += n
                    documents[i] = documents[i].replace(w, 'xxx')

        m = 100 * (words_in_lexicon / document_length)
        matches.append(m)
    
    return np.average(matches)


"""
Given a document-term 'matrix', it weights the values of the matrix depending on the prices of the 'market'
and on the mode passed as parameter ('type_of_lexicon').

'type_of_lexicon' can be one of:
    - only_delta: tf-idf values are replaced by the delta (close-open) recorded by 
      the company associated to the corresponding document on the corresponding date
    - only_abs_delta: tf-idf values are replaced by the absolute value of the delta (close-open) recorded by 
      the company associated to the corresponding document on the corresponding date
    - tfidf_x_delta: tf-idf values are multiplied by the delta (close-open) recorded by 
      the company associated to the corresponding document on the corresponding date
    - tfidf_x_abs_delta: tf-idf values are multiplied by absolute value of the delta (close-open) recorded by 
      the company associated to the corresponding document on the corresponding date
"""
def process_tfidf_matrix(matrix, dates, associated_companies, market, last_date, type_of_lexicon='only_delta'):
    
    if type_of_lexicon in ('only_delta', 'only_abs_delta'):
        matrix[matrix.nonzero()] = 1
    
    rows_to_remove = []
    for i in range(len(dates)):
        company = associated_companies[i]
        if (company not in market) or (market[company] == {}):
            rows_to_remove.append(i)
            continue
        
        delta_found = False
        td = 1
        while not delta_found:
            next_date = dates[i] + timedelta(days=td)
            if next_date > last_date:
                break
            try:
                delta = market[company][next_date]
                delta_found = True
            except KeyError:
                td += 1
                if td > 4:  #the 4 is to account for the days in which the market is closed
                    break
        if not delta_found:
            rows_to_remove.append(i)
            continue
                
        if type_of_lexicon == 'only_abs_delta':
            delta = abs(delta)
            
        matrix[i] *= delta
   
    matrix = delete_rows_csr(matrix, rows_to_remove)
    return matrix


"""
Applies preprocessing operations on the text contained in the title, snippet and body of a news, concatenated.
Returns the processed text
"""
def process_news_text(news_item, stemming=True, remove_stopwords=True):
    
    text = ''
    if 'title' in news_item and type(news_item['title']) != type(None):
        text = text + news_item['title'] + ' '
    if 'snippet' in news_item and type(news_item['snippet']) != type(None):
        text = text + news_item['snippet'] + ' '
    if 'body' in news_item and type(news_item['body']) == type('str'):
        text = text + news_item['body'] + ' '
    
    text = process_string(text, stemming=stemming, remove_stopwords=remove_stopwords)
    return text


"""
Used by process_news_text.
"""
def process_string(string, stemming=True, remove_stopwords=True):
    
    string = string.lower()
    abbreviations = re.findall(r'(?:[a-z]\.)+', string)
    for abbr in abbreviations:
        string = string.replace(abbr, abbr.replace('.',''))
    string = pproc.strip_punctuation(string)
    if remove_stopwords:
        string = pproc.remove_stopwords(string)
    if stemming:
        string = pproc.stem_text(string)
    string = string.strip()
    return string

"""
Loads from file and returns the lexicons defined by the parameters passed by the fuction, IF AND ONLY IF such lexicon was previously created
and saved to a file.
If create_if_not_found == True, the function creates the lexicons in case they are not found in any file.
See 'create_lexicon' for reference on the other parameters.
"""
def fetch_lexicons(industry, collection_name, min_date='2009-01-01', max_date='2019-01-01', look_back=28,
                   type_of_lexicon='only_delta', max_df=0.9, min_df=10, ngram_range=(1,3), 
                   stemming=True, remove_stopwords=True, excluded_words=False,
                   positive_percentile_range=(90,100), negative_percentile_range=(0,10), create_if_not_found=False):
    
    folder_name = (industry + '_' + type_of_lexicon + '_lookback_' + str(look_back) + '_ngrams_' + str(ngram_range) + '_stemming_' + str(stemming) 
                        + '_remove_stopwords_' + str(remove_stopwords) + '_max_df_' + str(max_df) + '_min_df_' + str(min_df) + ('_excluded_words' if excluded_words else ''))
    
    if type_of_lexicon in ('only_abs_delta', 'tfidf_x_abs_delta'):
        subfolder = '2 classes'
    else:
        subfolder = '3 classes'
            
    if folder_name not in os.listdir('lexicons/'+subfolder):
        if create_if_not_found:
            print('\nLexicons not found. Creating now and saving to file...')
            pls, nls =  create_lexicons(industry=industry, collection_name=collection_name, min_date='2010-02-01', max_date='2017-01-01',
                                                                    look_back=look_back, type_of_lexicon=type_of_lexicon, max_df=max_df, min_df=min_df, 
                                                                    ngram_range=ngram_range, stemming=stemming, remove_stopwords=remove_stopwords,
                                                                    positive_percentile_range=positive_percentile_range, negative_percentile_range=negative_percentile_range,
                                                                    excluded_words=excluded_words, save_in_file=True)
            
            positive_lexicons = {d:pls[d] for d in pls if d >= datetime.strptime(min_date, '%Y-%m-%d').date() 
                                                          and d <= datetime.strptime(max_date, '%Y-%m-%d').date()}
            negative_lexicons = {d:nls[d] for d in nls if d >= datetime.strptime(min_date, '%Y-%m-%d').date() 
                                                          and d <= datetime.strptime(max_date, '%Y-%m-%d').date()}
            return positive_lexicons, negative_lexicons
        else:
            raise Exception('Lexicons not found. Use create_if_not_found option to create')
    
    positive_lexicons = {}
    negative_lexicons = {}
    
    for file_name in os.listdir('lexicons/'+subfolder+'/' + folder_name):
        
        date = datetime.strptime(file_name[:-4], '%Y-%m-%d').date()
        if date < datetime.strptime(min_date, '%Y-%m-%d').date():
            continue
        if date > datetime.strptime(max_date, '%Y-%m-%d').date():
            break
        
        ranked_words = []
        with open('lexicons/'+subfolder+'/' + folder_name + '/' + file_name, 'r') as file:
            for line in file:
                fields = line.split(',')
                word = fields[0]
                score = float(fields[1])
                ranked_words.append((word,score))
                
        scores = [s for w,s in ranked_words]
        
        positive_upper_bound = np.percentile(scores, positive_percentile_range[1])
        positive_lower_bound = np.percentile(scores, positive_percentile_range[0])
        negative_upper_bound = np.percentile(scores, negative_percentile_range[1])
        negative_lower_bound = np.percentile(scores, negative_percentile_range[0])
        
        positive_lexicon = [(w,s) for w,s in ranked_words if s > 0 and s <= positive_upper_bound and s >= positive_lower_bound]
        negative_lexicon = [(w,s) for w,s in ranked_words if s <= 0 and s <= negative_upper_bound and s >= negative_lower_bound]
                
        positive_lexicons[date] = positive_lexicon
        negative_lexicons[date] = negative_lexicon
        
    return positive_lexicons, negative_lexicons
        
                
            
"""
Creates and returns the lexicons defined by the parameters passed to the function.

Params:
    - industry : one of Information Technology, Financial or Industrials
    - collection_name : name of the mongodb collection where the news related to the 'industry' are stored
    - min_date, max_date : bounds of the time interval of the news based on which the lexicons are created
    - look_back : length of the time interval used to create the lexicon for a SINGLE day
    - type_of_lexicon : see process tfidf matrix
    - max_df : words that appear in more than max_df documents are filtered out 
      (can be an int indicating the exact number of documents or a float between 0 and 1 indicating the proportion)
    - min_df : words that appear in less than min_df documents are filtered out 
      (can be an int indicating the exact number of documents or a float between 0 and 1 indicating the proportion)
    - ngram_range : The lower and upper boundary of the range of n-values for different n-grams to be extracted. 
      All values of n such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, 
      (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams 
    - stemming : apply stemming to the words
    - remove_stopwrds : filter out words such as 'the', 'it', 'a', 'if'
    - positive_percentile_range : the lower and upper bounds of the percentiles used to select the final lexicon from
      the sorted ranking of words. For example, (90, 100) will select for the positive lexicon the words between
      the 90th and the 100th percentile.
    - negative_percentile_range : same as positive_percentile_range, but for negative lexicon
      !!! PLEASE NOTE !!! set this to (0,0) if you are not interested in positive and negative scores,
      but only to the absolute values.
    - excluded_words : calculate the lexicons using the words that would be normally filtered out
      (you can ignore this and always set to False, onyl used for some internal experiment)
    - save_in_file : save the lexicons to files.
    
Returns:
    - two dicts, containing the positive and negative lexicons, respectively, already selected.
      The keys are dates of class datetime.date and format '%Y-%m-%d'; the values are lists of terms. 
    
The saving happens in the following manner:
    - the folder that contains ALL the lexicons is named using the parameters that define the lexicons;
      this is used also by fetch_lexicons to check if the specified lexicons exist
    - the csv file that contains EACH SINGLE lexicon is named after the day for which the lexicon
      should be used (usually the day following the last news used to create the lexicon)
"""     
def create_lexicons(industry, collection_name, min_date='2009-01-01', max_date='2019-01-01', look_back=28,
                    type_of_lexicon='only_delta', max_df=0.9, min_df=10, ngram_range=(1,3), 
                    stemming=True, remove_stopwords=True,
                    positive_percentile_range=(90,100), negative_percentile_range=(0,10),
                    excluded_words=False, save_in_file=False):
    
    if save_in_file:
        if industry == 'SP500':
            min_date = '2009-10-01'
        else:
            min_date = '2005-01-01'
        max_date = '2018-01-01'
        print('\nWill create and save lexicons from',min_date,'to',max_date)
        folder_name = (industry + '_' + type_of_lexicon + '_lookback_' + str(look_back) + '_ngrams_' + str(ngram_range) + '_stemming_' + str(stemming) 
                        + '_remove_stopwords_' + str(remove_stopwords) + '_max_df_' + str(max_df) + '_min_df_' + str(min_df) + ('_excluded_words' if excluded_words else ''))
        
        if type_of_lexicon in ('only_abs_delta', 'tfidf_x_abs_delta'):
            subfolder = '2 classes'
        else:
            subfolder = '3 classes'
        if folder_name not in os.listdir('lexicons/'+subfolder):
            os.mkdir('lexicons/'+subfolder+'/' + folder_name)
        
    min_date = datetime.strptime(min_date, '%Y-%m-%d').date()
    max_date = datetime.strptime(max_date, '%Y-%m-%d').date()
    
    companies = get_companies_by_industry(industry)
    
    # get all news relevant to the companies, in the time interval
    news_ids, news_dates = get_all_ids(companies=companies, min_date=min_date-timedelta(days=look_back), max_date=max_date, relevance_mode='about', mongo_collection=collection_name)
    client = pymongo.MongoClient()
    all_news = client.financial_forecast.get_collection(collection_name).find({'an': {"$in": list(news_ids)}}).sort([('ingestion_datetime',1)])
    all_news = np.array(list(all_news))
     
    market = get_market_per_day(min_date=min_date-timedelta(days=look_back), 
                                max_date=max_date+timedelta(days=10),
                                companies=companies, industry=industry)
            
    documents = []
    associated_companies = []
    dates = []
    ids = []
    for i in range(len(all_news)):
        news_item = all_news[i]
        for c in news_item['about_companies']:
            if c in companies:
                text = process_news_text(news_item, stemming=stemming, remove_stopwords=remove_stopwords)
                documents.append(text)
                associated_companies.append(c)
                dates.append(datetime.strptime(news_item['converted_ingestion_datetime_utc-5'], '%Y-%m-%d %H:%M:%S').date())
                ids.append(news_item['an'])
   
    positive_lexicons = {}
    negative_lexicons = {}
          
    current_date = min_date
    start_index = 0
    while current_date <= max_date:    
        for i in range(start_index, len(documents)):
            if dates[i] >= current_date - timedelta(days=look_back):
                start_index = i
                break
        for j in range(i, len(documents)):
            if dates[j] >= current_date:
                break
        
        selected_documents = documents[i:j]
        selected_associated_companies = associated_companies[i:j]
        selected_dates = dates[i:j]
        selected_ids = ids[i:j]
        
        if len(selected_documents) * max_df < min_df:
            print('Not enough documents on', current_date)
            current_date = current_date + timedelta(days=1)
            continue
        
        vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram_range)
        try:
            matrix = vectorizer.fit_transform(selected_documents)
        except ValueError:
            print('Not enough documents on', current_date)
            current_date = current_date + timedelta(days=1)
            continue
        features = np.array(vectorizer.get_feature_names())
        
        if excluded_words:
            all_words = []
            for doc in selected_documents:
                all_words.extend(doc.split())
            all_words = list(set(all_words))
            vocabulary = [f for f in all_words if f not in features]
            vocabulary.extend(stopwords.words('english'))
            vocabulary = list(set(vocabulary))
            vectorizer_excluded_words = TfidfVectorizer(vocabulary=vocabulary)
            matrix = vectorizer_excluded_words.fit_transform(selected_documents)
            features = np.array(vectorizer_excluded_words.get_feature_names())
        
        if type_of_lexicon != 'only_tfidf':
            matrix = process_tfidf_matrix(matrix, dates=selected_dates, associated_companies=selected_associated_companies, 
                                          market=market, last_date=current_date, type_of_lexicon=type_of_lexicon)
        

        nonzeros = np.array([matrix[:,j].count_nonzero() for j in range(matrix.shape[1])]) 
        matrix_avg = np.asarray(np.sum(matrix, axis=0)).flatten()
        matrix_avg /= nonzeros
                
        sorted_indices = np.asarray(matrix_avg.argsort()[::-1])

        ranked_words = [(features[i], matrix_avg[i]) for i in sorted_indices if not np.isnan(matrix_avg[i])]

        if save_in_file:
            with open('lexicons/'+subfolder+'/' + folder_name + '/' + str(current_date) + '.csv', 'w') as writer:
                for word,score in ranked_words:
                    writer.write(word + ',' + str(score) + '\n')
        
        scores = [s for w,s in ranked_words]

        positive_upper_bound = np.percentile(scores, positive_percentile_range[1])
        positive_lower_bound = np.percentile(scores, positive_percentile_range[0])
        negative_upper_bound = np.percentile(scores, negative_percentile_range[1])
        negative_lower_bound = np.percentile(scores, negative_percentile_range[0])
        
        positive_lexicon = [(w,s) for w,s in ranked_words if s > 0 and s <= positive_upper_bound and s >= positive_lower_bound]
        negative_lexicon = [(w,s) for w,s in ranked_words if s <= 0 and s <= negative_upper_bound and s >= negative_lower_bound]
                
        positive_lexicons[current_date] = positive_lexicon
        negative_lexicons[current_date] = negative_lexicon
    
        current_date = current_date + timedelta(days=1)
    
    return positive_lexicons, negative_lexicons
                

if __name__ == "__main__":
    pos_lexicons, neg_lexicons = create_lexicons(industry='Information Technology', 
                                                 collection_name='Information Technology_news_2000-2019',
                                                 min_date='2002-01-01', max_date='2019-01-01', look_back=40,
                                                 ngram_range=(1,1), stemming=True, remove_stopwords=True,
                                                 type_of_lexicon='only_abs_delta',
                                                 positive_percentile_range=(90,100), negative_percentile_range=(0,0),
                                                 max_df=0.8, min_df=10, save_in_file=False)
    
    for d in pos_lexicons:
        print()
        print(d)
        print(pos_lexicons[d])
        print()
        print(neg_lexicons[d])

        
        