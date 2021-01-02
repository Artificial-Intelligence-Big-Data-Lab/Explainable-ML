# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import pymongo, pandas
import numpy as np


"""
Get the ids ('an' field in the json documents) and the dates of all the news included in the specified time interval, where
at least on of the companies passed as parameter is relevant.
'relevance_mode' determines whether we pick the articles where the company appears in relevant_companies or about_companies or in both.
"""
def get_all_ids(companies, min_date, max_date, relevance_mode, mongo_collection):
    
    if type(min_date) == type('str'):
        min_date = datetime.strptime(min_date, '%Y-%m-%d').date()
    if type(max_date) == type('str'):   
        max_date = datetime.strptime(max_date, '%Y-%m-%d').date()
    
    client = pymongo.MongoClient()
    try:
        if relevance_mode == 'both':
            all_news = client.financial_forecast.get_collection(mongo_collection).find({'$or': [{'relevant_companies': {"$in": companies}}, {'about_companies': {"$in": companies}}]},
                                                                                         {'an':1, 'ingestion_datetime':1, 'converted_ingestion_datetime_utc-5':1})
        else:    
            all_news = client.financial_forecast.get_collection(mongo_collection).find({relevance_mode+'_companies': {"$in": companies}},
                                                                                       {'an':1, 'ingestion_datetime':1, 'converted_ingestion_datetime_utc-5':1})        
        
        
        sorted_news = sorted([(r['ingestion_datetime'], r['converted_ingestion_datetime_utc-5'], r['an']) 
                              for r in all_news 
                                  if (datetime.strptime(r['converted_ingestion_datetime_utc-5'], '%Y-%m-%d %H:%M:%S').date() >= min_date and 
                                      datetime.strptime(r['converted_ingestion_datetime_utc-5'], '%Y-%m-%d %H:%M:%S').date() <= max_date)], 
                                      key = lambda x : x[0])
        sorted_dates = np.array([datetime.strptime(x[1], '%Y-%m-%d %H:%M:%S').date() for x in sorted_news])
        sorted_ids = np.array([x[2] for x in sorted_news])
        return sorted_ids, sorted_dates

    finally:
        client.close()
        
        
        
"""
Searches for news included in the look-back interval, where the company passed as parameter is relevant.

news_per_day is a dictionary indexed by date (1st order key) and company (2nd order key).
At news_per_day[d][c] there is the list of the indices (wrt the news_items array in 'build_samples') of all the
news published on the date d which are relevant for c.

Returns a list of indices of news wrt to news_items (see 'build_samples'). 
"""
def fetch_previous_news(news_per_day, company, current_date, look_back):
    
    previous_news = []
    #we iterate through the dates in reverse order, unitl we get off the look-back interval
    for date in sorted(news_per_day, reverse=True):
        if date >= current_date:
            continue
        if date < current_date - timedelta(days=look_back):
            break
        if company in news_per_day[date]:
            for news_index in news_per_day[date][company]:
                previous_news.append(news_index)
    
    return previous_news


"""
Given a list of news, organizes them in a dictionary indexed by date (1st order key) and company (2nd order key).
news_items must be sorted by date.
"""
def get_news_per_day(news_items, companies, relevance_mode):
    
    news_per_day = {}
    
    #useful to enter a new 1st order key in the dict whenever the date changes
    previous_date = datetime.strptime('1900-01-01', '%Y-%m-%d')  
    #we iterate through the news (already sorted by date)
    for news_index in range(len(news_items)):
        item = news_items[news_index]
        date = datetime.strptime(item['converted_ingestion_datetime_utc-5'], '%Y-%m-%d %H:%M:%S').date()
        
        #whenever the date changes, enter a new 1st order key in the dict
        if date != previous_date:
            news_per_day[date] = {}
            previous_date = date
        
        #for any company which we are interested in that is also relevant for the news,
        #we add the news index to the corresponding list 
        for c in companies:
            if ((relevance_mode == 'both' and (c in item['relevant_companies'] or c in item['about_companies']))
                or (relevance_mode in ('about','relevant') and (c in item[relevance_mode+'_companies']))):
                try:
                    news_per_day[date][c].append(news_index)
                except KeyError:
                    news_per_day[date][c] = [news_index]
    return news_per_day



"""
Gets the price variations for the companies passed as parameter in the speficied time interval.
Params:
    - min_date: first date to start storing price variations
    - max_date: last date to start storing price variations.
    - companies: stocks we are interested in
    - type_of_delta: can be:
        - delta: the price variation is calculated as close - open
        - delta_percentage: the price variation is calculated as close - open / open
        - delta_percentage_previous_day: the price variation is calculated as close_t1 - close_t0 / close_t0
    - forecast_horizon: number of days we want to look ahead for the forecast.
      0 means daily prediction (close and open refer to the same day), 1 means the timespan of 2 days is
      used for the prediction, so close is taken from the following day and open from the current day.
      In cases where the market is shut, we consider the next useful date in the price dataset which is 
      equal or bigger than current date + forecast horizon
      
Returns:
    market_per_day: a dict indexed by company (1st order key) and date (2nd order key).
    At market_per_day[c][d] is the price variation wrt d for stock c.
"""
def get_market_per_day(min_date, max_date, companies, industry, type_of_delta='delta_percentage_previous_day', forecast_horizon=0):
    
    #dict that maps company names used in DNA to names used in SP500
    dna_to_sp500 = {}
    
    if industry == 'SP500':
        dna_to_sp500 = {'SP500' : 'SP500'}
    else:
        with open('../companies_per_industry/'+industry+'.csv', 'r') as mapping:
            for x in mapping.readlines()[1:]:
                fields = x.strip().split('\t')
                sp500_name = fields[1]
                dna_names = fields[2].split(',')
                for n in dna_names:
                    dna_to_sp500[n] = sp500_name
  
    market_per_day = {}
    for c in companies:
#        print(c)
        market_per_day[c] = {}
        prices = pandas.read_csv('../SP500_market_data/'+dna_to_sp500[c]+'.csv')
        prices['Date'] = [datetime.strptime(d, '%Y-%m-%d').date() for d in prices['Date']]
        
        dates = prices['Date'].tolist()
        opens = prices['Open'].tolist()
        closes = prices['Adj Close'].tolist()

        #for iterate through dates first with index i and then with j to calculate the delta
        for i in range(len(dates)):
            if dates[i] < min_date:
                continue
            if dates[i] > max_date:
                break
            for j in range(i, len(dates)):
                #we need >= because, due to closing of the market, there are jumps in the dates.
                #the variation is always taken between the current date (at i) and
                #the first date that is equal or bigger than the current date plus the
                #forecast horizon
                if dates[j] >= dates[i] + timedelta(days=forecast_horizon):
                    if type_of_delta == 'delta':
                        value = closes[j] - opens[i]
                    elif type_of_delta == 'delta_percentage':
                        value = 100*((closes[j] - opens[i]) / opens[i])
                    elif type_of_delta == 'delta_percentage_previous_day':                    
                        value = 100*((closes[j] - closes[i-1]) / closes[i-1])


                    market_per_day[c][dates[i]] = value
                    break    
        
    return market_per_day


def get_companies_by_industry(industry):
                    
    if industry == 'SP500':
        return ['SP500']
    
    sp500_to_dna = {}
    with open('companies_per_industry/'+industry+'.csv', 'r') as mapping:
        for x in mapping.readlines()[1:]:
            fields = x.strip().split('\t')
            sp500_name = fields[1]
            dna_names = fields[2].split(',')
            sp500_to_dna[sp500_name] = dna_names
            
    cs = [sp500_to_dna[c] for c in sp500_to_dna]
    return [item for sublist in cs for item in sublist]


"""
Utility function used to find days on which a company stock price records a variation larger than min_variation.
"""
def find_variations(min_date, max_date, company, industry, min_variation=5):
    
    min_date = datetime.strptime(min_date, '%Y-%m-%d').date()
    max_date = datetime.strptime(max_date, '%Y-%m-%d').date()
    
    market = get_market_per_day(min_date, max_date, companies=[company], industry=industry)
    
    for date in market[company]:
        if abs(market[company][date]) >= min_variation:
            print(date, market[company][date])
            
            
def get_deltas_per_date(min_date, max_date, delta_timespan=7):
    
    min_date = datetime.strptime(min_date, '%Y-%m-%d').date()
    max_date = datetime.strptime(max_date, '%Y-%m-%d').date()
    
    prices = pandas.read_csv('../SP500_market_data/SP500.csv')

    dates = prices['Date'].tolist()
    opens = prices['Open'].tolist()
    closes = prices['Adj Close'].tolist()
    
    deltas_per_date = {}
    for i in range(len(dates)-delta_timespan):
        current_date = datetime.strptime(dates[i], '%Y-%m-%d').date()
        if current_date < min_date:
            continue
        if current_date > max_date:
            break
        for j in range(i+1, len(dates)):
            next_date = datetime.strptime(dates[j], '%Y-%m-%d').date()
            if (next_date - current_date).days >= delta_timespan:
                delta =  100 * ((closes[j] - opens[i]) / opens[i]) 
#                label = 1 if delta > delta_threshold else 0
#                print('\nCurrent:', current_date)
#                print('Next:', next_date)
#                print('Delta:', delta, '(', closes[j], '-', opens[i], ')')
#                print('Label:', label)
                deltas_per_date[current_date] = delta
                break
    
    return deltas_per_date
                
