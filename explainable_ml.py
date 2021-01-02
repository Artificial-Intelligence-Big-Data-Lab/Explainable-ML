# -*- coding: utf-8 -*-
import pymongo, os
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from xgboost import XGBClassifier
from sklearn.preprocessing import Normalizer, RobustScaler
import matplotlib.pyplot as plt

from utils import get_all_ids, get_companies_by_industry, fetch_previous_news, get_market_per_day, get_news_per_day
from create_lexicons import process_news_text, create_lexicons, fetch_lexicons, calc_lexicon_match
import evaluation


"""
Searches market_per_day (a dict indexed by company (1st order key) and date (2nd order key)
and finds the first day on which the market is open, after start_day; returns the absolute value of delta.
Assumes that the horizon is 1 day (checks only the next date after start_day)
"""
def find_delta(company, market_per_day, start_date, last_date=None):
        
    if company not in market_per_day or market_per_day[company] == {}:
        return 'not_found'
    
    delta_found = False
    td = 1
    while not delta_found:
        next_date = (start_date + timedelta(days=td))
        if last_date and (next_date >= last_date):
            break
        try:
            delta = market_per_day[company][next_date]
            delta_found = True
        except KeyError:
            td += 1
            if td > 4:  #the 4 is to account for the days in which the market is closed
                break
    if not delta_found:
        return 'not_found'
    
    return delta

    

def get_news_texts_by_delta_interval(industry, news_array, texts_array, current_date, market, HIGH_delta_interval, LOW_delta_interval):
    
    companies = get_companies_by_industry(industry)
    
    texts = []
    associated_companies = []
    dates = []
    ids = []
    for i in range(len(news_array)):
        for c in news_array[i]['about_companies']:
            if c in companies:
                texts.append(texts_array[i])
                associated_companies.append(c)
                dates.append(datetime.strptime(news_array[i]['converted_ingestion_datetime_utc-5'], '%Y-%m-%d %H:%M:%S').date())
                ids.append(news_array[i]['an'])
            
    HIGH_documents = []
    LOW_documents = []
    
    HIGH_ids = []
    LOW_ids = []
    for i in range(len(texts)):
        if len(texts[i]) == 0:
            continue
        company = associated_companies[i]
        if (company not in market) or (market[company] == {}):
            continue
        delta = find_delta(company, market, start_date=dates[i], last_date=current_date)
        if delta == 'not_found':
            continue
        else:
            delta = abs(delta)
        if delta > HIGH_delta_interval[0] and delta < HIGH_delta_interval[1]:
            HIGH_documents.append(texts[i])
            HIGH_ids.append(ids[i])
        elif delta > LOW_delta_interval[0] and delta < LOW_delta_interval[1]:
            LOW_documents.append(texts[i])
            LOW_ids.append(ids[i])
            
    return HIGH_documents, LOW_documents
    


def format_metrics(metrics, test_labels, predicted_labels):
    
    n_HIGH = len([x for x in test_labels if x == 1])
    n_LOW = len([x for x in test_labels if x == 0])
    
    metrics = {'samples_in_test':[n_HIGH, n_LOW],
               'random_precision_HIGH': round(n_HIGH / (n_HIGH + n_LOW),2),
               'random_recall_HIGH': 0.5,
               'balanced_accuracy': round(metrics['balanced_accuracy'],2),
               'coverage_HIGH': round(metrics['coverage_1'],2),
               'precision_HIGH': round(metrics['precision_1'],2),
               'precision_HIGH_over_random': round(metrics['precision_impr_1'],2),
               'recall_HIGH': round(metrics['recall_1'],2),
               'recall_HIGH_over_random': round(metrics['recall_impr_1'],2)
            }
 
    return metrics


def run_baseline_classifier(vectors, labels, dates, min_test_date='2013-01-01', max_test_date='2019-01-01', industry=None):
    
    if type(min_test_date) == type('str'):
        min_test_date = datetime.strptime(min_test_date, '%Y-%m-%d').date()
    if type(max_test_date) == type('str'):   
        max_test_date = datetime.strptime(max_test_date, '%Y-%m-%d').date()
        
    if min_test_date >= max_test_date:
        raise Exception('min_test_date cannot be bigger than max_test_date')
        
    
    for i in range(len(dates)):
        if dates[i] >= min_test_date:
            break
    for j in range(i, len(dates)):
        if dates[j] > max_test_date:
            break

    true_labels = []
    predicted_labels_per_match_threshold = [[],[],[],[],[],[],[],[],[],[],[]]
    for index in range(i,j):
                

        match_HIGH_news = vectors[index][0]
        match_LOW_news = vectors[index][1]
        match_current_news = vectors[index][-1]
        
        true_labels.append(labels[index])
        
        # we define the thresholds used by the predictor to assign the label to the sample
        step = (match_HIGH_news - match_LOW_news) / 10
        match_thresholds = [match_LOW_news+step*t for t in range(11)]
        
        # make a prediction for each value of match_thresholds
        for i in range(len(match_thresholds)):
            if match_current_news > match_thresholds[i]:
                predicted_label = 1
            else:
                predicted_label = 0
            predicted_labels_per_match_threshold[i].append(predicted_label)
    
    
    precisions_by_t = []
    precisions_improvements_by_t = []
    recalls_by_t = []
    recalls_improvements_by_t = []
    fscores_by_t = []
    bal_accuracies_by_t = []
    for i in range(len(match_thresholds)):
        metrics = evaluation.compute_metrics(true_labels, predicted_labels_per_match_threshold[i], labels_to_include=[0,1], write_to_file=False)
        precisions_by_t.append(metrics['precision_1'])
        precisions_improvements_by_t.append([metrics['precision_impr_1']])
        recalls_by_t.append(metrics['recall_1'])
        recalls_improvements_by_t.append([metrics['recall_impr_1']])
        fscores_by_t.append(metrics['fscore_1'])
        bal_accuracies_by_t.append(metrics['balanced_accuracy'])
    
    random_precision = len([x for x in true_labels if x == 1]) / len(true_labels)
    if industry == 'Information Technology':
        rf_precision = 0.18
        rf_precision_impr = 0.26
        rf_recall = 0.43
        rf_acc = 0.55
    elif industry == 'Financial':
        rf_precision = 0.12
        rf_precision_impr = 0.22
        rf_recall = 0.3
        rf_acc = 0.52
    elif industry == 'Industrials':
        rf_precision = 0.24
        rf_precision_impr = 0.5
        rf_recall = 0.46
        rf_acc = 0.59
    rf_fscore = 2 * (rf_precision * rf_recall) / (rf_precision + rf_recall)
    random_fscore = 2 * (random_precision * 0.5) / (random_precision + 0.5) 
    
        
    x = np.arange(len(match_thresholds))
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_title(industry+'\nBalanced accuracy for different match thresholds') 
    ax.plot(x, bal_accuracies_by_t, linestyle='-', marker='o', color='red', label='baseline')
    ax.plot(x, [rf_acc]*len(match_thresholds), linestyle='--', marker='.', color='orange', label='random forest')
    ax.plot(x, [0.5]*len(match_thresholds), linestyle='dotted', marker='.', color='pink', label='random classifier')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.set_xlabel('match threshold')
    ax.legend()
    plt.show()
    
    x = np.arange(len(match_thresholds))
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_title(industry+'\nPrecision for different match thresholds') 
    ax.plot(x, precisions_by_t, linestyle='-', marker='o', color='midnightblue', label='baseline')
    ax.plot(x, [rf_precision]*len(match_thresholds), linestyle='--', marker='.', color='blue', label='random forest')
    ax.plot(x, [random_precision]*len(match_thresholds), linestyle='dotted', marker='.', color='cyan', label='random classifier')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.set_xlabel('match threshold')
    ax.legend()
    plt.show()
    
    x = np.arange(len(match_thresholds))
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_title(industry+'\nRecall for different match thresholds') 
    ax.plot(x, recalls_by_t, linestyle='-', marker='o', color='darkgreen', label='baseline')
    ax.plot(x, [rf_recall]*len(match_thresholds), linestyle='--', marker='.', color='green', label='random forest')
    ax.plot(x, [0.5]*len(match_thresholds), linestyle='dotted', marker='.', color='lime', label='random classifier')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.set_xlabel('match threshold')
    ax.legend()
    plt.show()
    
    x = np.arange(len(match_thresholds))
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_title(industry+'\nF1-score for different match thresholds') 
    ax.plot(x, fscores_by_t, linestyle='-', marker='o', color='indigo', label='baseline')
    ax.plot(x, [rf_fscore]*len(match_thresholds), linestyle='--', marker='.', color='purple', label='random forest')
    ax.plot(x, [random_fscore]*len(match_thresholds), linestyle='dotted', marker='.', color='violet', label='random classifier')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.set_xlabel('match threshold')
    ax.legend()
    plt.show()
    
        
    
def run_rf_tuning(vectors, labels_binary, labels_ternary, dates,
                  n_estimators_values, max_depth_values, min_samples_split_values, min_samples_leaf_values,
                  n_walks=5, n_training_splits=10, file_name='rf_tuning.csv'):
    
    results = []
    for ne in n_estimators_values:
        print('\nN. estimators:', ne)
        for md in max_depth_values:
            print('Max depth:', md)
            for mss in min_samples_split_values:
                for msl in min_samples_leaf_values:
                    p1, pneg1, c1, cneg1 = run_walk_forward(vectors, labels_binary, labels_ternary, dates, 
                                                            n_walks=n_walks, n_training_splits=n_training_splits,
                                                            n_estimators=ne, max_depth=md, min_samples_split=mss, min_samples_leaf=msl)
                    
                    p = (p1 + pneg1) / 2
                    results.append((ne,md,mss,msl,p1,pneg1,p,c1,cneg1))
                    
    results = sorted(results, key=lambda x : x[-3], reverse=True)
    with open('rf_tuning/'+file_name, 'w') as w:
        w.write('N. estimators,Max depth,Min samples split,Min samples leaf,Precision UP,Precision DOWN,Avg Precision,Coverage UP,Coverage DOWN\n')
        for r in results:
            w.write(str(r[0])+','+str(r[1])+','+str(r[2])+','+str(r[3])+','+str(r[4])+','+str(r[5])+','+str(r[6])+','+str(r[7])+','+str(r[8])+'\n')


def run_walk_forward(vectors, labels, dates, n_walks=10, 
                     min_test_date='2013-01-01', max_test_date='2019-01-01', training_set_size=1000,
                     algorithm='random_forest', n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, 
                     learning_rate=0.1, min_child_weight=1, hidden_layers=(100,),
                     random_state=0, industry=None, write_predictions=False, print_trees=False):
    
    print('\nTest set from', min_test_date, 'to', max_test_date)
    if type(min_test_date) == type('str'):
        min_test_date = datetime.strptime(min_test_date, '%Y-%m-%d').date()
    if type(max_test_date) == type('str'):   
        max_test_date = datetime.strptime(max_test_date, '%Y-%m-%d').date()
        
    if min_test_date >= max_test_date:
        raise Exception('min_test_date cannot be bigger than max_test_date')
    
    print('\n\n*************************\nN. WALKS:', n_walks)
    
    if write_predictions:
        if not industry:
            raise Exception('You must pass industry to write to file.')
        n_folder = len(os.listdir('random_forest_output'))
        folder_name = (str(n_folder) + ') ' + industry + '_test_' + str(min_test_date) + '_' + str(max_test_date) + '_trainset_size_' + str(training_set_size)
                       + '_n_estimators_' + str(n_estimators) + '_max_depth_' + str(max_depth) + '_max_samples_split_' + str(min_samples_split)
                       + '_min_samples_leaf_' + str(min_samples_leaf) + ' (' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ')')
        os.mkdir('random_forest_output/' + folder_name)
        os.mkdir('random_forest_output/' + folder_name + '/walks')
        with open('random_forest_output/'+folder_name+'/global.csv', 'w') as global_writer:
            global_writer.write('date,company,delta,label,predicted_label\n')
        
    if algorithm == 'random_forest':
        classifier = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            random_state=random_state, 
                                            class_weight='balanced')
    elif algorithm == 'decision_tree':
        classifier = DecisionTreeClassifier(max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            random_state=random_state, 
                                            class_weight='balanced')    
    elif algorithm == 'mlp':
        classifier = MLPClassifier(hidden_layer_sizes=hidden_layers)
    
    all_test_labels = []
    all_predicted_labels = []
    all_predicted_probabilities = []
    metrics_dict = {}
        
        
    """
    1) get the test window from dates
    2) split the window in n walks
    3) for each test split, get n training samples (if None, all the previous samples)
    """
    try:
        first_test_index = list(dates).index(min_test_date)
    except ValueError:
        try:
            first_test_index = list(dates).index(min_test_date + timedelta(days=1))
        except ValueError:
            first_test_index = list(dates).index(min_test_date + timedelta(days=2))
    
    try:
        last_test_index = list(dates).index(max_test_date)
    except ValueError:
        try:
            last_test_index = list(dates).index(max_test_date + timedelta(days=1))
        except ValueError:
            last_test_index = list(dates).index(max_test_date + timedelta(days=2))
        
        
    test_split_size = (last_test_index - first_test_index) // n_walks
    boundary_dates = [first_test_index+i*test_split_size for i in range(n_walks+1)]
    
    if not training_set_size or training_set_size > boundary_dates[0]:
        training_set_size = boundary_dates[0]
        
    training_splits = []
    test_splits = []
    for i in range(len(boundary_dates)-1):
        test_indices = list(range(boundary_dates[i], boundary_dates[i+1]))
        training_indices = list(range(boundary_dates[i]-training_set_size, boundary_dates[i]))
        training_splits.append(training_indices)
        test_splits.append(test_indices)
        
    if test_splits[-1][-1] < last_test_index:
        test_splits[-1].extend(list(range(test_splits[-1][-1], last_test_index)))
        
    f = 0
    for training_indices, test_indices in zip(training_splits, test_splits): 
        
        f += 1     
        training_indices = [i for i in training_indices]

        print('\nWalk', f, '/', len(training_splits))
        print('\nTraining from', dates[training_indices[0]], 'to', dates[training_indices[-1]], len(training_indices))
        print('Test from', dates[test_indices[0]], 'to', dates[test_indices[-1]], len(test_indices))
        print()
                
        training_vectors = vectors[training_indices]
        training_labels = labels[training_indices]
        test_labels = labels[test_indices]
        
        if algorithm == 'gradient_boosting':
            pos_weight = len([x for x in training_labels if x == 0]) / len([x for x in training_labels if x == 1])
            classifier = XGBClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       scale_pos_weight=pos_weight,
                                       learning_rate=learning_rate,
                                       min_child_weight=min_child_weight)
            
        classifier.fit(training_vectors, training_labels)
        
        if algorithm == 'decision_tree' and print_trees:
            feature_names=['perc_industry_month_HIGH_news',
                           'perc_industry_month_LOW_news',
                           'perc_industry_week_HIGH_news',
                           'perc_industry_week_LOW_news',
                           'perc_industry_current_news',
                           'perc_company_month_HIGH_news',
                           'perc_company_month_LOW_news',
                           'perc_company_week_HIGH_news',
                           'perc_company_week_LOW_news',
                           'perc_company_current_news']
#            plt.figure(figsize=(22,15))
#            plot_tree(classifier, 
#                      feature_names=feature_names,
#                      class_names=['LOW', 'HIGH'],
#                      proportion=True,
#                      label='all',
#                      filled=True, fontsize=12)
            
            print('\nFeature importances:')
            for i,im in enumerate(classifier.feature_importances_):
                print(feature_names[i], ':', im)
            plt.show()
        
        predicted_labels = []
        predicted_probabilities = []
        for ti in test_indices:
            probabilities = classifier.predict_proba(vectors[ti].reshape(1,-1))[0]
            prediction = np.argmax(probabilities)
            predicted_labels.append(prediction)
            predicted_probabilities.append(probabilities)
        
        walk_metrics = evaluation.compute_metrics(test_labels, predicted_labels, labels_to_include=[0,1], write_to_file=False)
        metrics_dict['walk_'+str(f)] = format_metrics(walk_metrics, test_labels, predicted_labels)
        
        all_test_labels.extend(test_labels)
        all_predicted_labels.extend(predicted_labels)
        all_predicted_probabilities.extend(predicted_probabilities)
        
        if write_predictions:
            with open('random_forest_output/'+folder_name+'/walks/walk_'+str(f)+'.csv', 'w') as walk_writer:
                with open('random_forest_output/'+folder_name+'/global.csv', 'a') as global_writer:
                    walk_writer.write('date,company,abs_delta,label,predicted_label,predicted_probabilities\n')
                    for ti,pl,ppr in zip(test_indices, predicted_labels, predicted_probabilities):
                        walk_writer.write(str(dates[ti])+','+companies[ti]+','+str(deltas[ti])+','+str(labels[ti])+','+str(pl)+','+str(ppr)+'\n')
                        global_writer.write(str(dates[ti])+','+companies[ti]+','+str(deltas[ti])+','+str(labels[ti])+','+str(pl)+','+str(ppr)+'\n')


    all_test_labels = all_test_labels[10:-10]
    all_predicted_labels = all_predicted_labels[10:-10]

    metrics = evaluation.compute_metrics(all_test_labels, all_predicted_labels, labels_to_include=[0,1], write_to_file=False)
    metrics_dict['global'] = format_metrics(metrics, all_test_labels, all_predicted_labels)
    
    reduced_labels = [(t,p) for t,p in zip(all_test_labels, all_predicted_labels) if p != 0]
    metrics['balanced_accuracy'] = balanced_accuracy_score([t for t,p in reduced_labels], [p for t,p in reduced_labels])    
    return metrics_dict    

def fetch_samples(industry, min_date, max_date, lexicons, class_threshold=2, 
                  lexicon_parameters=None, create_if_not_found=False):
    
    sample_vectors = []
    sample_labels = []
    sample_deltas = []
    sample_dates = []
    sample_companies = []
    
    file_name = (industry + '_' + lexicon_parameters['type_of_lexicon'] + '_lex_lb_' + str(lexicon_parameters['look_back']) 
                 + '_ngrams_' + str(lexicon_parameters['ngram_range']) + '_stem_' + str(lexicon_parameters['stemming']) 
                 + '_rmv_stopwords_' + str(lexicon_parameters['remove_stopwords']) + '_max_df_' + str(lexicon_parameters['max_df']) 
                 + '_min_df_' + str(lexicon_parameters['min_df']) + '_pos_perc_' + str(lexicon_parameters['positive_percentile_range'])
                 + '_neg_perc_' + str(lexicon_parameters['negative_percentile_range'])
                 + '_cthreshold_' + str(class_threshold)
                 + ('_old_lex_' if lexicon_parameters['old_lexicons'] else '')
                 + ('_excluded_words_' if lexicon_parameters['excluded_words'] else '')
                 + '.csv')
    
    if file_name not in os.listdir('samples/'):
        if create_if_not_found:
            (vectors, labels, deltas, 
             dates, companies) = create_samples(industry=industry, min_date='2009-02-01', max_date='2019-01-01', 
                                                lexicons=lexicons,
                                                class_threshold=class_threshold,
                                                verbose=False, lexicon_parameters=lexicon_parameters, save_in_file=True)
            
            for i in range(len(dates)):
                if dates[i] < datetime.strptime(min_date, '%Y-%m-%d').date():
                    continue
                if dates[i] > datetime.strptime(max_date, '%Y-%m-%d').date():
                    break
                sample_vectors.append(vectors[i])
                sample_labels.append(labels[i])
                sample_deltas.append(deltas[i])
                sample_dates.append(dates[i])
                sample_companies.append(companies[i])
            
            sample_vectors = np.array(sample_vectors)
            sample_labels = np.array(sample_labels)
            sample_deltas = np.array(sample_deltas)
            sample_dates = np.array(sample_dates)
            sample_companies = np.array(sample_companies)
            
            return sample_vectors, sample_labels, sample_deltas, sample_dates, sample_companies

        else:
            raise Exception('Samples not found.')
            
            
    with open('samples/'+file_name, 'r') as file:
        for line in file:
            fields = line.split(',')
            date = datetime.strptime(fields[0], '%Y-%m-%d').date()
            if date < datetime.strptime(min_date, '%Y-%m-%d').date():
                continue
            if date > datetime.strptime(max_date, '%Y-%m-%d').date():
                break
                          
            company = fields[1]
            delta = float(fields[2])
            label = int(fields[3])
            features_string = ','.join(fields[4:]).replace('[', '').replace(']', '')
            features = np.fromstring(features_string, sep=',')
            
            sample_vectors.append(features)
            sample_labels.append(label)
            sample_deltas.append(delta)
            sample_dates.append(date)
            sample_companies.append(company)
            
    sample_vectors = np.array(sample_vectors)
    sample_labels = np.array(sample_labels)
    sample_deltas = np.array(sample_deltas)
    sample_dates = np.array(sample_dates)
    sample_companies = np.array(sample_companies)
            
    return sample_vectors, sample_labels, sample_deltas, sample_dates, sample_companies



def create_samples(industry, min_date, max_date, lexicons,
                   class_threshold=2,
                   verbose=False, lexicon_parameters=None, save_in_file=False):
        
    if save_in_file:
        if not lexicon_parameters:
            raise Exception('You must pass lexicon_parameters to save in file.')
        print('\nWill create and save samples from', min_date, 'to', max_date)
        
    print('\n*************************')
    print(industry, 'from', min_date, 'to', max_date)
    print('Delta threshold:', class_threshold)
    print()
    
    companies = get_companies_by_industry(industry)

    # we define the two classes based on a class threshold (refers to the percentage price variation)
    HIGH_delta_interval = (class_threshold, 100)
    LOW_delta_interval = (0, class_threshold)
    
    # we build news_per_day even if use_news is False, because we need its keys to iterate
    # in the same way for all configurations, to have a fair comparison
    if industry == 'SP500':
        collection_name = 'sp500_news_2009-2019'
    else:
        collection_name = industry+'_news_2000-2019'
        
    # get all news relevant to the companies, in the time interval
    news_ids, news_dates = get_all_ids(companies=companies, min_date=datetime.strptime(min_date, '%Y-%m-%d').date()-timedelta(days=31), max_date=max_date, relevance_mode='about', mongo_collection=collection_name)
    client = pymongo.MongoClient()
    all_news = client.financial_forecast.get_collection(collection_name).find({'an': {"$in": list(news_ids)}}).sort([('ingestion_datetime',1)])
    all_news = np.array(list(all_news))
    all_news_texts = np.array([process_news_text(n) for n in all_news])
    
    # sort all the news in a dict that, for every day d and for every company c,
    # stores the indices (w.r.t all_news) of the news published on d about c
    news_per_day = get_news_per_day(all_news, companies=companies, relevance_mode='about')
    company_count = {}
    for d in news_per_day:
        for c in news_per_day[d]:
            if c in company_count:
                company_count[c] += len(news_per_day[d][c])
            else:
                company_count[c] = len(news_per_day[d][c])
    
    counts = [company_count[c] for c in company_count]
    avg = np.average(counts)
    stdv = np.std(counts)
    
    max_v = -999
    max_c = ''
    min_v = 999
    min_c = ''
    for c in company_count:
        if company_count[c] > max_v:
            max_v = company_count[c]
            max_c = c
        if company_count[c] < min_v:
            min_v = company_count[c]
            min_c = c
    
    print('Average:', avg)
    print('Std:', stdv)
    print('Max:', max_v, max_c)
    print('Min:', min_v, min_c)
    raise Exception
            
        
    # we get the delta of the prices, sorted in a dict indexed by company (1st order key) and date (2nd order key)            
    market_per_day = get_market_per_day(min_date=datetime.strptime(min_date, '%Y-%m-%d').date() - timedelta(days=31), 
                                                       max_date=datetime.strptime(max_date, '%Y-%m-%d').date() + timedelta(days=10), 
                                                       companies=companies, industry=industry, 
                                                       type_of_delta='delta_percentage_previous_day', forecast_horizon=0)
    
    market_days = list(market_per_day[list(market_per_day.keys())[0]].keys())
    
    sample_vectors = []
    sample_labels = []
    sample_deltas = []
    sample_dates = []
    sample_companies = []
    
    for d in market_days:
        
        if (d < datetime.strptime(min_date, '%Y-%m-%d').date()):
            continue
        if (d > datetime.strptime(max_date, '%Y-%m-%d').date()):
            break
        
        if verbose:
            print('\n********************************\nCurrent date:', d)
        
        if d.month == 1 and d.day == 1:
            print(d)
            
        valid_date = True
        
        try:
            current_lexicon = [w for w,s in lexicons[d-timedelta(days=1)]]
        except KeyError:
            if verbose:
                print('\nLexicon not found.')
            valid_date = False
            
        if d-timedelta(days=1) not in news_per_day:
            valid_date = False
            if verbose:
                print('\nSample not valid, because there are no news on the previous day')
        
        
        if valid_date:
                        
            """
            INDUSTRY MONTH
            """
            print('\nINDUSTRY MONTH')
            # we fetch all the news published in the last month about some company of the industry
            industry_month_news = []
            for c in companies:
                month_news = fetch_previous_news(news_per_day, company=c, current_date=d-timedelta(days=1), look_back=30)
                industry_month_news.extend(month_news)
                    
            if len(industry_month_news) > 0:
                industry_month_HIGH_news, industry_month_LOW_news = get_news_texts_by_delta_interval(industry, all_news[industry_month_news], all_news_texts[industry_month_news],
                                                                                                     market=market_per_day, current_date=d-timedelta(days=1),
                                                                                                     HIGH_delta_interval=HIGH_delta_interval, LOW_delta_interval=LOW_delta_interval)
                        
                if verbose:
                    print('\nIndustry month_ HIGH news:', len(industry_month_HIGH_news))
                    print('Industry month LOW news:', len(industry_month_LOW_news))
                  
                if len(industry_month_HIGH_news) < 1:
                    match_industry_month_HIGH_news = 0
                else:                
                    match_industry_month_HIGH_news = calc_lexicon_match(industry_month_HIGH_news, current_lexicon)
                
                if len(industry_month_LOW_news) < 1:
                    match_industry_month_LOW_news = 0
                else:                
                    match_industry_month_LOW_news = calc_lexicon_match(industry_month_LOW_news, current_lexicon)
            
            else:
                match_industry_month_HIGH_news = 0
                match_industry_month_LOW_news = 0
                
            """
            INDUSTRY WEEK
            """
            # we fetch all the news published in the last month about some company of the industry
            industry_week_news = []
            for c in companies:
                week_news = fetch_previous_news(news_per_day, company=c, current_date=d-timedelta(days=1), look_back=7)
                industry_week_news.extend(week_news)
                    
            if len(industry_week_news) > 0:
                industry_week_HIGH_news, industry_week_LOW_news = get_news_texts_by_delta_interval(industry, all_news[industry_week_news], all_news_texts[industry_week_news],
                                                                                                   market=market_per_day, current_date=d-timedelta(days=1),
                                                                                                   HIGH_delta_interval=HIGH_delta_interval, LOW_delta_interval=LOW_delta_interval)
                        
                if verbose:
                    print('\nIndustry week HIGH news:', len(industry_week_HIGH_news))
                    print('Industry week LOW news:', len(industry_week_LOW_news))
                
                if len(industry_week_HIGH_news) < 1:
                    match_industry_week_HIGH_news = 0
                else:                
                    match_industry_week_HIGH_news = calc_lexicon_match(industry_week_HIGH_news, current_lexicon)
                
                if len(industry_week_LOW_news) < 1:
                    match_industry_week_LOW_news = 0
                else:                
                    match_industry_week_LOW_news = calc_lexicon_match(industry_week_LOW_news, current_lexicon)
            
            else:   
                match_industry_week_HIGH_news = 0
                match_industry_week_LOW_news = 0
            
            """
            INDUSTRY CURRENT NEWS
            """
            industry_current_news = []
            for c in companies:
                if c in news_per_day[d-timedelta(days=1)]:
                    for n in news_per_day[d-timedelta(days=1)][c]:
                        industry_current_news.append(all_news_texts[n])
                                        
            match_industry_current_news = calc_lexicon_match(industry_current_news, current_lexicon)
                    
            # on day d, we only make a prediction for companys that have at least 1 news item on d
            for c in companies:
                                    
                try:
                    delta = abs(market_per_day[c][d])
                except KeyError:
                    continue
                    
                if delta >= HIGH_delta_interval[0] and delta < HIGH_delta_interval[1]:
                    true_label = 1
                else:
                    true_label = 0
                
                if verbose:
                    print('\nCurrent sample:', c, delta, true_label)
                    
                if c in news_per_day[d-timedelta(days=1)]:
                    
                    """
                    COMPANY WEEK
                    """
                    # we fetch all the news published in the last month about some company of the industry
                    company_week_news = fetch_previous_news(news_per_day, company=c, current_date=d-timedelta(days=1), look_back=7)
                            
                    if len(company_week_news) > 0:
                        company_week_HIGH_news, company_week_LOW_news = get_news_texts_by_delta_interval(industry, all_news[company_week_news], all_news_texts[company_week_news],
                                                                                                         market=market_per_day, current_date=d-timedelta(days=1),
                                                                                                         HIGH_delta_interval=HIGH_delta_interval, LOW_delta_interval=LOW_delta_interval)
                        if verbose:
                            print('\nCompany week HIGH news:', len(company_week_HIGH_news))
                            print('Company week LOW news:', len(company_week_LOW_news))
                          
                        if len(company_week_HIGH_news) < 1:
                            match_company_week_HIGH_news = 0
                        else:                
                            match_company_week_HIGH_news = calc_lexicon_match(company_week_HIGH_news, current_lexicon)
                        
                        if len(company_week_LOW_news) < 1:
                            match_company_week_LOW_news = 0
                        else:                
                            match_company_week_LOW_news = calc_lexicon_match(company_week_LOW_news, current_lexicon)
                    else:
                        match_company_week_HIGH_news = 0
                        match_company_week_LOW_news = 0


                    """
                    COMPANY MONTH
                    """
                    # we fetch all the news published in the last month about some company of the industry
                    company_month_news = fetch_previous_news(news_per_day, company=c, current_date=d-timedelta(days=1), look_back=30)
                            
                    if len(company_month_news) > 0:
                        company_month_HIGH_news, company_month_LOW_news = get_news_texts_by_delta_interval(industry, all_news[company_month_news], all_news_texts[company_month_news],
                                                                                                           market=market_per_day, current_date=d-timedelta(days=1),
                                                                                                           HIGH_delta_interval=HIGH_delta_interval, LOW_delta_interval=LOW_delta_interval)
                                
                        if verbose:
                            print('\nCompany month_ HIGH news:', len(company_month_HIGH_news))
                            print('Company month LOW news:', len(company_month_LOW_news))
                          
                        if len(company_month_HIGH_news) < 1:
                            match_company_month_HIGH_news = 0
                        else:                
                            match_company_month_HIGH_news = calc_lexicon_match(company_month_HIGH_news, current_lexicon)
                        
                        if len(company_month_LOW_news) < 1:
                            match_company_month_LOW_news = 0
                        else:                
                            match_company_month_LOW_news = calc_lexicon_match(company_month_LOW_news, current_lexicon)
                    else:
                        match_company_month_HIGH_news = 0
                        match_company_month_LOW_news = 0
                        
                        
                    """
                    COMPANY CURRENT NEWS
                    """
                    
                    # fetch and process (tokenize, stem, etc) the news published about company c on day d
                    if verbose:
                        print('\nLatest news on', d-timedelta(days=1))
                    company_current_news = []
                    for n in news_per_day[d-timedelta(days=1)][c]:
                        if verbose:
                            print(all_news[n]['an'], all_news[n]['converted_ingestion_datetime_utc-5'])
                        company_current_news.append(all_news_texts[n])
                    
                    ccnids = []
                    for n in news_per_day[d-timedelta(days=1)][c]:
                        ccnids.append(all_news[n]['an'])
                    match_company_current_news = calc_lexicon_match(company_current_news, current_lexicon)                            
                    
                    features = np.array([match_industry_month_HIGH_news,
                                         match_industry_month_LOW_news,
                                         match_industry_week_HIGH_news,
                                         match_industry_week_LOW_news,
                                         match_industry_current_news,
                                         match_company_month_HIGH_news,
                                         match_company_month_LOW_news,
                                         match_company_week_HIGH_news,
                                         match_company_week_LOW_news,
                                         match_company_current_news])
        
    
                    sample_labels.append(true_label)
                    sample_deltas.append(delta)
                    sample_companies.append(c)
                    sample_dates.append(d)                       
                    sample_vectors.append(features)
                
                
    if save_in_file:
                           
        file_name = (industry + '_' + lexicon_parameters['type_of_lexicon'] + '_lex_lb_' + str(lexicon_parameters['look_back']) 
                     + '_ngrams_' + str(lexicon_parameters['ngram_range']) + '_stem_' + str(lexicon_parameters['stemming']) 
                     + '_rmv_stopwords_' + str(lexicon_parameters['remove_stopwords']) + '_max_df_' + str(lexicon_parameters['max_df']) 
                     + '_min_df_' + str(lexicon_parameters['min_df']) + '_pos_perc_' + str(lexicon_parameters['positive_percentile_range'])
                     + '_neg_perc_' + str(lexicon_parameters['negative_percentile_range'])
                     + '_cthreshold_' + str(class_threshold)
                     + ('_old_lex_' if lexicon_parameters['old_lexicons'] else '')
                     + ('_excluded_words_' if lexicon_parameters['excluded_words'] else ''))
                     

        with open('samples/' + file_name + '.csv', 'w') as writer:
            for dat,c,delt,l,v in zip(sample_dates,sample_companies,sample_deltas,sample_labels,sample_vectors):
                writer.write(str(dat) + ',' + c + ',' + str(delt) + ',' + str(l) + ',' + str(list(v)) + '\n')
                
    sample_vectors = np.array(sample_vectors)
    sample_labels = np.array(sample_labels)
    sample_deltas = np.array(sample_deltas)
    sample_dates = np.array(sample_dates)
    sample_companies = np.array(sample_companies)
    
    return sample_vectors, sample_labels, sample_deltas, sample_dates, sample_companies



industry='Industrials'
collection = 'Industrials_news'
type_of_lexicon = 'only_abs_delta'
stemming=True
remove_stopwords=True
old_lexicons = False
excluded_words = False


lexicon_look_back = 28
ngram_range=(1,1)
max_df = 0.7
min_df = 10
negative_percentile_range = (0,0)
positive_percentile_range = (95,100)

results = {}
pos_lexicons, neg_lexicons = fetch_lexicons(industry=industry, collection_name=collection, type_of_lexicon=type_of_lexicon,
                                             min_date='2005-01-01', max_date='2018-01-01', look_back=lexicon_look_back,
                                             ngram_range=ngram_range, stemming=stemming, remove_stopwords=remove_stopwords,
                                             max_df=max_df, min_df=min_df, excluded_words=excluded_words,
                                             positive_percentile_range=positive_percentile_range, negative_percentile_range=negative_percentile_range, 
                                             create_if_not_found=False)
        
lexicon_parameters = {'type_of_lexicon':type_of_lexicon,
                      'look_back':lexicon_look_back,
                      'ngram_range':ngram_range,
                      'stemming':stemming,
                      'remove_stopwords':remove_stopwords,
                      'max_df':max_df,
                      'min_df':min_df,
                      'positive_percentile_range':positive_percentile_range,
                      'negative_percentile_range':negative_percentile_range,
                      'excluded_words':excluded_words,
                      'old_lexicons':old_lexicons}


(vectors, labels, deltas, 
 dates, companies) = fetch_samples(industry=industry, min_date='2005-01-05', max_date='2017-12-31', 
                                    lexicons=pos_lexicons, class_threshold=2,
                                    lexicon_parameters=lexicon_parameters, 
                                    create_if_not_found=False)


metrics = run_walk_forward(vectors, labels, dates, 
               n_walks=10, training_set_size=None, algorithm='random_forest',
               min_test_date='2016-01-05', max_test_date='2017-12-28',
               max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=0, 
               write_predictions=False, print_trees=True)

print('\nBalananced accuracy:', metrics['global']['balanced_accuracy'])
print('\nRandom Precision HIGH', metrics['global']['random_precision_HIGH'])
print('Random Recall HIGH', metrics['global']['random_recall_HIGH'])
print('\nPrecision HIGH', metrics['global']['precision_HIGH'])
print('Precision HIGH over random', metrics['global']['precision_HIGH_over_random'])
print('\nRecall HIGH', metrics['global']['recall_HIGH'])
print('\nRecall HIGH over random', metrics['global']['recall_HIGH_over_random'])

run_baseline_classifier(vectors, labels, dates, min_test_date='2016-01-05', max_test_date='2017-12-28', industry=industry)
