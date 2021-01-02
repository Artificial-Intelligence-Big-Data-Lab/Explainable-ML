# -*- coding: utf-8 -*-
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, fbeta_score
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import os

def check_events(companies, dates, labels, deltas, clusters, predicted_labels=[], delay=0):
    
    first_found = True
    for i in range(len(companies)):
        if (companies[i] == 'AMZCOM' and (dates[i] - timedelta(days=delay)).strftime('%Y-%m-%d') == '2017-10-27' or
            companies[i] == 'APPLC' and (dates[i] - timedelta(days=delay)).strftime('%Y-%m-%d') == '2019-01-03' or
            companies[i] == 'ONLNFR' and (dates[i] - timedelta(days=delay)).strftime('%Y-%m-%d') == '2018-07-26' or
            companies[i] == 'GOOG' and (dates[i] - timedelta(days=delay)).strftime('%Y-%m-%d') == '2015-07-17' or
            companies[i] == 'BOEING' and (dates[i] - timedelta(days=delay)).strftime('%Y-%m-%d') == '2019-03-11' or
            companies[i] == 'COCA' and (dates[i] - timedelta(days=delay)).strftime('%Y-%m-%d') == '2019-02-14' or
            companies[i] == 'CCCCMT' and (dates[i] - timedelta(days=delay)).strftime('%Y-%m-%d') == '2019-02-14' or
            companies[i] == 'CNYC' and (dates[i] - timedelta(days=delay)).strftime('%Y-%m-%d') == '2016-06-24' or
            companies[i] == 'BIGMAC' and (dates[i] - timedelta(days=delay)).strftime('%Y-%m-%d') == '2015-10-22' or
            companies[i] == 'WLMRT' and (dates[i] - timedelta(days=delay)).strftime('%Y-%m-%d') == '2018-02-20' or
            companies[i] == 'JAJOHI' and (dates[i] - timedelta(days=delay)).strftime('%Y-%m-%d') == '2018-12-14'):
            
            if predicted_labels and first_found:
                first_found = False
                print('\nEvents in test set:')
            if (not predicted_labels) and first_found:
                print('\nEvents in training set:')
            
            print('\n')
            print(dates[i], companies[i])
            print('Delta:', deltas[i], 'Label:', labels[i])
            if predicted_labels:
                print('Predicted label:', predicted_labels[i])
            print('Clustered news:')
            for k in clusters[i]:
                print(k)
                for n in clusters[i][k]:
                    print(n[1:])
    
    

def compute_metrics(test_labels, predicted_labels, labels_to_include, write_to_file=False, file_name=''):
        
    total_samples = len(test_labels)
    if total_samples > 0:
        perc_neg1 = round(len([x for x in test_labels if x == -1]) / total_samples, 2)
        perc_0 = round(len([x for x in test_labels if x == 0]) / total_samples, 2)
        perc_1 = round(len([x for x in test_labels if x == 1]) / total_samples, 2)
#        print(perc_neg1)
#        print(perc_0)
#        print(perc_1)
    else:
        perc_neg1 = 'nan'
        perc_0 = 'nan'
        perc_1 = 'nan'
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(test_labels, predicted_labels)
    metrics['balanced_accuracy'] = balanced_accuracy_score(test_labels, predicted_labels)
    
    precisions = precision_score(test_labels, predicted_labels, labels=labels_to_include, average=None)
    recalls = recall_score(test_labels, predicted_labels, labels=labels_to_include, average=None)
    fscores = f1_score(test_labels, predicted_labels, labels=labels_to_include, average=None)
#    f05_scores = fbeta_score(test_labels, predicted_labels, 0.5, labels=labels_to_include, average=None)
    
    metrics['coverage_0'] = len([x for x in predicted_labels if x == 0]) / len(predicted_labels)
    metrics['coverage_1'] = len([x for x in predicted_labels if x == 1]) / len(predicted_labels)
        
    if len(labels_to_include) == 2:
        metrics['precision_0'] = precisions[0]
        metrics['precision_1'] = precisions[1]
        metrics['precision_impr_0'] = (precisions[0] - perc_0) / perc_0
        metrics['precision_impr_1'] = (precisions[1] - perc_1) / perc_1
        metrics['recall_0'] = recalls[0]
        metrics['recall_1'] = recalls[1]
        metrics['recall_impr_0'] = (recalls[0] - 0.5) / 0.5
        metrics['recall_impr_1'] = (recalls[1] - 0.5) / 0.5
        metrics['fscore_0'] = fscores[0]
        metrics['fscore_1'] = fscores[1]
#        metrics['f0.5_score_0'] = f05_scores[0]
#        metrics['f0.5_score_1'] = f05_scores[1]
        
    elif len(labels_to_include) == 3:
        metrics['coverage_-1'] = len([x for x in predicted_labels if x == -1]) / len(predicted_labels)
        metrics['precision_-1'] = precisions[0]
        metrics['precision_0'] = precisions[1]
        metrics['precision_1'] = precisions[2]
        metrics['precision_impr_-1'] = (precisions[0] - perc_neg1) / perc_neg1
        metrics['precision_impr_0'] = (precisions[1] - perc_0) / perc_0
        metrics['precision_impr_1'] = (precisions[2] - perc_1) / perc_1
        metrics['recall_-1'] = recalls[0]
        metrics['recall_0'] = recalls[1]
        metrics['recall_1'] = recalls[2]
        metrics['recall_impr_-1'] = (recalls[0] - 0.33) / 0.33
        metrics['recall_impr_0'] = (recalls[1] - 0.33) / 0.33
        metrics['recall_impr_1'] = (recalls[2] - 0.33) / 0.33
#        metrics['fscore_-1'] = fscores[0]
#        metrics['fscore_0'] = fscores[1]
#        metrics['fscore_1'] = fscores[2]


    
    if write_to_file:
        with open(file_name, 'w') as output_file:
            for m in metrics:
                output_file.write(m + ' : ' + str(round(metrics[m],2)) +'\n')
            output_file.write('\n\n')
            output_file.write('Total samples in test set:' + str(total_samples))
            output_file.write('\n-1:' + str(perc_neg1))
            output_file.write('\n0:' + str(perc_0))
            output_file.write('\n1:' + str(perc_1))
    
#    print(metrics)
#    print('\n\nTotal samples in test set:', total_samples)
#    print('-1:', perc_neg1)
#    print('0:', perc_0)
#    print('1', perc_1)
                
    return metrics

                   
                   
def compute_average_metrics(metrics_array, write_to_file=False, file_name=''):    
    
    n_classes = 3
    accuracies = [m['accuracy'] for m in metrics_array]
    try:
        precisions_neg1 = [m['precision_-1'] for m in metrics_array]
    except KeyError:
        n_classes = 2
    precisions_0 = [m['precision_0'] for m in metrics_array]
    precisions_1 = [m['precision_1'] for m in metrics_array]
    
    if n_classes == 3:
        recalls_neg1 = [m['recall_-1'] for m in metrics_array]
    recalls_0 = [m['recall_0'] for m in metrics_array]
    recalls_1 = [m['recall_1'] for m in metrics_array]

    if n_classes == 3:
        fscores_neg1 = [m['fscore_-1'] for m in metrics_array]
    fscores_0 = [m['fscore_0'] for m in metrics_array]
    fscores_1 = [m['fscore_1'] for m in metrics_array]
    
    avg_metrics = {'accuracy': np.average(accuracies),
                   'precision_0': np.average(precisions_0),
                   'precision_1': np.average(precisions_1),
                   'recall_0': np.average(recalls_0),
                   'recall_1': np.average(recalls_1),
                   'fscore_0': np.average(fscores_0),
                   'fscore_1': np.average(fscores_1)}
    if n_classes == 3:
        avg_metrics['precision_-1'] = np.average(precisions_neg1)
        avg_metrics['recall_-1'] = np.average(recalls_neg1)
        avg_metrics['fscore_-1'] = np.average(fscores_neg1)

    if write_to_file:
        with open(file_name, 'w') as output_file:
            for m in avg_metrics:
                output_file.write(m + ' : ' + str(round(avg_metrics[m],2)) +'\n')
    
    return avg_metrics


def get_labels_from_file(predictions_file_path):
    
    test_labels = []
    predicted_labels = []
    probabilities = []
    with open(predictions_file_path, 'r') as predictions_file:
        for line in predictions_file:
            fields = line.strip().split(',')
            test_label = float(fields[3])
            predicted_label = float(fields[4])
            prob = [float(fields[5]), float(fields[6])]
            test_labels.append(test_label)
            predicted_labels.append(predicted_label)
            probabilities.append(prob)
    return test_labels, predicted_labels, probabilities

#for folder in os.listdir('results/4) label threshold'):
#        print('\n\n')
#        print(folder)
#        tl, pl, pr = get_labels_from_file('results/4) label threshold/'+folder+'/predictions.csv')
#        compute_metrics(tl, pl, write_to_file=False)
#
#raise Exception
            
            
def plot_metrics(metrics_array, save_plots=False, file_name=''):
    
    x = list(range(len(metrics_array)))
    n_classes = 3
    accuracies = [m['accuracy'] for m in metrics_array]
    try:
        precisions_neg1 = [m['precision_-1'] for m in metrics_array]
    except KeyError:
        n_classes = 2
    precisions_0 = [m['precision_0'] for m in metrics_array]
    precisions_1 = [m['precision_1'] for m in metrics_array]
    
    if n_classes == 3:
        recalls_neg1 = [m['recall_-1'] for m in metrics_array]
    recalls_0 = [m['recall_0'] for m in metrics_array]
    recalls_1 = [m['recall_1'] for m in metrics_array]

    if n_classes == 3:
        fscores_neg1 = [m['fscore_-1'] for m in metrics_array]
    fscores_0 = [m['fscore_0'] for m in metrics_array]
    fscores_1 = [m['fscore_1'] for m in metrics_array]
    
    fig, axs = plt.subplots(2, 2, figsize = (14,10))
    axs[0,0].plot(x, accuracies, label='accuracy')
    axs[0,0].set_title('Accuracy')
    
    if n_classes == 3:
        axs[0,1].plot(x, precisions_neg1, label='precision -1')
    axs[0,1].plot(x, precisions_0, label='precision 0')
    axs[0,1].plot(x, precisions_1, label='precision 1')
    axs[0,1].set_title('Precision')
    
    if n_classes == 3:
        axs[1,0].plot(x, recalls_neg1, label='recall -1')
    axs[1,0].plot(x, recalls_0, label='recall 0')
    axs[1,0].plot(x, recalls_1, label='recall 1')
    axs[1,0].set_title('Recall')
    
    if n_classes == 3:
        axs[1,1].plot(x, fscores_neg1, label='fscore -1')
    axs[1,1].plot(x, fscores_0, label='fscore 0')
    axs[1,1].plot(x, fscores_1, label='fscore 1')
    axs[1,1].set_title('F-score')
    
    for ax in axs.flat:
        ax.set(xlabel='iterations')
        ax.set(ylim=(0,1))
        ax.set(xticks=x)
        ax.legend()
        
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.4)
    
    if save_plots:
        plt.savefig(file_name)
        plt.close(fig)
    else:
        plt.show()
            

