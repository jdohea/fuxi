import numpy as np
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
    

def dump_results(
        y_true,
        y_pred,
        model_config,
        model_config_id,
        time_pred,
        time_train,
        model,
        model_type='DeepFM',
):
    valid_weights = []
    y_valid = y_true
    for t in y_valid:
        if t == 0:
            valid_weights.append(200)
        else:
            valid_weights.append(1)

    def real_prob(k,p):
        if p == 0:
            return 0
        return 1/(1-k+(k/p))


    baseline_ll = log_loss(y_valid, [real_prob(200, np.mean(y_valid))]*len(y_valid), sample_weight=valid_weights)
    model_ll = log_loss(y_valid, y_pred, sample_weight=valid_weights)
    nll = 1 - model_ll/baseline_ll
    auc = roc_auc_score(y_valid, y_pred, sample_weight=valid_weights)


    folder_path = 'results/run_' + model_type +'_'+ datetime.now().strftime("%d-%m-%Y_%H:%M:%S")  + '/'

    # dump model as pickle
    import os
    import pickle

    # Specify the folder path and file name
    file_name = 'model.pkl'

    # Create the directory if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the model to the specified file path
    file_path = os.path.join(folder_path, file_name)
    pickle.dump(model, open(file_path, 'wb'))


    # dump all stats as json
    import json
    with open(folder_path + 'stats.json', 'w') as f:
        json.dump({
            'model_config': model_config,
            'model_config_id': model_config_id,
            'time_pred': time_pred,
            'time_train': time_train,
            'nll': nll,
            'auc': auc,

        }, f)
    run_id = folder_path

    import csv
    # write new line to AUC.csv for columns run_id, model_type, metric, model_config
    with open('results/AUC.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([run_id, model_type, auc, model_config_id])

    # write new line to NLL.csv for columns run_id, model_type, metric, model_config
    with open('results/NLL.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([run_id, model_type, nll, model_config_id])

    # write new line to pred_time.csv for columns run_id, model_type, metric, model_config
    with open('results/pred_time.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([run_id, model_type, time_pred, model_config_id])

    # write new line to train_time.csv for columns run_id, model_type, metric, model_config
    with open('results/train_time.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([run_id, model_type, time_train, model_config_id])

if __name__ == '__main__':
    y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5]
    model_config = 'test'
    model_config_id = 1
    time_pred = 1
    time_train = 2
    model = 'test'
    dump_results(y_true, y_pred, model_config,model_config_id, time_pred, time_train, model)
