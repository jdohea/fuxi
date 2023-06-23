# %%
import sys
sys.path.append('../')
import os
import logging
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import H5DataLoader
from model_zoo import DeepFM
from model_zoo import FM
import pickle


# Load params from config files
config_dir = './config/FM_ONLY_h5_config'
experiment_id = 'FM_default' # corresponds to h5 input `data/tiny_h5`
params = load_config(config_dir, experiment_id)

# set up logger and random seed
set_logger(params)
logging.info("Params: " + print_to_json(params))
seed_everything(seed=params['seed'])

# Load feature_map from json
data_dir = os.path.join(params['data_root'], params['dataset_id'])
feature_map_json = os.path.join(data_dir, "feature_map.json")
feature_map = FeatureMap(params['dataset_id'], data_dir)
_ = feature_map.load(feature_map_json, params)
_ = logging.info("Feature specs: " + print_to_json(feature_map.features))


# %%
# Get train and validation data generators from h5
train_gen, valid_gen = H5DataLoader(feature_map,
                                    stage='train',
                                    train_data=params['train_data'],
                                    valid_data=params['valid_data'],
                                    batch_size=params['batch_size'],
                                    shuffle=params['shuffle']).make_iterator()


# %%
# Model initialization and fitting
model = FM(feature_map, gpu=-1, **params)
start_time = datetime.now()
model.fit(train_gen, validation_data=valid_gen, epochs=params['epochs'])
train_time = datetime.now() - start_time



# %%

start_time = datetime.now()
y_pred = model.predict(valid_gen)
pred_time = datetime.now() - start_time

with open(os.path.join(config_dir, 'model_config.yaml'), 'r') as cfg_file:
    model_config = cfg_file.read()

# %%
import numpy as np

y_true=[]
for batch_data in valid_gen:
    y_true.extend(model.get_labels(batch_data).data.cpu().numpy().reshape(-1))
y_true = np.array(y_true, np.float64)

from dump_results import dump_results

pred_time = str(pred_time)
train_time = str(train_time)
try:
    num_epochs_to_converge= model._epoch_index + 1
except:
    num_epochs_to_converge = 'failed'
dump_results(y_true, y_pred,model_config, experiment_id+'_no_shuffle', pred_time, train_time, model, model_type='FM_Only',num_epochs_to_converge=num_epochs_to_converge )
# get test predictions
test_gen = H5DataLoader(feature_map,
                        stage='test',
                        test_data=params['test_data'],
                        batch_size=params['batch_size'],
                        shuffle=False).make_iterator()
y_pred = model.predict(test_gen)
dump_results(y_true, y_pred,model_config, experiment_id+'_no_shuffle_test', pred_time, train_time, model, model_type='DCN',num_epochs_to_converge=num_epochs_to_converge )




