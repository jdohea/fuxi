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

if __name__ == '__main__':
    # Load params from config files
    config_dir = './config/full_h5_config'
    experiment_id = 'full_h5_initial'  # corresponds to h5 input `data/tiny_h5`
    params = load_config(config_dir, experiment_id)

    # set up logger and random seed
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    # Load feature_map from json
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    # Get train and validation data generators from h5
    train_gen, valid_gen = H5DataLoader(feature_map,
                                        stage='train',
                                        train_data=params['train_data'],
                                        valid_data=params['valid_data'],
                                        batch_size=params['batch_size'],
                                        shuffle=params['shuffle']).make_iterator()

    # Model initialization and fitting
    model = DeepFM(feature_map, **params)
    model.fit(train_gen, validation_data=valid_gen, epochs=params['epochs'])

    logging.info('***** Validation evaluation *****')
    val_results = model.evaluate(valid_gen)
    logging.info(val_results)

    logging.info('***** Test evaluation *****')
    test_gen = H5DataLoader(feature_map,
                            stage='test',
                            test_data=params['test_data'],
                            batch_size=params['batch_size'],
                            shuffle=False).make_iterator()

    # Start the timer
    start_time = datetime.now()

    test_results = model.evaluate(test_gen)
    elapsed_time = datetime.now() - start_time

    logging.info(f'Test set inference time: {elapsed_time}')

    # Write results to file
    with open('results.txt', 'w') as f:
        f.write("\nModel configuration:\n")
        with open(os.path.join(config_dir, 'model_config.yaml'), 'r') as cfg_file:
            f.write(cfg_file.read())
        f.write(f"\nValidation results: {val_results}\n")
        f.write(f"Test results: {test_results}\n")
        f.write(f"Elapsed time: {elapsed_time}\n")
