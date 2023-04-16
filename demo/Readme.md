# Step 1

First we need to generate the data in our other repository Data_transformation.ipynb
This file has 2 outputs:
    1. x3 .h5 files. These files are train, validation and test files.
    2. A columns.txt file. This file is used to make creating the dataset_config.yaml file in the config folder in the demo section of this repository.

Before we run our experiment we need a folder in the data section of this repository. Evey experiment needs to contain the following
 1. train.h5 valid.h5 and test.h5 files
 2. feature_map.json, feature_vocab.json and feature_processor.pkl

We will work on creating the "feature_" files first.

Create a config folder in the demo section of the directory. 
Copy an existing config folder so we can edit that.
Take all of the data in the "columns.txt" file from the other repository and use it to configure the dataset_config.yaml file. Note that this columns file does not have the setup for 3 columns: converted, release_date and release_msrp. So input these manually. Use previously created config files as a guide.

Create a new directory in the data section of this repository and paste the x3 outputed .h5 folders from the data_transformation. Put the location of these 3 files into the beginneing of  dataset_config.yaml where it says train/valid/test_data respectively. 

Change the identifier on the first line of dataset_config to match the name of the folder you created in the data section.

in smadex_setup_full_h5.py change the config direcories to your new respective config locations and the dataset_id to the name of the new data folder also.

Now we run "smadex_setup_full_h5.py". This will take 10 to 15 minutes to run.

When this finishes running we can delete the outputted train/test/valid.h5 files as we only needed the feature_ files as outputs from this step.

_______________________
Our data is now ready for an experiment

Go to the config file in the demo section that we created earlier. In model_config.yaml set a new experiment id (the tag of the first section after "Base"). Then change the dataset_id yin your new experiment to the name of your data folder in the data section of this repository. Change the experiment_id in "smadex_run_experiment_full_h5.py to this new tag. 

Run "smade_run_experiment_full_h5.py". Loading takes approximately 3 minutes per data file (train,val,test). It also takes approximately 20 minutes per epoch (DeepFM) to train based on our full dataset with ~ 1500 columns. Finally validation takes 2 minutes (likewise for test). Overall the whole code for 1 epoch takes between 30 and 35 minutes.
_______________________

To run new experiments with the same model (DeepFM in the example here), just add a new tag in the model_config.yaml file and continue to change the hyperparameres. Then set the experiment_id in the python file to the new tag and the new hyperparameters will be used

If we are running an experiment with a new model, DeepCross, we should create a new config folder, copy over the dataset_config.yaml file and then create a new model_config.yaml setup with the new set of hyperparameters relevant for that model.

We should also create a new run_experiment.py file corresponding to the model we are experimenting with, for example changing the DeepFM() initialisation to DeepCross() initialisation.

_______________

# Results

| No | Model                                    | Benchmark                                                                                                       | Score                             |
|:--:|:----------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|-----------------------------------|
| 1 | [DeepCrossing](./model_zoo/DeepCrossing) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DeepCrossing) | logloss: 0.002606 - AUC: 0.999680 |
| 2 | [DeepFM](./model_zoo/DeepFM)             | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DeepFM)       | logloss: 0.000486 - AUC: 0.999999                               |
| 3 | [DCN](./model_zoo/DCN)                   | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DCN)          | logloss: 0.000215 - AUC: 1.000000 |
| 4 | Logistic Regression                      |                     -                                                                                            | logloss: 0.000391 - AUC: - |


    Todo:

    - set up a portfolio of experiments to run and report on for our current model, let's look at the processed other papers used to compare models. Like in the deepfm and deepcross papers


    - code a timer for how long it takes to inference X samples to compare time of the model

    - look for (code if we can't find) and configure a normalised logloss evaluation for our experiments --> 

    - code an ability to store test inferences to analyse how if we train multiple times with different seeds how volatile our predictions are. i.e. how much they change every time the model is re trained

    - add weights for sampling bias, class weights (200) for logistic regression

    - 1/1-k+(k/p) --> weighted average prediction
    
    - overwrite log loss, result1 = log, result0 = 200log
     
    - Normalised Log loss: - around 1: good, around 0: roughly same, <0: worse average

    - NLL: 1 - (log(y_test,yhat_test)/log(y_test,ybar_test)

    - ybar_test: avg(yhat_test)

    - Amazon Sagemaker: GPU access SSH connection

