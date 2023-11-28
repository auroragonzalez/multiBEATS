# multiBEATS

The original `.mat` data is stored in a separated folder named `data`. Each of the datasets will have their own folder. The datasets which we have work with are *Arabic Digits*, *AUSLAN*, *Character Trayectories*, *CMUsubject16*, *ECG*, *JapaneseVowels*, *LIbras* and *Wafer*, and they are prepared to download in the following Mega link: https://mega.nz/file/vf43USRJ#2VswQAk05ZmJdU1qr3c2f-XCr1EManFSVMWj5ENxCOk

## Preparing the multivariate time series, from .mat to .npy

We use the script **`generate-dataset.py`**. The paths towards the data need to be changed: `conjunto` will have the name of the dataset, for example *JapaneseVowels* and `path` is the path to the folder where the data is stored. For example:
`conjunto = 'JapaneseVowels'`
`path = r"/home/aurorax/thisrepo/data/original/"+ conjunto+ "/"`

With this, we have generated the `.npy` for the X_train, X_test, y_train, y_test subsets and they are stored in the same folder as the original `.mat` file.

## Reducing the X_train and X_test sets using multiBEATS

To apply the reduction to these sets we use the script **`multiBEATStransf.R`**, in which we define the value of *slide (SD)*, *window (WD)* and *nfeat*. The script must be slightly modified to deal with each dataset. Hence, in their respective folder inside /data there is the script used in that particular case.

To use it, we open it and put the directory that contains the script as working directory. It recognizes directly the name of the dataset with which we are working and loads the corresponding mat

We put the parameters to test of `SL`, `WD` and `NFEAT` and do a Run All. This should result in a new folder, at the same level as the dataset, with the `X_train` and `X_test` reduced, and whose name is the name of the dataset, the parameter values and the reduction percentage achieved.

Notes:

- In some cases you have to change the order of the process: most of the time you work first on test and then on train. You have to make sure that in the block that goes before, either train or test, the line where the variable ncolm is defined is not commented, while in the second block it is. If this is not the error that usually gives the complaining of the dimensionality when we do `k = lm2a()`, then we have to make sure that the line in which the variable ncolm is defined is not commented, while in the second block it is.
- As the reduction percentage is calculated at the moment, there are cases in which it is done well and cases in which it is not. The datasets in which the percentage calculated at the beginning of the script and the one calculated with the resulting file sizes are: AUSLAN, JapaneseVowels and Wafer. 
- The correct values are calculated by making the ratio between the previous and new .npy, either with `X_test.npy` or `X_train.npy` . In any case, the name of the folder does not have an affect on the rest of the process.



## File description

- **`generate_dataset.py`** to transform `.mat` into `.npy`

- **`modelAU.py`** to test the default model.

- **`modelAU_cv.py`** to run trials applying cross validation, select the best hyperparameters for LSTM-FCN and train a final model with the optimal configuration.

- **`modelAU_seq2var_lstmfcn.py`** merges the reducction process with seq2var (instead of multiBEATS) followed by training with a LSTM-FCN, which takes the architecture from the best configuration obtained in `modelAU_cv.py.`

- **`seq2Var_rf.py`** for trials applying an alternative reduction to multiBEATS and train a Random Forest.

- **`seq2Var_rf_cv_py`** to run the same tests as before but adding the cross validation process for the Random Forest algorithm.

- **`rf_raw.py`** is used to run test on raw datasets and with multiBEATS reducction applying crossvalidaton with a RandomForestClassifier to compare with what cross validation returns in the process seq2var + Random Forest.

- **`results_to_df.ipynb`** takes the result of the scripts described above (below is explained which script creates each result file). Is used to calculate the mean and standard deviation of the accuracy, time and reduction percentage with **`results_cv.txt`**, to extract the best hyperparameter configuration of the LSTM-FCN for each dataset from **`results_train_cv.txt`**. For **`results_seq2var_rf_cv.txt`** the same is done but obtaining accuracy and its deviation for Var, Seq2Var and bSeq2Var methods along with time. For **`results_rf.txt`** the same process is repeated. Finally, for **`results_seq2var_lstmfcn.txt`**, something similar is done, this time pulling the result with multiindex for one dataset and the three methods (there may be some error in how the times have been calculated).

  They are presented as separate sections in the notebook because there is a reordering step that includes the column names and is different for the results coming from each model, so they are treated separately.

- **`01.functions.R`** has auxiliar functions used in multiBEATStransf.R

- **`/data`** has a folder for each dataset, with the raw data as .mat and .npy and a script **`multiBEATStransf.R`** with which to make the reductions. When executing this script a folder is created with name "dataset" _ "slide" _ "window" - "nfeat"-"reduction" that has the `X_train.npy` and `X_test.npy` and a csv with the times to reduce each one of them. To be able to work with the folder of the reduced dataset it is also necessary to copy `y_train.npy` and `y_test.npy` from the folder of the original dataset to the folder of the reduced dataset.

- **`/utils`**

  - new_constants.py
  - layer_utils.py
  - keras_utils.py
  - keras_utils_cv.py
  - keras_utils_seq2var.py
  - generic_utils.py
  - embedding_utils.py
  - /seq2Var
    - argument_parser_permutations.py
    - data_generator_permutations.py
    - modules.py
    - utils.py



## Detailed script description

### `modelAU.py`

This script contains the function `generate_model_2()` to define the neural network used, `squeeze_excite_block()` which is called by the aforemention function and `main()` which call the one that generates the network and does training and evaluation using `keras_utils.py` imports.

So, the only thing to edit here is the `DATASET_INDEX`, to which we give the value that corresponds to the dataset to study in `new_constants.py`.

It should also be noted that when executing this the `main()` does several iterations of training and evaluation and whose results are sent to where indicated within the function `evaluate_model()` of `keras_utils.py` (at the moment it points to `results.txt`). In each iteration we add a row with dataset index, accuracy, execution time, % reduction (we put this manually).



#### `utils/new_constants.py`

Here are the paths to the directories of raw and reduced datasets, which must be edited manually. So, for example, to add a new reduced dataset to work with, a path on `TRAIN_FILES` and another on `TEST_FILES` must be added, and it is recommended that a number following # is used, to keep track of what dataset corresponds to the following variables:

Como se ha mencionado más arriba aquí están los directorios y valores únicamente de los datasets que sabemos que nos van a ser útiles para el paper. Los valores de los datasets sin reducir son los que estaban en constants.py, puesto que eso no ha cambiado, y para los datasets reducidos:

- `MAX_NB_VARIABLES`: same as the raw dataset.
- `MAX_TIMESTEPS_LIST`: on the last lines of `results_to_df.ipynb` it is calculated. We just load the train or test`.npy` and look for the second index of its shape. 
- `NB_CLASSES_LIST`: same as the raw dataset

The ordering we are following is keeping all variants of the same dataset together (i.e. all ECGs in a row), as this makes it much easier to edit the three parameters above to add more cases. (When adding more examples later we no longer follow this method, we add alternate datasets). As long as the order of the values corresponding to a dataset is consistent, it does not matter.



### Cross Validation Aplication (`modelAU_cv.py`)

We define a copy of the model and functions like `modelAU_cv.py` and `keras_utils_cv.py` to edit things (rename imports).

Approach and explanation:

- The hyperparameter lists are defined, which will be the number of nodes of the AttentionLSTM layer and its dropout, and the number of filters of the two Conv1D layers.
- The network architecture is configured, this function would take as arguments the names of the lists within the parameters on which to iterate.
- In the main of modelAU we do a for on each parameter to adjust and inside the lowest level loop:
  - We call the `generate_model()` function.
  - We call to the `train_model()` function of keras_utils.

- About keras_utils:
  - In `train_model()` we add as input arguments the hyperparameters. This function will be the one in which the cross validation tests are made.
    - We take `X` and `y` from the dataset; we remove the _train suffix because we reserve that to do it inside the KFold.
    - All calls to `X_train` and `y_train` in this function are changed to `X` and `y`.
    - After `compile.model` we define the `KFold()` object.
    - We iterate with enumerate on the `KFold` object and inside this loop we do use the names of `X_train` and `y_train`. The `fit` is done and the model is evaluated with each fold. We save the results in **`results_train_cv.txt`** with dataset_id, n_fold, accuracy, filters1, filters2, nodes1, dropout1.
  - We define `train_selected_model()`, which does what the original `train_model()` did: it takes the dataset, separates into train and test, and trains the model. 
    - We add a return model at the end to be able to access the trained model from the main of modelAU.

- Out of the hyperparameters loop we load the results txt as pandas dataframe.
- For the dataset we do the average of the accuracy of each fold and select the row with the best set of hyperparameters.
- We call `generate_model()`, passing as parameters the best set chosen.
- We call `train_selected_model()`, pass it the model from the previous line and store the trained model it returns in the model variable.
- Finally, we call `evaluate_model()` passing as input the trained model.
- The `evaluate_model()` stores in **`results_cv.txt`** the id of the dataset, the accuracy and the execution time, which starts counting from the time the model with the best set of parameters is generated until the evaluation is finished.
- To do several iterations (only with the best model) we add a for before the `generate_model()` that goes with the selected hyperparameters and we make sure that the `start_time` is inside this for to consider the time to generate the model, train it and evaluate it.
- For each test we only have to edit the value of `DATASET_INDEX`.



### `seq2Var_rf.py`

- The necessary packages for seq2var are imported.
- The `modelAU.py` packages are imported. 
- seq2var code:
  - Path to the dataset and load the .npy.
  - We put the data as a torch tensor, prior redimensionalization to add a time dimension for `X_train` and `X_test`. For example, for ECG X_train has shape (100, 2, 147) and after the torch.Tensor it is torch.Size([100, 2, 147, 1]).
  - With the `argument_parser()` several parameters are prepared. The `args.sd = 0 ` can be used to control the standard deviation or the addition of noise. The `args.nb_systems` takes the number of dimensions of the training set. The `args.timesteps` takes the number of timesteps in the training set. The `args.num_atoms` also takes the number of dimensions in the training set.
  - You create `TensorDataset()` objects and prepare to iterate over them with `DataLoader()`.
  - Seq2Var:
    - The timer starts.
    - We call `argument_parser_seq2var()` and set its number of epochs.
    - We use the `RelationalEncoder(nn, Module)` of `modules.py` passing as arguments the `args.timesteps` defined before, the `args_seq2var.encoder_hidden()` which defaults to 64 and the `args.lags` of the `argument _parser()` which defaults to 1 (this is in argument_parser_permutations.py).
    - As the `args.cuda` defaults to True, the encoder_seq2var takes the value of the .cuda() that comes out of the `RelationalEncoder()`.
    - The optimizer is Adam with default value of argument_parser_seq2var(), which is 1e-3.
    - We call the `train_seq2var()` function of `/seq2Var/utils.py`. This function takes the PyTorch DataLoader (train_data_loader), the encoder (encoder_seq2var), the optimizer, scheduler, args and args_seq2var and binary as False to specify that the output is not binary. What the function does is to put the encoder in training mode (so it can be evaluated later).
  - Binary Seq2Var:
    - Torch tensors and a `RelationalEncoder` are prepared as before.
    - The `train_seq2var()` function is called, this time to train the encoder_bseq2var.
  - NRI: we don't, a Decoder was missing in utils.py.
  - Performances:
    - empty lists are created for various `A_var`,` A_seq2var` and `A_bseq2var` arrays and labels, `l_test`.
    - We put the encoders in evaluation mode with `.eval()`.
    - Iterate over the test_data_loader with each data-label pair: 
      - Adds the label to the `l_test` list.
      - Extracts the input data as `X`.
      - VAR:
        - For each sequence `d` in the data, fits a VAR model and stores the coefficients in `A_vars`.
      - Seq2VAR and bSeq2VAR:
        - Initializes an off-diagonal `off_diag` matrix and encodes it in a one-hot encoding for the sender and receiver indices (`rel_rec` and `rel_send`).
        - Calculate the adjacency matrix using the respective encoders (`encoder_seq2var` and `encoder_bseq2var`) and convert it to a PyTorch tensor.
        - For bSeq2VAR, binarize the adjacency matrix by applying a threshold.
    - Convert the obtained adjacency matrices and labels into PyTorch tensors.
    - For each method (`A_vars`, `A_seq2vars`, `A_bseq2vars`), perform the following steps:
      - Split the data into training sets and test using `train_test_split`.
      - Trains a RandomForest (`rf`) with train data.
      - Compute classifier accuracy with test data.
      - Prints the method name and accuracy.



### `seq2Var_rf_cv.py`

- To apply the cross validation in the last item of the previous steps (the "For each method (`A_vars`, `A_seq2vars`, `A_bseq2vars`), perform the following steps") we first define the parameter grid for the RandomForest and define a list with the three methods to be tested (Var, Seq2Var, bSeq2Var).

- A few steps before, in the VAR part, it is possible that with some dataset the following error appears: 

   File "seq2Var_rf_cv.py", line 145, in <module> results = model.fit(1, trend='c') File "/home/antonio/.pyenv/versions/TSC/lib/python3.8/site-packages/statsmodels/tsa/vector_ar/var_model. py", line 696, in fit return self._estimate_var(lags, trend=trend) File "/home/antonio/.pyenv/versions/TSC/lib/python3.8/site-packages/statsmodels/tsa/vector_ar/var_model.py", line 717, in _estimate_var z = util. get_var_endog(endog, lags, trend=trend, has_constant="raise") File "/home/antonio/.pyenv/versions/TSC/lib/python3.8/site-packages/statsmodels/tsa/vector_ar/util.py", line 35, in get_var_endog Z = tsa. add_trend(Z, prepend=True, trend=trend, File "/home/antonio/.pyenv/versions/TSC/lib/python3.8/site-packages/statsmodels/tsa/tsatools.py", line 152, in add_trend raise ValueError(msg) ValueError: x contains one or more constant columns. Columns 33, 45 are constants. Adding a constant with trend='c' is not allowed.

   This refers to the function in ".pyenv/versions/TSC/lib/python3.8/site-packages/statsmodels/tsa/vector_ar/var_model. py", in which we see the values that the trend parameter can take: trend : str {"c", "ct", "ctt", "n"}, "c" - add constant, "ct" - constant and trend, "ctt" - constant, linear and quadratic trend, "n" - co constant, no trend

   Changing the trend='c' to trend='n' of model.fit() should solve the problem.

- Then we do the fit on the GridSearchCV and once finished we do 3 repetitions of training the RandomForestClassifier for each of the three methods, we save the accuracy and the parameters selected by the cross validation (all in the same file).

- The results are stored in **`results_seq2var_rf_cv.txt`** and are the id of the dataset, accuracy of var, seq2var and bseq2var, the runtime and the selected RandomForest hyperparameters.

- Neither in this code nor in the following code have we calculated the % reduction of the datasets for each method. For the moment this is left pending. In case of doing it, in **`modelAU_seq2var_lstmfcn`** from line 180, which is when the reduced datasets are saved, we would stop the execution and save the time up to that point, and then divide it by two to have the time taken by Seq2Var and bSeq2Var (remember that Var was not trained). To **`seq2Var_rf_cv.py`** we would have to add those lines to save the datasets as well, together with the time counter.



### `modelAU_seq2var_lstmfcn.py`

- We define the number of the dataset (`DATASET_INDEX`) to train and the `MAX_TIMESTEPS`, `MAX_NB_VARIABLES` and `NB_CLASS` and load the train and test set.
- The process of `seq2var_rf.py` is done.
- Save reduced datasets. For this you have to manually create a folder for each method (Var, Seq2Var, bSeq2Var) inside /"dataset"_Seq2Var. The dataset folder has to be edited each time we change dataset. I forgot to do the latter, so the datasets reduced by these methods are not saved, only those of the last test that was done in the corresponding folder.
- Functions of **`modelAU.py`** to generate the model.
- The results of **`modelAU_cv.py`** are loaded to use the hyperparameters of the network selected for each dataset in previous trainings.
- We call the main. Before we define a `mid_time` to store the time of the seq2var process and then, inside the loop for the two repetitions we start another timer.
- When generating the model we must specify that the number of filters and nodes are integers.
- In the `evaluate_model` we add the `mid_time` and `methods` parameters, and for this we also modify the **`keras_utils_seq2var.py`**, to calculate the `executionTime` correctly and include in the output file the name of the method to which the results correspond.
- The output file is **`results_seq2var_lstm_fcn.txt`** and carries the dataset id, accuracy, execution time and reduction method.



### `rf_raw.py`

- We import `utils.new_constants` and the necessary functions.
- We load train and test with `load_dataset_at()`, except with the *AUSLAN* dataset that for some reason (it gives a ValueError: Unknown label type: 'continuous') has to be loaded accessing the .npy by relative path.
- The hyperparameter grid is defined for the `RandomForestClassifier()`.
- The search is performed with `GridSearchCV()`.
- Three iterations of the `fit` of the RandomForest are made, saving in **`results_rf.txt`** the dataset_id, accuracy, runtime and hyperparameter set.
- To analyze the results in the **`results_to_df.ipynb`** notebook we load the .txt, select the dataset we are working with, average the accuracy and time in the three repetitions and calculate the standard deviation of the accuracy.





