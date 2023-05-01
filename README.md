# multiBEATS

The original .mat data is stored in a separated folder named `data`. Each of the datasets will have their own folder.

## Preparing the multivariate time series, from .mat to .npy

We use the script `generate-dataset.py`. The paths towards the data need to be changed: `conjunto` will have the name of the dataset, for example *ArabicDigist* and `path` is the path to the folder where the data is stored. For example:
conjunto = 'ArabicDigist'
path = r"/home/aurorax/thisrepo/data/original/"+ conjunto+ "/"

With this, we have generated the .npy for the X_train, X_test, y_train, y_test subsets and they are stored in the same folder as the original .mat file.

## Reducing the X_train and X_test sets using multiBEATS




