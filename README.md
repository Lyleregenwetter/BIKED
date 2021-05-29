BIKED
=======
This dataset and code are presented in the paper: [BIKED: A Datset and Machine Learning Benchmarks for Data-Driven Bicycle Design](https://arxiv.org/abs/2103.05844)


Also check out the [BIKED project page](http://decode.mit.edu/projects/biked/).

## License
This code is licensed under the MIT license. Feel free to use all or portions for research or related projects as long as you cite the paper:

Regenwetter, L., Curry, B., and Ahmed, F., 2021. “Biked: A dataset and machine learning benchmark for data-driven bicycle design”.



## Required packages
- tensorflow > 2.0.0
- tensorflow-probability
- sklearn
- matplotlib
- seaborn
- pathlib
- shap
- opencv

Pip doesn't seem to work for [SHAP](https://github.com/slundberg/shap), so I recommend conda. Use either pip or conda for the other packages.
## Data

### Numerical Data

**Data/BIKED_processed.csv** contains the final processed data after miscellaneous processing steps (Sections 3.1-3.6 in the paper). This data is a csv file containing 4512 models and 2395 parameters. The first column of the csv file contains the model numbers and the first row contains the parameter names. To load the dataset into a Pandas DataFrame, use the following code. This dataframe will have model numbers as indices and parameters as column names.

```bash
import Pandas as pd
df=pd.read_csv("data/BIKED_processed.csv", index_col=0)
```

### Image Data

1. **Images** contains raw image exports after removal of dimensions and dimensional labels (Section 3.3 in the paper) Warning: Images and processed parametric data do not contain the exact same set of models

2. **Standardized Images** contains standardized image exports after reexporting images from the processed BIKED data (Section 3.7 in the paper)

3. **Segmented bike images** contains all image segmentation data (Section 3.3 in the paper) (To be updated to match model numbering)

<img src="/flow.png" width="400">

## Example Applications

### Unsupervised Methods
See **Functions/unsupervised.py** for the code used the t-SNE and PCA embeddings

<img src="/tsnelabeled.png" width="600">

### Parametric Classification Comparison

See **Functions/Classification.ipynb** for the comparison of many classifiers with different train sizes.

### Tensorflow Deep Classifiers

See **Functions/TF Classifier.ipynb** for comparison of Deep classifiers using Images, Parametric Data, and a Combination of the two. The SHAP analysis can also be found in **Functions/TF Classifier.ipynb**. The image data will need to be preprocessed first:

1. Navigate to functions directory:

   ```bash
   cd Functions
   ```

2. Run **preprocessImages.py**:

   ```bash
   python preprocessImages.py
   ```

<img src="/shapfinal.png" width="800">

### Generation Examples
See **Functions/VAEv2.ipynb** for VAE implementations.

<img src="/6gen.png" width="400">


## Other Numerical Data

### Other Preprepared Data
In case you want to use BIKED but require different processing steps than the generic approach, two other versions of biked are provided besides the standard BIKED_processed data.

1. **BIKED_raw.csv** contains the raw parameter space of the data after File Standardization (Section 3.2 in the paper). This data contains 4791 models and 23813 parameters. Due to file size constraints, this raw data is not included in the repo, but can be downloaded from [Dropbox](https://www.dropbox.com/sh/b5y25zdjq9q0890/AABULjSD9ZmK-bmuDwoWua7Ba?dl=0).

2. **BIKED_reduced.csv** contains the reduced parameter space of the data after parameter space reduction (Section 3.5 in the paper). This data contains 4512 models and 1320 parameters.


### Custom Processing Steps

Alternatively, BIKED can be processed using custom processing steps. The functionality to process and curate the dataset is included in the **Functions/prepareData.py** which calls a set of helper functions contained within **Functions/dataFrameTools.py** and **Functions/paramRedux.py**. If custom data processing steps are desired, these functions can be used as a starting point. Currently the dataset is processed (or reprocessed) using the BIKED_raw as a starting point (Reminder: This needs to be downloaded from the [Dropbox](https://www.dropbox.com/sh/b5y25zdjq9q0890/AABULjSD9ZmK-bmuDwoWua7Ba?dl=0) and moved to the Data/ folder.) To reprocess the dataset:

1. Navigate to functions directory:

   ```bash
   cd Functions
   ```

2. Run **prepareData.py**:

   ```bash
   python prepareData.py
   ```

This will generate several more csv files in the **Data/** folder that may potentially be valuable. **BIKED_datatypes.csv** and **BIKED_processed_datatypes.csv** contain lookup tables of parameter types of BIKED_reduced and BIKED_processed. **OHdf.csv**, contains the data after one hot encoding but before imputation. **meddf.csv** Contains a lookup table of indices of the median bikes of every class. **BIKED_normalized.csv** contains a min-max normalized version of BIKED_processed. **classdf.csv** contains a dataframe of the median values for each parameter for each class of bike using the same min-max normalization as **BIKED_normalized.csv**.

## Generating BikeCAD files from parametric data
The functionality to generate synthesized BikeCAD files is provided in **processGen.py** in the Functions folder, which calls functions from **dataFrameTools.py**. ProcessGen will process a file from the /Data folder, **synthesized.csv** to be generated by the user. This file is assumed to have the same column structure as the processed or normalized data (**BIKED_processed.csv**/**BIKED_normalized.csv**) with the same column labels but different indices and data. Additionally, generated data is expected to have the same scaling as **BIKED_normalized.csv**. An example **Data/synthsized.csv** file is included with 4 sample synthesized models. Synthesized models will be generated in **/Generated BCAD files/Files/**. BikeCAD files can be opened in the BikeCAD software to export images.

1. Navigate to functions directory:

   ```bash
   cd Functions
   ```
2. Run **processGen.py**:

   ```bash
   python processGen.py
   ```

**processGen.py** will also generate two intermediate csv files for analysis: **synthesized_DeOH** contains the dataframe of synthesized parametric data after reversing the min-max scaling applied to the original data. **synthesized_DeOH** contains the data after reversing the one-hot encoding. This is the exact parametric data that is inserted into the template .bcad files for BCAD file generation.
