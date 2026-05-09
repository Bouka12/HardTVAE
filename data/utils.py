"""
This  module contains utility functions for minimal preprocessing of the data
such as :
    * fix obvious erroneous tokens, 
    * unify missing value markers, 
    * remove exact duplicates, 
    * drop columns that are entirely metadata or constant

Each dataset has its own preprocessing function that calls the utility functions in this module.
Each dataset-specific minimal preprocessing is supported with refernces from the oiginal paper and subsequent research on the same dataset.
Each step is well documented with the rational behind it.
15/04/2026 11.26 PM

this module will be done tomorrow InshAllah, since the steps are already done in separate notebooks.
"""
import numpy as np
import pandas as pd
import os
from sklearn  import preprocessing
from ucimlrepo import fetch_ucirepo 

# print(__file__)
# # for dirname, _, filenames in os.walk('./data'):
# #     for filename in filenames:
# #         print(os.path.join(dirname, filename))
# # print(os.path.dirname(__file__))
# print(os.path.abspath(__file__))


DATA_SOURCE_URL = {
    # uci repo datasets urls:
    "BCWDD": "https://archive.ics.uci.edu/dataset/45/heart+disease", # 1
    "HeartCleveland": "https://archive.ics.uci.edu/dataset/45/heart+disease", # 2
    "Hepatitis": "https://archive.ics.uci.edu/dataset/102/thyroid+disease", # 3
    "ILPD": "https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset", # 5
    "Thoracic": "https://archive.ics.uci.edu/dataset/277/thoracic+surgery+data", # https://archive.ics.uci.edu/dataset/277/thoracic+surgery+data
    "Vertebral": "https://archive.ics.uci.edu/dataset/212/vertebral+column",
    # keel datasets urls:
    "Pima": "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/pima.zip", # 8 
    "NewThyroid1": "https://sci2s.ugr.es/keel/dataset/data/imbalanced/new-thyroid1.zip", # 9
    "NewThyroid2": "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/new-thyroid2.zip", # 10
    # kaggle datasets urls:    
    "Hypothyroid": "https://www.kaggle.com/datasets/yasserhessein/thyroid-disease-data-set" # 4 # 
} 

# UCI REPO DATASETS IDS:
UCIREPO_ID = {
    "BCWDD": 17,
    "HeartDisease": 45,
    "Hepatitis": 46,
    "ILPD": 225,
    "ThoracicSurgery": 277, 
    "Vertebral": 212
}

# KEEL DATASETS URLS:
KEEL_URL = {
    "Pima": "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/pima.zip",
    "NewThyroid1": "https://sci2s.ugr.es/keel/dataset/data/imbalanced/new-thyroid1.zip",
    "NewThyroid2": "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/new-thyroid2.zip"

}

# KAGGLE DATASETS URLS:
KAGGLE_URL = {
    "Hypothyroid": "https://www.kaggle.com/datasets/yasserhessein/thyroid-disease-data-set"
}



def get_BCWDD():
    """ BCWDD dataset"""

    # 0. Upload: from ucimlrepo using fetch_ucirepo
    ## fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    ## data (as pandas dataframe)
    X = breast_cancer_wisconsin_diagnostic.data.features
    print(X.head())
    print(X.columns)

    y = breast_cancer_wisconsin_diagnostic.data.targets
    print(y.head())
    print(y.columns)
    ## metadata
    print(breast_cancer_wisconsin_diagnostic.metadata)
    ## variable information
    print(breast_cancer_wisconsin_diagnostic.variables)

    # 1.  Replace: "B" by 0 and "M" by 1 in the column "Diagnosis"
    print(y['Diagnosis'].value_counts())
    y['Diagnosis'].replace({"B": 0, "M": 1}, inplace=True)
    print(y['Diagnosis'].value_counts())
    # 4. rename the column "diagnosis" to "Outcome"
    y = y.rename(columns={"Diagnosis": "Outcome"})
    print(y.head())

    # 5. save the preprocessed dataframe to a new csv file in the processed data directory
    # concatenate X and  to a single preprocessed dataframe
    preprocessed_df = pd.concat([X, y], axis=1)
    print(preprocessed_df.head())
    print(preprocessed_df.columns)
    # save the preprocessed data in ./data/preprossed
    # preprocessed_df.to_csv("./data/processed/BCWDD.csv", index=False)
    return preprocessed_df

#***********************************************************************************************#
def get_HeartCleveland():
    """ HeartCleveland dataset:"""
    # 0. Upload: from ucimlrepo using fetch_ucirepo
    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45)  # (303)
    
    # data (as pandas dataframes) 
    X = heart_disease.data.features 
    print(f"size of the data: {X.shape}")
    print(X.head()) 
    y = heart_disease.data.targets 
    print(y.head())
    # metadata 
    print(heart_disease.metadata) 
    
    # variable information 
    print(heart_disease.variables) 

    # 1. Replace the column "num" to "Outcome" 
    print(y.value_counts())
    # 2. Replace 0 by 0 and 1, 2, 3, 4 by 1 in the column "num"
    y['num'].replace({0: 0, 1: 1, 2: 1, 3: 1, 4: 1}, inplace=True)
    print(y.value_counts())
    y = y.rename(columns={"num": "Outcome"})

    # 3. check the values used to describe NaN values : features with NaN : "thal" and "ca"
    print(X['thal'].value_counts())
    print(X['ca'].value_counts())
    # No other preprocessing steps => Save the preprocessed dataframe
    # concat the X and y
    preprocessed_df = pd.concat([X, y], axis=1)
    print(preprocessed_df.head())
    # preprocessed_df.to_csv("./data/processed/HeartCleveland.csv", index=False)
    return preprocessed_df

#***********************************************************************************************#   
def get_Hepatitis():
    """ Hepatitis dataset:"""
    # 0. Upload: from ucimlrepo using fetch_ucirepo
    # fetch dataset
    hepatitis = fetch_ucirepo(id=46) 
    
    # data (as pandas dataframes) 
    X = hepatitis.data.features 
    y = hepatitis.data.targets 
    
    # metadata 
    print(hepatitis.metadata) 
    
    # variable information 
    print(hepatitis.variables) 

    # 1. Replace  "live" by 0 and "die" by 1 in the column "class"
    print(y['Class'].value_counts())
    y['Class'].replace(2, 0, inplace=True)
    # 2. rename the column "class" to "Outcome"
    y = y.rename(columns={"Class": "Outcome"})
    print(y['Outcome'].value_counts())

    # It has missing values
    # print(X['Varices'].value_counts())

    # The following steps should be done in the preprocessing as we need to identify the categorical variables and the numerical in the preprocessing
    # 3. Replace male by 0 and female by 1 in the column "sex"

    # 4. Replace True by 1 and False by 0 in the columns of categorical variables:
    ## ['steroid', 'fatigue', 'malaise', 'anorexia', 'liver_big',
    #  'liver_firm', 'spleen_palpable', 'spiders', 'ascites', 'varices']

    # 5. Save the preprocessed dataframe to a new csv file in the processed data directory
    preprocessed_df = pd.concat([X, y], axis=1)
    print(preprocessed_df.head())
    # preprocessed_df.to_csv("./data/processed/Hepatitis.csv", index=False)
    return preprocessed_df

#***********************************************************************************************#
def get_ILPD():
    """ ILPD dataset"""
    # 0. Upload: from ucimlrepo using fetch_ucirepo
    
    # fetch dataset 
    ilpd_indian_liver_patient_dataset = fetch_ucirepo(id=225) 
    
    # data (as pandas dataframes) 
    X = ilpd_indian_liver_patient_dataset.data.features 
    y = ilpd_indian_liver_patient_dataset.data.targets 

    print(X.head())
    print(y.head())  
    # metadata 
    print(ilpd_indian_liver_patient_dataset.metadata) 

    # variable information 
    print(ilpd_indian_liver_patient_dataset.variables)

    # 1. Replace 1 by 0 and 2 by 1 in the column "Outcome"
    print(y['Selector'].value_counts())

    y['Selector'].replace({1:0, 2:1}, inplace=True)
    print(y['Selector'].value_counts())
    y = y.rename(columns = {"Selector": "Outcome"})
    print(y.head())

    # Save the preprocessed data in ./data/processed/
    preprocessed_df = pd.concat([X,y], axis=1)
    print(preprocessed_df.head())
    # preprocessed_df.to_csv("./data/processed/ILPD.csv", index=False)
    return preprocessed_df


#***********************************************************************************************#
def get_Thoracic():
    """ Thorcacic Surgery dataset:"""
    # 0. Upload: from ucimlrepo using fetch_ucirepo
    
    # fetch dataset 
    thoracic_surgery_data = fetch_ucirepo(id=277) 
    
    # data (as pandas dataframes) 
    X = thoracic_surgery_data.data.features 
    y = thoracic_surgery_data.data.targets 

    print(X.head())
    print(y.head())  
    # metadata 
    print(thoracic_surgery_data.metadata) 
    
    # variable information 
    print(thoracic_surgery_data.variables) 

    # 2. Rename the column "Risk1Yr" to "Outcome"
    print(y.value_counts())
    y['Risk1Yr'].replace({"F":0, "T":1}, inplace=True )
    print(y.value_counts())
    y.rename(columns={'Risk1Yr': 'Outcome'}, inplace=True)
    print(y.head())
    # 3. Save the preprocessed dataframe.
    preprocessed_df = pd.concat([X, y], axis=1)
    print(preprocessed_df.head())
    # preprocessed_df.to_csv("./data/processed/Thoracic.csv", index=False)
    return preprocessed_df


#***********************************************************************************************#
###### Vertebral Column dataset:
def get_Vertebral():
    """    ###### Vertebral Column dataset:"""
    # 0. Upload: from ucimlrepo using fetch_ucirepo
    
    # fetch dataset 
    vertebral_column = fetch_ucirepo(id=212) 
    
    # data (as pandas dataframes) 
    X = vertebral_column.data.features 
    y = vertebral_column.data.targets 
    print(X.head())
    print(y.head())

    # metadata 
    print(vertebral_column.metadata) 
    
    # variable information 
    print(vertebral_column.variables) 

    # 1. Replace "class" by "Outcome"
    print(y.value_counts())
    y['class'].replace({'Spondylolisthesis':0, 'Hernia':0, 'Normal':1}, inplace=True)
    print(y.value_counts())

    # 2. Replace "AB" by 1 and "NO" by 0 in the column "Outcome"
    y.rename(columns = {'class':'Outcome'}, inplace=True)
    print(y.head())
    # 3. Save the preprocessed dataframe
    preprocessed_df = pd.concat([X,y], axis=1)
    # preprocessed_df.to_csv("./data/processed/Vertebral.csv", index=False)
    return preprocessed_df




#***********************************************************************************************#
# Hypothyroid from Kaggle:
### Kaggle API token:
# KGAT_75fe0ea0a1e853cf36fd75d727d519cd
# API TOKEN

# To use this token, set the KAGGLE_API_TOKEN environment variable:

# export KAGGLE_API_TOKEN=KGAT_75fe0ea0a1e853cf36fd75d727d519cd
# After setting KAGGLE_API_TOKEN, you can use the client as follows:

import kaggle
# # kaggle competitions list
def get_Hypothyroid():
    """Hypothyroid from Kaggle:"""

    kaggle.api.dataset_download_files(
        'yasserhessein/thyroid-disease-data-set',
        path='./data/thyroid_data-kaggle',
        unzip=True
    )

    import pandas as pd

    df = pd.read_csv("./data/thyroid_data-kaggle/hypothyroid/hypothyroid.csv", na_values=["?"])
    print(df.head())
    print(df.shape)


    # Preprocessing steps:
    # 1. Replace "P" by 1 and "N" by 0 in the column "binaryClass"
    print(df['binaryClass'].value_counts())
    df['binaryClass'].replace({'P':1, 'N':0}, inplace=True)
    print(df['binaryClass'].value_counts())

    # 2. Rename the column "class" to "Outcome"
    df.rename(columns={'binaryClass':'Outcome'}, inplace=True)
    print(df.head())

    # 3. Remove the 'TBG' and 'TBG measured column it is all nan values
    df.drop(['TBG', 'TBG measured'], axis=1, inplace=True)
    # 4. Remove the categorical encoding of the variables: 
    ## "FTI measured" for "FIT", "TT4 measured" for "TT4", "T4U measured" for "T4U", "T3 measured" for "T3"
    ## "TSH measured" for "TSH"
    df.drop(["FTI measured", "TT4 measured", "T4U measured", "T3 measured", "TSH measured"], axis=1, inplace=True)
    # 3. Save the preprocessed dataframe to a new csv file in the processed data directory
    df.to_csv(r"C:\Users\MABROUKAs\Downloads\IEEE-Access-2026\HardVAE\data\processed\Hypothyroid.csv", index=False)
    return df

df = get_Hypothyroid()
print(df.head())
print(df.info())
print(df.columns)
#************************************************************************************************#
####### Pima dataset:

import pandas as pd
import requests
import io
import zipfile

def load_keel_dat(path):
    attributes = []
    data_started = False
    data_lines = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # Extract attribute names
            if line.lower().startswith("@attribute"):
                # Format: @attribute Name type [range]  OR  @attribute Class {positive, negative}
                parts = line.split()
                attr_name = parts[1]
                attributes.append(attr_name)

            # Detect start of data
            elif line.lower().startswith("@data"):
                data_started = True
                continue

            # Collect data rows
            elif data_started and line:
                data_lines.append(line)

    # Convert to DataFrame
    df = pd.DataFrame(
        [row.split(",") for row in data_lines],
        columns=attributes
    )

    # Strip whitespace from values
    df = df.applymap(lambda x: x.strip())

    return df




# #### Pima dataset:
def get_Pima():
    url = "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/pima.zip"

    # Download the zip file
    response = requests.get(url)
    response.raise_for_status()  # ensure download success

    # Extract zip content in memory
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("./data/pima_dataset")  # folder where files will be extracted

    print("Files extracted to pima_dataset/")

    # # import pandas as pd

    file_path = "./data/pima_dataset/pima.dat"

    df = load_keel_dat(file_path)
    print(df.head())
    print(df.columns)

    print(df['Class'].value_counts())
    # The target variable is "Class": map its values {'positive':1, 'negative':0}

    df['Class'].replace({'positive':1, 'negative':0}, inplace=True)
    print(df['Class'].value_counts())


    # # Change values of 0 in variables : "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
    # # let's do a mapping to apply it to those variables:
    cols = ['Preg', 'Plas', 'Pres', 'Skin', 'Insu', 'Mass', 'Pedi', 'Age']
    int_cols = ['Preg', 'Plas', 'Skin', 'Insu']
    float_cols = ['Mass', 'Pedi']
    for col in int_cols:
        # convert to numeric 
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].astype(float).astype('Int64')

    for col in float_cols:
        df[col] = df[col].astype(str).str.strip().astype(float)

    # Columns where 0 MUST be NaN
    cols_to_fix = ['Plas', 'Pres', 'Skin', 'Insu', 'Mass']
    print(df.isnull().sum())

    for col in cols_to_fix:
        df.loc[df[col]==0, col] = np.nan


    print(df.head())
    print(df.info())

    # ## check nan values now:
    print(df.isnull().sum())

    df.rename(columns = {"Class":"Outcome"}, inplace=True)

    # # Save the `Pima` df as csv in ./preprocessed
    # df.to_csv('./data/processed/Pima.csv', index=False)
    return df

# #### New Thyroid 1 dataset:
def get_NewThyroid1():
    url = "https://sci2s.ugr.es/keel/dataset/data/imbalanced/new-thyroid1.zip"
    # Download the zip file
    response = requests.get(url)
    response.raise_for_status()  # ensure download success

    # Extract zip content in memory
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("./data/new_thyroid1_dataset")  # folder where files will be extracted

    print("Files extracted to new_thyroid1_dataset/")

    import pandas as pd

    file_path = "./data/new_thyroid1_dataset/new-thyroid1.dat"

    df = load_keel_dat(file_path)
    print(df.head())
    print(df.columns)
    print(df.info())

    ## 1. Map `Class` values {"negative":0, "positive":1}
    df['Class'].replace({"negative": 0, "positive": 1}, inplace=True)
    ## 2. Rename `Class` as `Outcome`
    df.rename(columns = {"Class":"Outcome"}, inplace=True)


    # All columns are of type `object`:
    # columns of value `Int`: "T3resin"
    int_cols = ["T3resin"]
    # columns of value `float`: "Thyroxin", "Triiodothyronine", "Thyroidstimulating", "TSH_value"
    float_cols = [ "Thyroxin", "Triiodothyronine", "Thyroidstimulating", "TSH_value"]
    for col in int_cols:
        # convert to numeric 
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].astype(float).astype('Int64')

    for col in float_cols:
        df[col] = df[col].astype(str).str.strip().astype(float)
    print(df.head())
    print(df.info())
    # save the preprocessed new thyroid 1 df
    # df.to_csv("./data/processed/NewThyroid1.csv", index=False)
    return df


### New Thyroid 2 dataset:
def get_NewThryoid2():
    url = "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/new-thyroid2.zip"
    # https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/imb_IRlowerThan9/new-thyroid2.zip
    # Download the zip file
    response = requests.get(url)
    response.raise_for_status()  # ensure download success

    # Extract zip content in memory
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("./data/new_thyroid2_dataset")  # folder where files will be extracted

    print("Files extracted to new_thyroid2_dataset/")

    import pandas as pd

    file_path = "./data/new_thyroid2_dataset/newthyroid2.dat"

    df = load_keel_dat(file_path)
    print(df.head())
    print(df.columns)

    print(df.info())
    ## 1. Map `Class` values {"negative":0, "positive":1}
    df['Class'].replace({"negative": 0, "positive": 1}, inplace=True)
    ## 2. Rename `Class` as `Outcome`
    df.rename(columns = {"Class":"Outcome"}, inplace=True)


    # All columns are of type `object`:
    # columns of value `Int`: "T3resin"
    int_cols = ["T3resin"]
    # columns of value `float`: "Thyroxin", "Triiodothyronine", "Thyroidstimulating", "TSH_value"
    float_cols = [ "Thyroxin", "Triiodothyronine", "Thyroidstimulating", "TSH_value"]
    for col in int_cols:
        # convert to numeric 
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].astype(float).astype('Int64')

    for col in float_cols:
        df[col] = df[col].astype(str).str.strip().astype(float)
    print(df.head())
    print(df.info())

    # Save preprocessed NewThyroid2 df as csv file
    # df.to_csv("./data/processed/NewThyroid2.csv", index=False)
    return df

########################
########################
########################
## functions for getting the data from different sources
### 1. retrieval function mapping the name of the dataset of the source `url`
### 2. loading function for each function 

def get_data(dataset_name:str):
    """Get the data from its source (UCI, KEEL, Kaggle) with minimal preprocessing, and return a dataframe of (X,Y)"""

    # 1. Construct the function name string
    func_name = f"get_{dataset_name}"

    # 2. Retrieve the function from the global namespace and call it
    if func_name in globals():
        df = globals()[func_name]()
    else:
        raise AttributeError(f"Function {func_name} has not been defined.")

    return df
