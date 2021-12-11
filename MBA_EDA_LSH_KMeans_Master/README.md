# Market Basket Analysis Master

This is the demo of python script  

## Getting started

### 0. Please install anaconda 4.10.3 at first, the python version is 3.9.7
#### Note: when need to run the script on the cloud, you may use the latest Minicoda which is lighter than anaconda.

### 1. create virtual environment
```
conda create --name myenv
```

### 2. activate environment
```
conda activate myenv
```

### 3. Install packages in the environment using requirements.txt
```
conda install --file requirements.txt
```

### 4. Execute the scripts one-by-one
#### 4.1 Run MBA_EDA_Features.py to get the features and LSH results
```
python MBA_EDA_Features.py
```

#### 4.2 Run MBA_Optimal_K.py to get the optimal K for KMeans (for the dataset of instacart, we suggest Integer1:10 Integer2:100 Integer3:10)
```
python MBA_Optimal_K.py Integer1 Integer2 Integer3
```
For help:
```
python MBA_Optimal_K.py -h
```

#### 4.3 Run MBA_KMeans.py to get the KMeans results (for the dataset of instacart, we suggest Integer:40)
```
python MBA_KMeans.py Integer
```
For help:
```
python MBA_KMeans.py -h
```
