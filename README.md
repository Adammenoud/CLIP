# 

This project is concern in extracting meaningful ecological infromation from citizen science image data.

## Overview
We use contrastive learning with the GPS location as one of the modalities, and images from iNaturalist as the other.
It effectively trains a location encoder, which takes a coordinate (lat, lon) and outputs an embedding that contains ecological information and can be used for downstream tasks.

In order to evaluate these embeddings, we train species distribution models (SDMs) using the generated embeddings as covariates.
The AUC of the SDM gives a way to compare different models, by directly measuring the ecological information contained in the embeddings through the specie modelling task.

Using this metric, we compare the contrasive model learned against other architectures, setups and baselines. 

## Installation

### Virtual environement
You 


### Occurence and multimedia data

The GBIF raw data can be downloaded at the following link:
https://api.gbif.org/v1/occurrence/download/request/0003002-260119153935675.zip

In order for the data to be stored in the right directory, the following commandlines should be run from the project root:
```bash
mkdir -p Data/data_inaturalist &&
wget -P Data/data_inaturalist https://api.gbif.org/v1/occurrence/download/request/0003002-260119153935675.zip
```
and then unzip once the download is finished:
```bash
unzip Data/data_inaturalist/dataset.zip d -Data/data_inaturalist
```
The datasets should then be filtered with the following script:
```bash
python filter_inaturalist.py
```
Note that the final dataframes can be found in "Embeddings_and_dataframes/<dataset_name>/dictionary_inaturalist_FR_<dataset_name>", with <dataset_name> being either plants, mushrooms, or arthropods.

### Downloading embeddings

Since the dataset would be too big to be downloaded if we were to download the images, we download the images, embed them with DINO-v2/Bioclip-2, and keep only the embeddings in a .h5 file.
This process can be done by running the 'data_extraction.py' script: CHECK!!

```bash
python data_extraction.py
```

### Logging in wandb


### Running the models


### Evaluation


### Visualization