# HPI-DHC @ BioASQ DisTEMSIT

## Preparation

- Download DisTEMIST data (v5.0) from Zenodo (https://zenodo.org/record/6532684) and unzip content into `data/distemist`
- Download trained models and dictionaries from Zenodo (https://zenodo.org/record/6642064) and extract into `dicts` and `models`
- Install Python dependencies (*TODO: describe how*)

## Subtrack 1: Named Entity Recognition

### Model Training

The Hydra config file `ner_config.yaml` contains all hyperparameters to reproduce the NER training results.

To train the model on CUDA device `n`, run:

`CUDA_VISIBLE_DEVICES=<n> python scripts/run_ner_training.py`

We performed a hyperparameter grid search over the parameters listed in `ner_hyperparamter_sweep.sh`. 

### Prediction

See: [notebooks/01_Entities.iypnb](notebooks/01_Entities.iypnb)

## Subtrack 2: Entity Linking

See: [notebooks/02_Entity_Linking.iypnb](notebooks/02_Entity_Linking.iypnb)