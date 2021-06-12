Code for the paper: [Graph Neural Networks for Predicting Chemical Reaction Performance](https://chemrxiv.org/articles/preprint/Graph_Neural_Networks_for_Predicting_Chemical_Reaction_Performance/14589498)
## Installation

1. First install Anaconda. 
2. Create a conda environment with
```
conda create --name rxntorch python=3.6
```
3. Then, activate the new conda environment with
```
conda activate rxntorch
```
4. Install RDKit
```
conda install -c rdkit rdkit 
```
5. Installing PyTorch with a CUDA enabled version
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
6. Install scikit-learn with
```
conda install scikit-learn
```

Finally, clone this repository to your local machine.

## Getting Started


1. Run: 
```
 pip install requirements
```
2. Download the data from the below google drive links and put it in: ./data/<data_name>/raw/

[Suzuki data (su)](http://shorturl.at/qCQ16)

[Doyle data (dy)](http://shorturl.at/eBC59)

[AstraZeneca data (az)](http://shorturl.at/hiBH2)


3. Prepare the domain features (chemical properties) by running 
```
python 01_prepare_data.py --dataset_name <data_name> --use_rdkit_feats <rdkit or no_rdkit> --test_ratio <test_ratio>
```
4. Depending on the type of the features you're using (with rdkit or no rdkit) you can do feature selection using:
```
python 02_train_rf.py --dataset_name <data_name>
```
If you're not using rdkit, you don't have to do feature selection because the feature set is not too large.

5. To train the model, run:
```
python train_yield.py
```
Important arguments:
```
-p: Dataset path
-dn: Dataset name (su,dy,az)
-op: Output path
-o: Output model name
--split_set_num: Which split set to use. This is generated by running 01_prepare_data.py
--use_domain: Use chemical features or not. Options: (rdkit, no_rdkit, no_domain)

--epochs: Number of epochs
--seed: Random seed

--layers: Number of layes
--hidden: Hidden size for all layers
--lr_decay: Learning rate decay
--batch_size: Size of mini-batch
--dropout_rate: Droput rate
```
7. To generate model predictions and visualize the activations of the GNN, run:
```
05_load_model.ipynb
```

## Still working on these:
6. To plot the training curves and get the avg perfroamnce, run:
```
04_plots.ipynb
```



If using chemical features (domain features) you need the json file containg the features. Otherwise, you can just use smiles strings.
