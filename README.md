# IT_InformationTechnology2024
This repository contains the training and testing files of the paper in the journal IT - Information Technology. In order to refer to content from this repository, please cite the paper as attested here: DOI...
## Data
### Datasets
We provide one datasets that with some manipulation can be used to reproduce our results. It contains the transliterations, i.e. the textual data represented by latin letters, in the first part of each data line and tab-separated are the Part-of-Speech (PoS) label.
+ transliteration_PoS.txt

### Prepatation
For training the data should consist of lines with textual and PoS information, where each part is separated by an empty line. We chose to randomize the order of the data input based on the separation. From this we split the data in five-folds of 80/20% training/test data. Meaning we trained a FLAIR model under the same parameters 5 times, each with a different split of training and testing data. From the 100% of randomized data, fold one used the first 80% for training and the rest for testing, the second used the first 60 and last 20% for training and the range 60-80% for testing. The pattern was repeated for the last three folds. See 'fold_maker.py'.

## Training
As explained in the paper we used the FLAIR toolkit (Version 0.12.2) to train our model, see 'train_flair.py'. We iterated the training over the five folds.

## Rights
This repository is authored by Gustav Ryberg Smidt, Katrien De Graef and Els Lefever. Everything in the repository is under Creative Commons licenses (CC-BY-NC-4.0).

<img src="https://github.com/assyrugent/assets/blob/main/CUNE-IIIF-ORM_logo_1_rounded.png" alt="CUNE-IIIF-FORM logo" title="Logo of the CUNE-IIIF-ORM project" width="150"/><img src="https://github.com/assyrugent/assets/blob/main/AssyrUGent%20logo%20non_caps.png" alt="AssyrUGent logo" title="Logo of the AssyrUGent group" width="150"/>
