# ResMini Reimplementation

I tried to reimplement the model introduced in "A ResNet mini architecture for brain age prediction" using PyTorch. Results were not exactly reproducible, and the training and validation curves were not the same as presented in the paper, but the validation accuracy was close to the original.

The dataset used was the same as in the original paper, linked: https://openneuro.org/datasets/ds000228/versions/1.1.0. I only used middle axial slices from the T1 weighted .gz files. Ages were obtained from the participants.tsv file. The final data files used are provided in the "data" folder of the repository.
