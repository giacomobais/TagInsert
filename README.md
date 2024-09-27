## General info

This repository contains the code for the TagInsert model, which is a variant of the EncoderDecoder model that allows for arbitrary ordering of generation for tagging tasks.

The TagInsert model is described in the Thesis "Breaking left-to-right generation in Transformer models: arbitrary ordering on tagging tasks" included in the repository.

Additionaly, the repository contains the code for the VanillaEncoderDecoderTransformer model, which is the original Trasnformer model developed by Vaswani et al. (2017); the TagInsertL2R model, which is the TagInsert model with a forced left-to-right generation order; and the FineTuned DistilBERT model, which is a simple finetuning of DistilBERT for the tagging task proposed in the thesis.

The scripts that come with each model allow for training the model from scratch; using a pre-trained model as a starting point and evaluating the model on a dataset. The tagging tasks that were experimented with are POS tagging from the Penn Treebank; CCG non-constructive tagging and CCG constructive supertagging from the CCGBank and Rebank corpora. The dataset are not included in the repository, but the data loading code for the models was implemented in a way that allows for easy integration with new datasets. Moreover, a dummy dataset is provided to showcase the way datasets should be structured to work with the models.

A config file is provided for each model to showcase the different parameters that can be tuned. The file can be easily modified to try out different configurations.

## Training and evaluation

Both the train.py and evaluation.py scripts contain a main() function that can be used to run the models. The preprocess.py script contains the code for the data preprocessing pipeline used for the models, and is used by the train.py script to load the data.

The models trained will be automatically saved in the saved_models folder contained in each model's folder. When running the evaluation script of a model, the script will check if a saved model exists. If it does not, it will run the evaluation on an untrained model. Otherwise, it will load the saved model and evaluate it.

The predictions of the models on the dummy dataset can be found in the predictions folder. Predictions are saved in a file named sample_predictions.txt in the predictions folder of each model. In the case of the TagInsert model, the predictions also contain the ordering used during generation. The orderings should be read so that each number corresponds to the order of the POS tag with the same index. For example, [3,2,1] means that the first POS tag of the sentence was the third to be predicted, the second POS tag was the second to be predicted and the third POS tag was the first to be predicted.

## Acknowledgements

The Encoder-Decoder modules of each models are taken from the Annotated Transformer, by Harvard NLP: https://nlp.seas.harvard.edu/annotated-transformer/

The idea for the TagInsert model was inspired by the paper "Insertion Transformer: Flexible Sequence Generation via Insertion Operations" by Stern et al. (2019).

The code for the FineTuned DistilBERT model is based on the HuggingFace tutorial for finetuning BERT models



