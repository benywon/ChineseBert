# ChineseBert
This is a chinese Bert model specific for question answering. We provide two models, a large model which is a 16 layer 1024 transformer, and a small model with 8 layer and 512 hidden size. Our implementation is a different from the original paper https://arxiv.org/abs/1810.04805, in which we replace a position embedding with LSTM, which shows advantages when the text length varies a lot.

Currently it is run on python3 and pytorch

-------------------------------------

#Stats:

Data: 200m chinese internet question answering pairs.

tokenizer: we use the [sentencepiece](https://github.com/google/sentencepiece) tokenizer with vocab size equal to 35,000

For both large and small model, we train it for 2m steps, which did not suffer from overfit problem

large model takes 12 days for one epoch on 8-GPU NV-LINK v100.
Small model takes 2 days for one epoch on 8-GPU NV-LINK v100.

------------------------------------------
#Usage:

Fed with chinese question answer pair and get the combined representations.

You can refer to the main.py for more detail.

The model has been tested under sequence length less than 1024


------------------------------------

As the torch model file is very large, you should download it from the google drive and then put them to the root.

[Large model](https://drive.google.com/open?id=1H1b4G8a4plnZp762kmU4Sflo1OQ0kqVc)
unzip it into resource directory

small model in the resource

