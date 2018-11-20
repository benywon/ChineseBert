# ChineseBert
This is a chinese Bert model specific for question answering. We provide two models, a large model which is a 16 layer 1024 transformer, and a small model with 8 layer and 512 hidden size. Our implementation is a different from the original paper https://arxiv.org/abs/1810.04805, in which we replace a position embedding with GRU. And we adopt a memory efficient attention block.

-------------------------------------

Stats:

Data: 200m chinese internet question answering pairs.
Vocab: 52777, jieba CWS enhanced with forward maximum matching.

large model takes 12 days for one epoch on 8-GPU NV-LINK v100.
Small model takes 2 days for one epoch on 8-GPU NV-LINK v100.

------------------------------------------
Usage:

Fed with chinese question answer pair and get the combined representations.

You can refer to the interface.py for more detail.

------------------------------------

As the torch model file is very large, you should download it from the google drive and then put them to the root.

Large model : https://drive.google.com/open?id=1KWnMHBP39iIlU8lfMoaFnz6qn19_g7Gc

Small model : https://drive.google.com/open?id=1JMf3daDdMvi2GXH39EatcSmZws8cxFZW

