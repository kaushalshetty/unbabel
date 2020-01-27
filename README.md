## Word Unscramble

### Objective
To unscramble the input text file

### Noise in the data
Random order of sequences of input words

## Approaches

### Approach 1 (Thoughts)
When I intially saw the problem statement, I began to solve it using a Language Model  in an unsupervised way. The idea was to train a LSTM based langauge model using the `train_originnal.txt` data and then predict the probablity of all possible sentences that can be formed from the `test_scrambled.txt`.
After training the model , we input the  scrambled words and get all possible permutations for scrambled words. The idea was to predict the probablity of all possible permutations of the scrambled words , and select the permuation which gives the highest probablity. If the Language maodel is trained well enough , then it must predict the right permutation with the highest probablity.

#### Drawback
I figured out that is a completely infeasible solution because a very small sentence will give back a million permutaions and calculating probablity would take years.

### Approach 2 (Implemented)
My second approach was to make use of the unscrambled data that was shared. I started thinking on the lines of encoder decoder  based architecture and solve it like a seq2seq problem. The input to the RNN-based encoder would be the scrambled words and the decoder would decode the unscrambled words. This approch sounded good enough to the problem as the encoder and decoder would learn to predict the right positions of the scrambled words over training. But I felt this has a drawback


#### Drawback
The drawback is that the scrambled words are not recurrent in nature i.e there is no sequence in it, and hence by forcing the scrambled words to follow a recurrent nature (in the RNN encoder) would add noise to the context encodings. The decoder would be confused when the same sentence is scrambled in a different way.

### Approach 3 (Implemented)
**Bag of Embeddings approach (Implemented)**  
Instead of following a seq2seq model , we can solve the problem as a  **BaggedEmbedding2Seq** problem. The idea is to project all the input word embeddings of the scrambled words into another dimension using a linear projection layer. This would act as our encoder. Now, we decode each of the unscrambled word in the decoder side using **Bahdanau's attention** mechanism. While decoding each word , we make use of Bahdanau's attention mechanism to produce a dynamic context vector to decode each step. 

#### Training procedure:
Training the model was straightforward, where we send the input tokens to the encoder that embeds these tokens and transforms it linearly. The decoder would then produce a context vector, based on encoder outputs, for each decoding step and the output is softmaxed across the vocabulary. **Here lies another trick**. I wanted to take softmax across the complete vocabulary and not explicitly on the scrambled words, because I wanted the neural network to punish itself if it predicted a word other than the input scrambled words. A better way would be do sample these two action in a probablistic manner.

#### Inference:
Initially , I followed a straight forward greedy way to decode each word where  I selected the word that had the maxiumum probablity(argmax softmaxP(V)), but I observed that the words decoded were not coming from the input scrambled words. That is when I decided to **mask all the other words that were not part of the input scrambled words**. I also **masked out those words that were already decoded**.  

#### Observations:
 Even after training for only three epochs, the model started producing the right sequence for small sentences. I observed that the model learnt to identify the first word by the capital letters. It also started producing sequence of words that were grammatically better than scrambled words. I also observed that the model performed **poorly on long sentences**.  
 The reason I think is because the input scrambled words are not contextually aware of the other scrambled words in the input side. My thought process was to implement a self attention layer(Dot-product attention(Attention is all you need)) in the encoder side

#### Challenge  
The main challenge was that the network took a lot of time to train. I was able to train only for 3 epochs in my machine. I think the reason was because I did not cut down my vocabulary while training.

### Approach 4 (IMPLEMENTED):
This approach is an added feature on top of the previous approach. I implemented a custom self attention layer on the encoder side.

#### Observations:
Some of the  longer sentences started to get decoded almost perfectly.

#### Challenge:
Time to train increased further.

#### Output:
The output from this model that was only trained for three epochs is in the predictions/ folder. 


#### Some Observed Unscrambles from Test Set:
1) This is why there is high unemployment.
2) One of the most important issues of the energy. of course, is, storage
3) It seems that this system not did generate debt.
4) It has been not still solved.
5) There will be no additional until posts 2013.
6) I see this in my own country, situation Romania.
7) Mr President, the majority of the vast of the world have been in the history in which we should not a war in that we discuss there has been victims as Iraq immediately never realise war innocent civilians.
**NOTE**: The above results can be reproduced by running all the cells of the jupyter notebook** 
### Approach 5:(Implemented only the training part)
Now I started thinking, about the other features that can be used to the decoder to decode correctly. **POS tags** are an extremely important property in word sequences. For example, most of the sentences would not end with a noun. This property if included with the input embeddings would enrich the context even further. It would be  easy for the decoder because it would know the POS tags of each input words.
I implemented this feature by training a **seperate embedding layer for the POS tags**. The word embedding would then be concatenated with the POS embeddings and passed through a linear layer.
NLTK was used to extract the pos tags.

#### Observations:
I was not able to infer the results as the model was still getting trained but my guess would be better decodings.


### Future Improvements:
One thing that I wanted to implement was a beam search based decoder as it would increase the search space for the right sentence.
Currenlty, I have not taken care of OOV words and I am ommitting all the OOV words from the test_predictions.txt. One approach I can think of is to use a switching/pointer network that can copy some of the words from the input scrambled words even if the words are not part of the vocabulary.  I could have also trained the model on word2vec or fastext intialized embeddings. This would solve some part of the OOV problem as we would have embeddings for vocabulary that were unseen in the training data.
Furthermore, we could try and see how a Transformer based decoder would behave in the decoder side. Due to scarcity of time I could not implement these approaches.


#### FOLDER/FILES
**notebooks/**: 
1) Self_Attention_with_Bahdanau_att.ipynb   #  This notebook has the code used to train a Approach 4 model
2) bag2seq.ipynb  # Code to train Approach  3
3) Self_attn_with_POS_pos.ipynb  # code to train Approach 5

**models/**
* checkpoints for different models

**predictions/**
1) prediction_self_att.txt  # Predictions after 3 epochs by following Approach 4


**scripts/**:
- model.py  # code that has all the models (AttentiveEncoderPOS, BahdanauDecoder,AttentiveEncoder)   
- self_attention.py  # custom self attention layer  
- vocab.py  # code to build vocabulary  
- train_self_att.py  # code to train approach 4 model.  
- train_pos_att.py   # code to train approach 5 model  
- unscramble.py   # for evaluation of test_scrambled.txt  

**NOTE: For clarity on my neural network architectures and my thought process , please refer the jupyter notebooks(Self_Attention_with_Bahdanau_att.ipynb, Self_attn_with_POS_pos.ipynb). The scripts/ folder contents were written at the last minute.**

**data/**:
* input files 


##### To replicate the results:
Requirements:  
Python 3.7.2   
Pytorch   
NLTK   
1) Train the model
`python train_self_att.py`  # This will save the model in ../models/ folder
2) Run evaluation after training 
`python unscramble.py`   # This will save the predictions in the predictions/ folder
 
 
 ### Thank You!
 
#### Acknowledgement/ References:
1) https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
2) https://nlp.seas.harvard.edu/2018/04/03/attention.html
3) https://www.aclweb.org/anthology/W17-3531.pdf

