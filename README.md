# Unbabel AI Research Challenge #2

Dear candidate,

Congratulations for having passed to the challenge stage in our hiring process!

This is a short 48-hour challenge to understand how you work, your thought process and methodology.
You do not need to be familiar with the described problem to be able to do this challenge.
In fact, first and foremost, we are going to focus on your reasoning and the steps you took.

This challenge will be followed by a 1-hour interview, 
where we will discuss your approach to this exercise and will make other technical questions 
(focusing on algorithms, design, data structures, and coding proficiency).

Don't forget to plan your time accordingly, and good luck! :-)

-- The Unbabel Team

## Description

In this challenge, you will recover English sentences corrupted with random noise. 

As training data, you are given the following files (in folder `data`): 
- `train_original.txt`
- `train_scrambled.txt`

The first file contains the original (uncorrupted) text, one sentence per line. 
The second file contains the corrupted version of these sentences. 
The goal is to write a computer program that, when presented with similar corrupted text, 
recovers as much as possible of the original sentences. 
This involves understanding how the data was corrupted and to build a model that is suitable for this task. 
As test data, we provide only the file
- `test_scrambled.txt`

Your computer program, when taken this file as input, should produce as output another file, `test_predicted.txt`,  
with the best guesses of what the original text is. 
The evaluation metric is the BLEU score between the predicted file and the original (reference) one. 
The evaluation script is provided.

When you are done, please add a folder `predictions` containing the following files:
- a zip/tarball containing your code;
- the output of your program, the file `test_predicted.txt` -- you may send more than one version of this output if you made several runs with different models;
- a short description of what you did (one or two paragraphs are enough -- we will discuss this in more detail in the interview).

You can use whatever programming language you like (Python is suggested).

I recommend the following: look at the data to try to understand what kind of noise is corrupting it.
Start with a simple baseline to make sure you have something to submit by the deadline.
Then you can spend some time trying to improve over this baseline.

Please do not share your code in a public repository (since other candidates will work independently on this challenge).

Feel free to ask any clarification questions you may have regarding this exercise.

## Guidelines
* ***Fork this _git repo_*** and add relevant files and explanations for what you feel necessary 
(starting with a clear description of what is implemented in the README)
* ***Send us a link to your fork*** as soon as you start working on it, and then let us know when you're done.
* If you can't finish the challenge due to unforeseen personal reasons let us know ASAP so we can adapt your deadline and/or challenge.
* Any challenge related questions, technical or otherwise, feel free to contact us: `ai-challenge@unbabel.com`.