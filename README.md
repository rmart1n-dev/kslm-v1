# kslm-v1
Initial code for version 1 of my KSLM (Knowledgeless Small Language Model).

## What is a KSLM?

The KSLM (Knowledgeless Small Language Model) is an experiment in training a language model that is fully able to understand language, yet retains no knowledge pertaining to the real world. For example, it could understand this paragraph, yet if you were to ask it a knowledge-based question (i.e. "What is the capital of France?") it would be unable to respond.

## Why?
I recently saw the paper [https://openreview.net/pdf/1c4a46ec0583c91ff13b82a1a0a9ed916e9c2a95.pdf](Knowledgeless Language Models: Decoupling Linguistic Competence and Factual Knowledge) after thinking to myself "What if an AI could understand language but not know facts?" Due to the fact that there is no model available currently that does this using the method described in the paper, I decided to take things into my own hands to challenge myself and see if I could actually make a working knowledgeless language model.

## Results?

From an initial training run using the CNN/DailyMail dataset (preproccessed through an NER) a 150m parameter model exhibited rather low intelligence, often repeating tokens. By adding a penalty for repetition it was able to exhibit relatively coherent English, although overfitted to both the NER-processed tokens as well as the news-report style of the dataset. For a 3 hour experiment it isn't the worst.

<img width="2832" height="1660" alt="training_report" src="https://github.com/user-attachments/assets/f07255dd-53af-46e0-8dd8-3bf355276d63" />
