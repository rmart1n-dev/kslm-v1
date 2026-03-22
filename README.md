# kslm-v1
Initial code for version 1 of my KSLM (Knowledgeless Small Language Model).

## What is a KSLM?

The KSLM (Knowledgeless Small Language Model) is an experiment in training a language model that is fully able to understand language, yet retains no knowledge pertaining to the real world. For example, it could understand this paragraph, yet if you were to ask it a knowledge-based question (i.e. "What is the capital of France?") it would be unable to respond.

## Results?

From an initial training run using the CNN/DailyMail dataset (preproccessed through an NER) a 150m parameter model exhibited rather low intelligence, often repeating tokens. By adding a penalty for repetition it was able to exhibit relatively coherent English, although overfitted to both the NER-processed tokens as well as the news-report style of the dataset. For a 3 hour experiment it isn't the worst.

