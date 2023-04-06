# AI-sentiment-analysis
This mini project involves the automatic classification of sentiment from recorded tweets from Twitter. The task is to study the Tweets dataset, prepare it for machine learning, and to select the best classification model for automatically determining the sentiment displayed by the tweet. This is a classification task and will require a supervised learning approach.

## How to run?
1. Open the directory in the latest version of Matlab
2. Execute `run sentiment_analysis.m` in the Matlab `Command Window`
3. The script should display figures automatically (could take several seconds)

## Project description
The provided data is a `.csv` file with a simple table of tweets and the categorical sentiment associated with each tweet. There are 2 columns and 8040 rows; the first column one contains the categorical sentiment; the second column contains the plaintext tweet. The tweets are organized in an alphabetical order.

The four unique sentiments are `enthusiasm`, `happiness`, `relief`, and `surprise`. All categorical sentiments are found using the following line. 

```matlab
 unique_labels = unique(all_sentiments);
```

In order for the machine learning model to utilize this information, the data is processed and prepared. First, a `Bag-of-Words (BoW)` model is prepared for this model. A `BoW` (also known as a term-frequency counter) records the number of times that a specific word appears in a given document of a collection. However, Matlab function `bagOfWords()` does not split the text into words. For that, each individual tweet is tokenised using the `tokenizedDocument()` method.

Now that words from each tweet are tokenized and ready to be counted, all English punctuation, common stop words and infrequent words (words with frequency less than 100 times) can be removed, once again using Matlab NLP library. This filter allows separating the actual words from other parts of speech that might have little to no significance. 


```matlab 
 erasePunctuation(tokenizedDocument(data)
 removeWords(bag,stopWords)
 removeInfrequentWords(bag,99)
```

The next necessary step is to build a full TF-IDF matrix for the resulting BoW. 

```matlab 
 M1 = tfidf(bag); 
 full(M1);
```
For most of the classification algorithms to recognise this data, the matrix has to be spread, therefore it is essential to use `full()`. 

The final data preparation is to select the label vectors and feature matrix. There are two sets of these, one for training, and one for testing and evaluating. The training set creates a feature matrix and label vectors for training by selecting the first `m` rows of the TF-IDF matrix and all columns. The testing set of feature matrix and label vectors uses the rows after the `m` row.

## Results

To see the full report and results please refer to `data_report.pdf`.
