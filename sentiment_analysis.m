%% Data preparation

clear all; close all; clc;

% [1] import the .csv file
data = readtable('DataFiles/SentimentAnalysisData/text_emotion_data_filtered.csv');
all_tweets = lower(data.Content);
all_sentiments = data.sentiment;

% [2] build a Bag of Words containing all tokenized tweets (ignore
% punctuation)
bag = bagOfWords(erasePunctuation(tokenizedDocument(all_tweets)));

% [3] remove stop words + words with fewer than 100 occurances
bag = removeWords(bag,stopWords);
bag = removeInfrequentWords(bag,99);
top_words = topkwords(bag,100);

% [4] build a full TF-IDF matrix for the resulting bag
M1 = tfidf(bag); full(M1);

% [5] build a corresponding label vector from the column of sentiments
% (enthusiasm, happiness, relieff, surprise)
unique_labels = unique(all_sentiments); 

%% Features and labels

m = 6432; % size of testing Tweets
n = size(all_tweets,1); % size of all data in the document

% [1] create one feature matrix for training by selecting the first m 
% rows of the TF-IDF matrix and all columns
training_features = M1(1:m,:);full(training_features); % rows 1 -> m

% [2] create a corresponding label vector with the first m
training_labels = all_sentiments(1:m,:); % rows 1 -> m

% [3] create one feature matrix for testing by selecting all rows 
% of the TF-IDF matrix after row n (i.e. the remaining rows)
testing_features = M1(m:n,:); % rows m -> n

% [4] create a corresponding label vector
testing_labels = all_sentiments(m:n,:); % rows m -r m

%% Model training and evaluation
% Comparing three classification algorithms to evaluate the performance and
% model training of the data set

% [1] K-Nearest Neighbour
% fitcknn fit KNN classification model.
knnmodel = fitcknn(training_features,training_labels);
predictions = predict(knnmodel,testing_features);

% accuracy  
correct_predictions = sum(strcmp(testing_labels, predictions));
knn_accuracy = correct_predictions /size(testing_labels,1) % 0.4133

% plot
figure(1)
knn_chart = confusionchart(testing_labels, predictions); title("Figure 1. K-Nearest Neighbour");
knn_chart.RowSummary = 'row-normalized';
knn_chart.ColumnSummary = 'column-normalized';

% [2] Discriminant Analysis
% fitcdiscr Fit discriminant analysis.

discrmodel = fitcdiscr(training_features,training_labels);
predictions = predict(discrmodel,testing_features);

% accuracy  
correct_predictions = sum(strcmp(testing_labels, predictions));
discr_accuracy = correct_predictions /size(testing_labels,1)  % 0.5140

% plot
figure(2)
disc_chart = confusionchart(testing_labels, predictions); title("Figure 2. Discriminant Analysis");
disc_chart.RowSummary = 'row-normalized';
disc_chart.ColumnSummary = 'column-normalized';

% [3] SVM for Multiclass
% fitcecoc Fit a multiclass model for Support Vector Machine or other classifiers.
svmmodel = fitcecoc(training_features,training_labels, 'Learners', templateDiscriminant);
predictions = predict(svmmodel,testing_features);

% accuracy
correct_predictions = sum(strcmp(testing_labels, predictions));
svn_accuracy = correct_predictions /size(testing_labels,1) % 0.5109

% plot
figure(3)
svn_chart = confusionchart(testing_labels, predictions); title("Figure 3. SVM for Multiclass");
svn_chart.RowSummary = 'row-normalized';
svn_chart.ColumnSummary = 'column-normalized';