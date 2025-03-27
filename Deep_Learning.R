library(keras)
library(ggplot2)
library(tidyverse)
library(stringr)  


#Preprocess the text data and labels, using steps similar to the ones followed in the examples in tutorials 9-10. Present descriptive statistics and characteristics of the data and use reasonable values for the parameters num_words and maxlen.
train <- read.csv("Corona_NLP_train.csv", encoding='latin1')
test <- read.csv("Corona_NLP_test.csv", encoding='latin1')

#Keep only necessary columns
df_train <- train %>% select(OriginalTweet, Sentiment)
df_test <- test %>% select(OriginalTweet, Sentiment)

#look at the distribution of the Sentiment
table(df_test$Sentiment)

#PLOT IT GANG 

samples <- as.data.frame(train$OriginalTweet)
samples = rgx.Replace(samples, "");
#samples <- as.character(samples)

tokenizer <- text_tokenizer(num_words = 100000) %>% #text_tokenizer includes a set number of words           
  fit_text_tokenizer(samples) #feed it to fit_text_tokenizer which converts the text to tokens

tokenizer
# internal class, not interpretable by itself
# it has a keras internal structure representing the word index

sequences <- texts_to_sequences(tokenizer, samples)                 


#Define a simple NN that has one embedding layer, one dense hidden layer and one output layer. Use appropriate parameters and settings for the network, consistent with the size and dimensionality of the data. Choose proper loss and performance metrics.


#Compile and train the network, using a reasonable batch size, and using 20% of the data for validation. Make an optimal choice for the number of epochs using the validation performance. Record and report the results.


#Replace the previous network, with a RNN with one recurrent layer, keeping the embedding layer. Use reasonable values for any remaining hyperparameters. Record and compare the results.


#Now add a second recurrent layer, and observe and report and improvement in the model. Select a “best RNN model” based on the validation performance.


#Replace the simple RNN with a LSTM model, using also dropout. Comment on any improvement in the performance.


#Finally, evaluate your best-performing models one from each type (FFNN, RNN, LSTM) using the test data and labels.

#Include a section of lessons learned, conclusions, limitations and potential next steps, reflecting on your analysis.
