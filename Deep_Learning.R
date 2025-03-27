library(keras)
library(ggplot2)
library(tidyverse)

# Be better plot it gang

#Preprocess the text data and labels, using steps similar to the ones followed in the examples in tutorials 9-10. Present descriptive statistics and characteristics of the data and use reasonable values for the parameters num_words and maxlen.
#Keep only necessary columns
df_train <- read.csv("Corona_NLP_train.csv", encoding = "latin1") %>% 
  select(OriginalTweet, Sentiment)
df_test <- read.csv("Corona_NLP_test.csv", encoding = "latin1") %>% 
  select(OriginalTweet, Sentiment)


#look at the distribution of the Sentiment
table(df_test$Sentiment, useNA = "ifany")
table(df_train$Sentiment, useNA = "ifany")

# Inspecting column names
print(colnames(df_train))
print(head(df_train$OriginalTweet))

# Define sentiment levels
sentiment_levels <- c("Extremely Negative", "Negative", "Neutral", "Positive", "Extremely Positive")

# Convert sentiment column to a factor with the levels
df_train$Sentiment <- factor(df_train$Sentiment, levels = sentiment_levels)
df_test$Sentiment <- factor(df_test$Sentiment, levels = sentiment_levels)

# One-hot encode the sentiment labels (subtract 1 since keras expects classes starting at 0)
y_train <- to_categorical(as.integer(df_train$Sentiment) - 1, num_classes = 5)
y_test  <- to_categorical(as.integer(df_test$Sentiment) - 1, num_classes = 5)


# Set parameters for tokenization and padding/one-hot encoding
max_words <- 10000  # vocabulary size
maxlen    <- 50    # maximum tweet length (in words)

# Create tokenizer
tokenizer <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(df_train$OriginalTweet)

x_train <- texts_to_sequences(tokenizer, df_train$OriginalTweet) %>% pad_sequences(maxlen = maxlen)
y_train <- as.array(df_train$Sentiment)

x_test <- texts_to_sequences(tokenizer, df_test$OriginalTweet) %>% pad_sequences(maxlen = maxlen)
y_test <- as.array(df_test$Sentiment)





#Define a simple NN that has one embedding layer, one dense hidden layer and one output layer. Use appropriate parameters and settings for the network, consistent with the size and dimensionality of the data. Choose proper loss and performance metrics.


#Compile and train the network, using a reasonable batch size, and using 20% of the data for validation. Make an optimal choice for the number of epochs using the validation performance. Record and report the results.


#Replace the previous network, with a RNN with one recurrent layer, keeping the embedding layer. Use reasonable values for any remaining hyperparameters. Record and compare the results.


#Now add a second recurrent layer, and observe and report and improvement in the model. Select a “best RNN model” based on the validation performance.


#Replace the simple RNN with a LSTM model, using also dropout. Comment on any improvement in the performance.


#Finally, evaluate your best-performing models one from each type (FFNN, RNN, LSTM) using the test data and labels.

#Include a section of lessons learned, conclusions, limitations and potential next steps, reflecting on your analysis.
