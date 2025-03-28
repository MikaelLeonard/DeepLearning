library(keras)
library(ggplot2)
library(tidyverse)
library(RColorBrewer)

# Be better plot it gang

#Preprocess the text data and labels, using steps similar to the ones followed in the examples in tutorials 9-10. Present descriptive statistics and characteristics of the data and use reasonable values for the parameters num_words and maxlen.
#Keep only necessary columns
df_train <- read.csv("Corona_NLP_train.csv", encoding = "latin1") %>% 
  select(OriginalTweet, Sentiment) #read the csvfile, using the latin1 format to endode characters into one of the 191 recognized encodings of the latin script. This will exclude certain accented characters 
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
df_train$Sentiment <- factor(df_train$Sentiment, levels = sentiment_levels) #conversion to a factor will make it useful for subsequent numerical encoding
df_test$Sentiment <- factor(df_test$Sentiment, levels = sentiment_levels)

# One-hot encode the sentiment labels (subtract 1 since keras expects classes starting at 0)
y_train <- to_categorical(as.integer(df_train$Sentiment) - 1, num_classes = 5)
y_test  <- to_categorical(as.integer(df_test$Sentiment) - 1, num_classes = 5)

# Sentence Length distribution
Sentence_Length <- lengths(strsplit(df_train$OriginalTweet," ")) #create a vector of the number of words per sentence
hist(Sentence_Length, breaks=100) #By analyzing the distribution of sentence-lengths, it is clear that there is not an extremely long tail of data that when included would result in dramatically large sections of padding added to the shorter sentences
boxplot(Sentence_Length)

# Set parameters for tokenization and padding/one-hot encoding
max_words <- 10000  # vocabulary size
maxlen    <- quantile(Sentence_Length,0.95)    # maximum tweet length (in words), equals 48 words

# Create tokenizer
tokenizer <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(df_train$OriginalTweet)

x_train <- texts_to_sequences(tokenizer, df_train$OriginalTweet) %>% pad_sequences(maxlen = maxlen)

x_test <- texts_to_sequences(tokenizer, df_test$OriginalTweet) %>% pad_sequences(maxlen = maxlen)

# shuffle the training data 
set.seed(1568)
I <- sample.int(nrow(x_train))
x_train <- x_train[I,]
y_train <- y_train[I,]

#Define a simple NN that has one embedding layer, one dense hidden layer and one output layer. Use appropriate parameters and settings for the network, consistent with the size and dimensionality of the data. Choose proper loss and performance metrics.

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = 8,  # the new space dimension      
                  input_length = maxlen) %>%
  layer_flatten() %>%   
  layer_dense(units = 64, activation = "relu")  %>%   
  layer_dense(units = 5, activation = "softmax")    

# # of weights of the embedding layer = 10,000 x 8 = 80K
# Output shape = (50,8), since 50 is the length of the word sequence

model


model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

#train the doel with the data , 80% train, 20% validation 
#use embedding from the data 
#chose the max len to be so small - computation is not high but the accuracy is mediocre 
history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

plot(history)

model_fnn <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = 8, input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 5, activation = "softmax")

model_fnn %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

#train the model with the data , 80% train, 20% validation 
#use embedding from the data 
#chose the max len to be so small - computation is not high but the accuracy is mediocre 
history_fnn <- model_fnn %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

plot(history_fnn)

#Compile and train the network, using a reasonable batch size, and using 20% of the data for validation. Make an optimal choice for the number of epochs using the validation performance. Record and report the results.

model_fnn_final <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = 8, input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 5, activation = "softmax")

model_fnn_final %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

#train the model with the data , 80% train, 20% validation 
#use embedding from the data 
#choose the max len to be so small - computation is not high but the accuracy is mediocre 
history_fnn_final <- model_fnn_final %>% fit(
  x_train, y_train,
  epochs = 8,
  batch_size = 32,
  validation_split = 0.2
)

plot(history_fnn_final)

final_fnn_results <- model_fnn_final %>% evaluate(x_test, y_test)
final_fnn_results

#Replace the previous network, with a RNN with one recurrent layer, keeping the embedding layer. Use reasonable values for any remaining hyperparameters. Record and compare the results.
# RNN Model
model_rnn <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = 8) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 5, activation = "softmax")

model_rnn

model_rnn %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history_rnn <- model_rnn %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_split = 0.2
)

plot(history_rnn)

rnn_results <- model_rnn %>% evaluate(x_test, y_test)
rnn_results

#Now add a second recurrent layer, and observe and report any improvement in the model. Select a “best RNN model” based on the validation performance.

model_rnn_2 <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = 8) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 16) %>%
  layer_dense(units = 5, activation = "softmax")

model_rnn_2

model_rnn_2 %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history_rnn_2 <- model_rnn_2 %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_split = 0.2
)

plot(history_rnn_2)

rnn_results_2 <- model_rnn_2 %>% evaluate(x_test, y_test)
rnn_results_2

# RNN with dropout

model_rnn_final <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = 8) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE, 
                   dropout = 0.1, recurrent_dropout = 0.1) %>%
  layer_simple_rnn(units = 16, 
                   dropout = 0.1, recurrent_dropout = 0.1) %>%
  layer_dense(units = 5, activation = "softmax")

model_rnn_final %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history_rnn_final <- model_rnn_final %>% fit(
  x_train, y_train,
  epochs = 15,
  batch_size = 32,
  validation_split = 0.2
)

plot(history_rnn_final)

rnn_results_final <- model_rnn_final %>% evaluate(x_test, y_test)
rnn_results_final

#Replace the simple RNN with a LSTM model, using also dropout. Comment on any improvement in the performance.

model_lstm <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = 8) %>%
  layer_lstm(units = 32) %>%
  layer_dense(units = 5, activation = "softmax")

model_lstm

model_lstm %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history_lstm <- model_lstm %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_split = 0.2
)

plot(history_lstm)

# LSTM Model with Dropout
model_lstm_final <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = 8) %>%
  layer_lstm(units = 16, dropout = 0.2, recurrent_dropout = 0.2) %>%
  layer_dense(units = 5, activation = "softmax")

model_lstm_final %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history_lstm_final <- model_lstm_final %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_split = 0.2
)

plot(history_lstm_final)

lstm_final_results <- model_lstm_final %>% evaluate(x_test, y_test)
lstm_final_results

preds_lstm_final_results <- predict(model_lstm_final, cbind(x_test, y_test))
preds.cl <- max.col(preds_lstm_final_results) #grab the column which has the highest value per each row

lstm_final_Matrix <- matrix(table(max.col(y_test),preds.cl)[1:5,1:5],nrow=5)
colnames(lstm_final_Matrix) <- c('Ext. Neg.','Neg.','Neut.','Pos.','Ext. Pos.')#sentiment_levels
rownames(lstm_final_Matrix) <- c('Ext. Neg.','Neg.','Neut.','Pos.','Ext. Pos.')#sentiment_levels

install.packages('pheatmap') # if not installed already
library(pheatmap)
pheatmap(lstm_final_Matrix,
         display_numbers = T,
         treeheight_row=0,
         treeheight_col=0,
         cluster_rows=F,
         cluster_cols=F,
         color = brewer.pal(9,"YlGn"),
         number_color='red')

#TRY LSTM RESULTS WITHOUT DROPOUT


#Finally, evaluate your best-performing models one from each type (FFNN, RNN, LSTM) using the test data and labels.

#Include a section of lessons learned, conclusions, limitations and potential next steps, reflecting on your analysis.
