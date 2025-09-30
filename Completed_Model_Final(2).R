install.packages("tidyverse")
library(tidyverse)

################## Import Raw Data ##################

diagnosis <- read.csv("Disease_Symptom_Prediction.csv", header = T)  # Our rows have a header.
View(diagnosis)  # I noticed that some entries have spaces. This may be fine since we will be converting to binary

str(diagnosis)           # 17 Symptom categories
table(diagnosis$Disease) #all of our disease have 120 entries. No repeating entries

# Will take the original data set and clean it in Excel.

################## Import Data Cleaned Data ##################

symptoms <- read.csv("Disease_Symptom_Cleaned.csv", header = T)   # Our rows have a header.
View(symptoms)
str(symptoms)             # 131 symptoms that we need to convert to factors
table(symptoms$Disease)   # Notice there were empty rows. Rows are blank so we will remove them from the data fram


symptoms   <- symptoms[-c(4921,4922), ]
symptoms[] <- lapply(symptoms, factor)
str(symptoms)
# 0 equals absent and 1 equals yes or present

################## Exploratory Data Analysis ##################
install.packages("wordcloud")
library(wordcloud)

# Remove non-symptom columns if needed
symptom_only         <- symptoms[, !(names(symptoms) %in% c("Disease"))]

# Ensure they are numeric
symptom_only_numeric <- data.frame(lapply(symptom_only, function(x) as.numeric(as.character(x))))

# Count how many times each symptom appears (i.e., how many 1s)
symptom_freq <- colSums(symptom_only_numeric)

# Remove symptoms that never appear (optional)
wordcloud(names(symptom_freq), symptom_freq,
          min.freq = 5000, scale = c(5, 0.5), colors = brewer.pal(8, "Set2"))

sort(symptom_freq, decreasing = TRUE)[1:10]

################## Creating random training and test data set ##################
install.packages("caret")
library(caret)

set.seed(0409)
train_sample  <- createDataPartition(symptoms$Disease, p = 0.75,
                                     list = FALSE)
symptom_train <- symptoms[train_sample, ]
symptom_test  <- symptoms[-train_sample, ]

# Train Model
install.packages("C50")
library(C50)
symptoms_model <- C5.0(symptom_train[-1], symptom_train$Disease)
symptoms_model
summary(symptoms_model)

# Evaluating Model Performance
symptoms_pred <- predict(symptoms_model, symptom_test)
summary(symptoms_pred)

conf_matrix <- confusionMatrix(symptoms_pred, symptom_test$Disease)
conf_matrix$byClass[, c("Sensitivity", "Specificity", "F1","Precision")]

################## Test Case ##################

features <- setdiff(names(symptom_train),"Disease") # This could be a nice visual to show the number of diseases

# Create empty test case with all features = 0
mock_case <- as.data.frame(matrix(0, nrow = 1, ncol = length(features)))
colnames(mock_case) <- features

# Turn on the symptoms the patient has
mock_case$abdominal_pain       <- 1
mock_case$chills               <- 1
mock_case$fever                <- 1
mock_case$continuous_sneezing  <- 1
mock_case$shivering            <- 1
mock_case$watering_from_eyes   <- 1
mock_case$cough                <- 1


pred_disease <- predict(symptoms_model, mock_case)
pred_prob    <- predict(symptoms_model, mock_case, type = "prob")
pred_name    <- as.character(pred_disease)

# 3. Extract confidence for predicted class
confidence <- pred_prob[pred_disease]

# 4. Show the result
cat("Predicted Disease:", pred_name, "\n")
cat("Model Confidence:", round(confidence * 100, 2), "%\n")


################## Random Forest ##################

install.packages("randomForest")
library(randomForest)
RNGversion("3.5.2")
set.seed(400)

rf_model    <- randomForest(Disease ~ ., data = symptom_train)
predictions <- predict(rf_model,symptom_test)
actual      <- symptom_test$Disease

rf_conf_matrix <- confusionMatrix(predictions,actual)
rf_conf_matrix$overall["Accuracy"]
rf_conf_matrix$byClass[,"Precision"]
rf_conf_matrix$byClass[,"Specificity"]
rf_conf_matrix$byClass[,"Sensitivity"]

accuracy_vals  <- rf_conf_matrix$overall["Accuracy"]
precision_vals <- rf_conf_matrix$byClass[ , "Precision"]
spec_vals      <- rf_conf_matrix$byClass[ , "Specificity"]
sen_vals       <- rf_conf_matrix$byClass[ , "Sensitivity"]
f1_vals        <- rf_conf_matrix$byClass[ , "F1"]
kappa_val      <- rf_conf_matrix$overall["Kappa"]

round(accuracy_vals, digits = 4)
round(precision_vals, digits = 4)
round(spec_vals, digits = 4)
round(sen_vals, digits = 4)
round(f1_vals, digits = 4)

##########################################
for (col in colnames(mock_case)) {
  test_case[[col]] <- factor(mock_case[[col]], levels = c(0, 1))
}

rf_pred   <- predict(rf_model, mock_case)
print(rf_pred)
rf_prob   <- predict(rf_model, mock_case, type = "prob")

rf_confidence    <- max(rf_prob)
rf_pred_name     <- as.character(rf_pred)
rf_pred_disease  <- colnames(rf_prob)[which.max(rf_prob)]

cat("Predicted disease:", rf_pred_name, "\n")
cat("Confidence:", round(rf_confidence * 100, 4), "%\n")


###########################################################

# Get top N classes by actual frequency
conf_mat    <- table(predicted = symptoms_pred, Actual = symptom_test$Disease)
conf_df     <- as.data.frame(conf_mat)
top_classes <- names(sort(table(symptom_test$Disease), decreasing = TRUE))[1:5]

# Filter the confusion data to just these classes
conf_df_filtered <- subset(conf_df, Actual %in% top_classes & predicted %in% top_classes)

# Plot filtered confusion matrix
ggplot(conf_df_filtered, aes(x = Actual, y = predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 4) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_minimal() +
  labs(title = "Filtered Confusion Matrix (Top 5 Classes)", x = "Actual", y = "Predicted")

################## WORD CLOUD ##################

install.packages("wordcloud")
install.packages("RColorBrewer")
library(wordcloud)
library(RColorBrewer)

# we see the most frequent symptoms

sort(symptom_totals, decreasing = TRUE)[1:10]
