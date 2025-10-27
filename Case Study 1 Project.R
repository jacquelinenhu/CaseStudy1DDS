### CASE STUDY 1 PROJECT: EDA ###

## GOAL: Identify the top 3 factors that contribute to attrition (voluntary employee turnover)

# import CaseStudy1_data into R Studio
library(readr)
CaseStudy1_data <- read_csv("SMU - M.S. Data Science/CaseStudy1-data.csv")
View(CaseStudy1_data)

# understanding CaseStudy1_data
dim(CaseStudy1_data) # 870 rows by 36 columns
str(CaseStudy1_data) # explains the data type for each column
colSums(is.na(CaseStudy1_data)) # how many NA are present in each column? --> none

# remove any columns that don't contribute to attrition: ID and EmployeeNumber
CaseStudy1_data_mod <- CaseStudy1_data[, !names(CaseStudy1_data) %in% c("ID", "EmployeeNumber")]
dim(CaseStudy1_data_mod) # 870 rows by 34 columns

# ensure that the column "Attrition" is a factor (target variable - y-value)
# as.factor() converts character or numeric vectors into factor levels, which is essential when using the train().
CaseStudy1_data_mod$Attrition <- as.factor(CaseStudy1_data_mod$Attrition)

#########

# split CaseStudy1_data_mod into 70% training set and 30% testing set
set.seed(456)
trainIndices = sample(seq(1:length(CaseStudy1_data_mod$Attrition)), round(.7*length(CaseStudy1_data_mod$Attrition)))
CaseStudy1_data_train = CaseStudy1_data_mod[trainIndices,] # 609 observations
CaseStudy1_data_test = CaseStudy1_data_mod[-trainIndices,] # 261 observations

## Naive Bayes Model
library(e1071)
library(dplyr)

# build NB model
nb_model <- naiveBayes(Attrition ~ ., data = CaseStudy1_data_train)

# make predictions
predictions <- predict(nb_model, CaseStudy1_data_test)

# evaluate model
confusionMatrix(table(predicted = predictions, Actual = CaseStudy1_data_test$Attrition))
# accuracy: 80.08%
# sensitivity: 83.94%
# specificity: 60.47%

# average monthly income of employee at Frito Lay using NB model's testing data
mean(CaseStudy1_data_test$MonthlyIncome) # $6317.33

#########

# drop EmployeeCount and StandardHours
unique(CaseStudy1_data_mod$EmployeeCount) # contains all 1
unique(CaseStudy1_data_mod$StandardHours) # contains all 80

CaseStudy1_data_mod_2 <- CaseStudy1_data_mod[, !names(CaseStudy1_data_mod) %in% c("EmployeeCount", "StandardHours")]
dim(CaseStudy1_data_mod_2) # 870 rows by 32 columns

# identify numeric columns
numeric_cols <- sapply(CaseStudy1_data_mod_2, is.numeric)

# normalize numeric columns via scaling
CaseStudy1_data_mod_2[numeric_cols] <- scale(CaseStudy1_data_mod_2[numeric_cols])

# split CaseStudy1_data_mod into 70% training set and 30% testing set
set.seed(456)
trainIndices = sample(seq(1:length(CaseStudy1_data_mod_2$Attrition)), round(.7*length(CaseStudy1_data_mod_2$Attrition)))
CaseStudy1_data_train = CaseStudy1_data_mod_2[trainIndices,] # 609 observations
CaseStudy1_data_test = CaseStudy1_data_mod_2[-trainIndices,] # 261 observations

# separate predictors and target variables
train_X <- CaseStudy1_data_train %>% select(-Attrition)
train_Y <- CaseStudy1_data_train$Attrition

test_X <- CaseStudy1_data_test %>% select(-Attrition)
test_Y <- CaseStudy1_data_test$Attrition

# ensure that all predictors are numeric as KNN requires numeric input
train_X <- data.frame(lapply(train_X, function(x) {
  if (is.factor(x) || is.character(x)) as.numeric(as.factor(x)) else x
}))
test_X <- data.frame(lapply(test_X, function(x) {
  if (is.factor(x) || is.character(x)) as.numeric(as.factor(x)) else x
}))

## KNN model
# Loop through k = 1 to 20.
library(class)

accuracy <- numeric(20)

for (k in 1:20) {
  predicted_Y <- knn(train = train_X, test = test_X, cl = train_Y, k = k)
  accuracy[k] <- mean(predicted_Y == test_Y)
}

# Plot accuracy vs k. --> best k value is 7
library(ggplot2)

accuracy_df <- data.frame(k = 1:20, accuracy)

ggplot(accuracy_df, aes(x = k, y = accuracy)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "darkred") +
  labs(title = "KNN Accuracy vs K (CaseStudy1_data Dataset)",
       x = "Number of Neighbors (k)",
       y = "Accuracy") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# train KNN model
set.seed(456) 
predicted_Y_prob <- knn(train = train_X, test = test_X, cl = train_Y, k = 7, prob = TRUE) 

# extract vote proportions
prob_values <- attr(predicted_Y_prob, "prob")

# adjust for positive class ("No")
prob_No <- ifelse(predicted_Y_prob == "No", prob_values, 1 - prob_values)

# build and plot the ROC Curve
library(pROC)

roc_obj <- roc(test_Y, prob_No, levels = c("Yes", "No"), direction = "<")
plot(roc_obj, col = "darkorange", main = "ROC Curve for KNN (k = 7)")

# find optimal threshold for specificity
coords(roc_obj, "best", ret = c("threshold", "specificity", "sensitivity"))
# threshold: 78.57%
# specificity: 39.53%
# sensitivity: 81.65%

# reclassify using custom threshold
custom_pred <- ifelse(prob_No > 0.7857, "No", "Yes")  # adjust threshold as needed

# evaluate the KNN classification model
confusionMatrix(table(Predicted = factor(custom_pred), Actual = test_Y), positive = "No")
# accuracy: 74.71%
# sensitivity: 81.65%
# specificity: 39.53%

## identify the top 3 factors that contribute to attrition
# since the NB model is the better model of the two, use that model to identify the top 3 factors.

# access conditional probability tables
nb_model$tables # produces a bunch of conditional probability tables showing each column's values distribute across Attrition = "Yes" and "No"

# quantify separation between classes
# computes the maximum difference in conditional probabilities for each variable
# be mindful that some categories are numeric and not categorical
importance_score <- sapply(nb_model$tables, function(tbl) {
  if (is.matrix(tbl)) {
    # Check if both "Yes" and "No" columns exist
    if (all(c("Yes", "No") %in% colnames(tbl))) {
      max(abs(tbl[, "Yes"] - tbl[, "No"]))
    } else {
      NA  # Skip if columns are missing
    }
  } else if (is.vector(tbl)) {
    # Check if both "Yes" and "No" names exist
    if (all(c("Yes", "No") %in% names(tbl))) {
      abs(tbl["Yes"] - tbl["No"])
    } else {
      NA
    }
  } else {
    NA
  }
})

# sort and display the top 3 factors that contribute to attrition
sort(importance_score, decreasing = TRUE, na.last = TRUE)[1:3]
# OverTime, Age, BusinessTravel are the top 3 factors

# determine the number of employees who stayed vs left Frito Lay
table(CaseStudy1_data_test$Attrition)
# No: 218
# Yes: 43

# create df comparing the number of employees who stayed vs left Frito Lay
pred_attrition_df <- data.frame(pred_attrition = c("No", "Yes"), 
                                num_employees = c(218, 43),
                                percent_employees = c(218/261 * 100, 43/261 * 100))
pred_attrition_df

# pie chart comparing the number of employees who stayed vs left Frito Lay
ggplot(pred_attrition_df, aes(x = "", y = percent_employees, fill = pred_attrition)) + 
  geom_bar(stat = "identity", width = 1) + 
  coord_polar(theta = "y") + 
  scale_fill_manual(values = c("No" = "skyblue", "Yes" = "tomato")) +
  geom_text(aes(label = paste0(num_employees, " (", round(percent_employees, 2), "%)")),
            position = position_stack(vjust = 0.5)) +
  theme_void() +
  labs(title = "Attrition at Frito Lay Predicted by the NB Classification Model", fill = "Attrition") +
  theme(plot.title = element_text(hjust = 0.5))

# creating new df containing just the top 3 factors and Attrition
selected_top_3_factors_df <- CaseStudy1_data_test[, c("OverTime", "Age", "BusinessTravel")]
Attrition <- CaseStudy1_data_test$Attrition
length(Attrition) == nrow(selected_top_3_factors_df) # TRUE
merged_test_data <- cbind(Attrition, selected_top_3_factors_df)
head(merged_test_data)

# OverTime vs. Attrition Grouped Barchart
# group and count by OverTime and Attrition
library(dplyr)
plot_data <- merged_test_data %>%
  group_by(OverTime, Attrition) %>%
  summarise(Count = n(), .groups = "drop") %>%
  group_by(OverTime) %>%
  mutate(Percent = Count / sum(Count),
         Label = paste0(round(Percent * 100, 1), "%"))

# grouped bar chart
ggplot(plot_data, aes(x = Attrition, y = Percent, fill = OverTime)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("Yes" = "coral", "No" = "skyblue")) +
  scale_y_continuous(labels = scales::percent) +
  geom_text(aes(label = Label),
            position = position_dodge(width = 0.9),
            vjust = -0.5, size = 3.5) + 
  labs(title = "Attrition by Overtime Status",
       x = "Attrition",
       y = "Percent of Employees",
       fill = "Working Overtime?") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Age vs. Attrition
# determine the max and min age of working at Frito Lay
max(merged_test_data$Age, na.rm = TRUE) # max age: 60
min(merged_test_data$Age, na.rm = TRUE) # min age: 18

# create age groups
merged_test_data <- merged_test_data %>%
  mutate(AgeGroup = cut(Age, breaks = c(10, 20, 30, 40, 50, 60, 70), labels = c("10s", "20s", "30s", "40s", "50s", "60s")))

head(merged_test_data)

sum(is.na(merged_test_data$AgeGroup)) # no NA values

# grouped bar chart
ggplot(merged_test_data, aes(x = Attrition, fill = AgeGroup)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("10s" = "coral", "20s" = "yellow", "30s" = "lightgreen", "40s" = "skyblue", "50s" = "violet")) +
  geom_text(stat = "count", aes(label = ..count..),
            position = position_dodge(width = 0.9), vjust = -0.5) +
  labs(title = "Attrition by Age Group",
       x = "Attrition",
       y = "Percent of Employees",
       fill = "Age Group") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# grouped barchart comparing AgeGroup vs. OverTime
ggplot(merged_test_data, aes(x = OverTime, fill = AgeGroup)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("10s" = "coral", "20s" = "yellow", "30s" = "lightgreen", "40s" = "skyblue", "50s" = "violet")) +
  geom_text(stat = "count", aes(label = ..count..),
            position = position_dodge(width = 0.9), vjust = -0.5) +
  labs(title = "Overtime Status by Age Group",
       x = "Overtime Status",
       y = "Number of Employees",
       fill = "Age Group") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# BusinessTravel vs. Attrition Grouped Barchart
ggplot(merged_test_data, aes(x = Attrition, fill = BusinessTravel)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("Non-Travel" = "lightgreen", "Travel_Frequently" = "tomato", "Travel_Rarely" = "skyblue"),
                    labels = c("Non-Travel" = "No Travel", "Travel_Frequently" = "Travel Frequently", "Travel_Rarely" = "Travel Rarely")) +
  geom_text(stat = "count", aes(label = ..count..),
            position = position_dodge(width = 0.9), vjust = -0.5) +
  labs(title = "Attrition by the Frequency of Business Travel",
       x = "Attrition",
       y = "Number of Employees",
       fill = "Frequency of Business Travel") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

## COMPETITION PORTION ##

# import dataset
library(readr)
CaseStudy1CompSet_No_Attrition <- read_csv("SMU - M.S. Data Science/CaseStudy1CompSet No Attrition.csv")
View(CaseStudy1CompSet_No_Attrition)

# apply existing NB model to this dataset to predict attrition
predicted_attrition <- predict(nb_model, newdata = CaseStudy1CompSet_No_Attrition)

# add predictions to the new dataset
CaseStudy1CompSet_No_Attrition$PredictedAttrition <- predicted_attrition
View(CaseStudy1CompSet_No_Attrition) # contains PredictedAttrition

# creating new df containing ID and PredictedAttrition
pred_attrition_df <- CaseStudy1CompSet_No_Attrition[, c("ID", "PredictedAttrition")]
head(pred_attrition_df)

# convert pred_attrition_df to csv file
write.csv(pred_attrition_df, "Case1PredictionsVu_Attrition.csv", row.names = FALSE)