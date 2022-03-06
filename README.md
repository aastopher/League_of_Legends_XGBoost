League of Legends XGboost Model
================
Aaron Stopher, Luke Moore, Kristoffer Sorensen </br>
2022-03-04

DATA SOURCE: https://www.kaggle.com/paololol/league-of-legends-ranked-matches
#### [Project Summary](./Summary.md)

## R Library Imports

``` r
library(tidyverse)
library(xgboost)
library(Matrix)
library(stats)
library(data.table)
library(caret)
library(ggplot2)
library(dplyr)
library(Ckmeans.1d.dp)
library(rpart)
library(rpart.plot)
library(pROC)
```

## Github Library Imports

``` r
install.packages("devtools")
library(devtools)
install_github("AppliedDataSciencePartners/xgboostExplainer")
library(xgboostExplainer)
```

## Import & Format Data

``` r
# Raw Import
dat<-read.csv("res/data/Allstatshead.csv", header=TRUE, sep=",")

# Subset data
data.all <- subset(dat, select =c("win","player","item1","trinket","kills","deaths","assists","ownjunglekills","enemyjunglekills","visionscore","firstblood"))

# Change player,item1, and trinket to factors
data.all$player <- as.factor(data.all$player)
data.all$item1 <- as.factor(data.all$item1)
data.all$trinket <- as.factor(data.all$trinket)

# Check Dataframe
str(data.all)
```

    ## 'data.frame':    10000 obs. of  11 variables:
    ##  $ win             : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ player          : Factor w/ 10 levels "1","2","3","4",..: 1 2 3 4 5 6 7 8 9 10 ...
    ##  $ item1           : Factor w/ 134 levels "0","1001","1011",..: 40 40 40 40 40 40 40 40 40 40 ...
    ##  $ trinket         : Factor w/ 5 levels "0","3340","3341",..: 5 5 5 5 5 5 5 5 5 5 ...
    ##  $ kills           : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ deaths          : int  2 2 2 2 2 2 2 2 2 2 ...
    ##  $ assists         : int  12 12 12 12 12 12 12 12 12 12 ...
    ##  $ ownjunglekills  : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ enemyjunglekills: int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ visionscore     : int  30 30 30 30 30 30 30 30 30 30 ...
    ##  $ firstblood      : int  0 0 0 0 0 0 0 0 0 0 ...

## Creating Train & Test Data

``` r
# Set seed for repeatable results
set.seed(123456)

# Split data 70-30
data.split <- sample(sample(c(rep(1, 0.7 * nrow(data.all)),rep(0, 0.3 * nrow(data.all)))))

# Set training and test set respectively
dat.train <- data.all[data.split==1,]
dat.test <- data.all[data.split==0,]

# Prepare for matrix
setDT(dat.train)
setDT(dat.test)

# Create train matrix and response vector
train <- model.matrix(~.+0,data = dat.train[,-c("win"),with=F])
trainresults <- dat.train$win

# Create test matrix and response vector
test <- model.matrix(~.+0,data = dat.test[,-c("win"),with=F])
testresults <- dat.test$win
```

## Training a Decision Tree

``` r
# This block will train a tree model and output the resulting modeled tree graphs. This will help us visualize a clear path a particular record may take and what the prediction would be.

# Create folds for response vector
cv <- createFolds(trainresults, k = 10)

# Train Tree with Cross-Validation
tree.cv <- train(x = train, y = as.factor(trainresults), method = "rpart2", tuneLength = 4, # Note that y MUST be as factor!
                 trControl = trainControl(method = "cv",index = cv), control = rpart.control())
tree.model = tree.cv$finalModel
```

## Plot Trees

``` r
rpart.plot(tree.model,type = 2,extra = 7,fallen.leaves = T, main='Tree win probabilities') # extra = 7: the probability of the second class only. Useful for binary responses.
```

![](res/figure-gfm/plotting%20trees%201-1.png)<!-- -->

<span style="text-align:center; font-weight:bold;">Plot showing the
probability of win outcome at each node</span>

``` r
rpart.plot(tree.model,type = 2,extra = 2,fallen.leaves = T, main='Tree classification rate') # extra = 2: display the classification rate at the node, expressed as the number of correct classifications and the number of observations in the node.
```

![](res/figure-gfm/plotting%20trees%202-1.png)<!-- -->

<span style="text-align:center; font-weight:bold;">Plot showing the
classification rate at each node</span>

``` r
# Make Predictions based off our trained tree model
tree.preds = predict(tree.model, as.data.frame(test))[,2]
tree.roc_obj <- roc(testresults, tree.preds)
```

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

### Model Accuracy Results

``` r
cat("Tree AUC ", auc(tree.roc_obj))
```

    ## Tree AUC  0.7885916

## Training an XGBoost Model

``` r
# WARNING: Block will take a while to run!

# This block will train an XGBoost model and output an AUC model accuracy metric comparison against the tree model. This is meant to illustrate the model accuracy improvement when using gradient boosting, specifically XGBoost.

# Format our train data for the model
xgb.train.data = xgb.DMatrix(train, label = trainresults, missing = NA)

# Define what type of prediction we are doing
param <- list(objective = "binary:logistic", base_score = 0.5)

# Train our XGBoost model with Cross-Validation
xgboost.cv = xgb.cv(param=param, data = xgb.train.data, folds = cv, nrounds = 1500, early_stopping_rounds = 100, metrics='auc')
best_iteration = xgboost.cv$best_iteration

# Create XGBoost model object with our best_iteration
xgb.model <- xgboost(param=param, data=xgb.train.data, nrounds=best_iteration)

# Format our test data for prediction
xgb.test.data = xgb.DMatrix(test, missing = NA)

# Make Predictions based off our trained XGBoost model
xgb.preds = predict(xgb.model, xgb.test.data)
xgb.roc_obj <- roc(testresults, xgb.preds)
```

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

### Model Accuracy Results

``` r
cat("Tree AUC ", auc(tree.roc_obj))
```

    ## Tree AUC  0.7885916

``` r
cat("XGB AUC ", auc(xgb.roc_obj))
```

    ## XGB AUC  0.9999976

## XGBoost Importance Matrix

``` r
# For the purposes of visualization we are going to grab just the top (most predictive) features.

# Define number of top features we would like to use in our reduced model.
top_n_features = 10

# Create importance matrix
col_names = attr(xgb.train.data, ".Dimnames")[[2]]
importance_matrix <- xgb.importance(col_names,xgb.model)
```

``` r
# Plot Features in order of importance based on the 'Gain' measurement
gg <- xgb.ggplot.importance(importance_matrix, top_n = top_n_features, measure = "Gain", rel_to_first = TRUE)
gg + ggplot2::ylab("Importance")
```

![](res/figure-gfm/plotting%20importance%20matrix%20gain-1.png)<!-- -->

<span style="text-align:center; font-weight:bold;">This plot will help
us identify the most important predictive features</span>

``` r
# Plot Features in order of importance based on the 'Frequency' measurement
gg <- xgb.ggplot.importance(importance_matrix, top_n = top_n_features, measure = "Frequency", rel_to_first = TRUE)
gg + ggplot2::ylab("Frequency") + ggtitle('Feature frequency')
```

![](res/figure-gfm/plotting%20importance%20matrix%20frequency-1.png)<!-- -->

<span style="text-align:center; font-weight:bold;">This plot shows an
arguably more accurate representation of the ‘weightiness’ of each
feature in the model</span>

## Re-Train XGBoost with reduced importance model - (top\_n\_features = 10)

``` r
# WARNING: Block will take a while to run!

# Create reduced Dataframe for the purposes of visualizing our model further
reduced_imp <- head(importance_matrix[order(importance_matrix$Frequency, decreasing=TRUE), ], top_n_features)
train.reduced <- train[,reduced_imp$Feature]
test.reduced <- test[,reduced_imp$Feature]

# Format our reduced train data for the model
xgb.train.data.reduced = xgb.DMatrix(train.reduced, label = trainresults, missing = NA)

# Define what type of prediction we are doing
param <- list(objective = "binary:logistic", base_score = 0.5)

# Train our XGBoost model with Cross-Validation
xgboost.cv.reduced = xgb.cv(param=param, data = xgb.train.data.reduced, folds = cv, nrounds = 1500, early_stopping_rounds = 100, metrics='auc')
best_iteration = xgboost.cv.reduced$best_iteration

# Create XGBoost reduced model object with our best_iteration
xgb.model.reduced <- xgboost(param =param,  data = xgb.train.data.reduced, nrounds=best_iteration)

# Format our reduced test data for prediction
xgb.test.data.reduced = xgb.DMatrix(test.reduced, missing = NA)

# Make Predictions based off our trained XGBoost reduced model
xgb.preds.reduced = predict(xgb.model.reduced, xgb.test.data.reduced)
xgb.roc_obj.reduced <- roc(testresults, xgb.preds.reduced)
```

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

### Model Accuracy Results

``` r
cat("Tree AUC ", auc(tree.roc_obj))
```

    ## Tree AUC  0.7885916

``` r
cat("XGB AUC ", auc(xgb.roc_obj.reduced))
```

    ## XGB AUC  0.9999871

## XGBoost Explainer Visuals

``` r
# WARNING: Block will take a while to run!

# Create a model explainer object and prediction breakdown object for visualization.
explainer = buildExplainer(xgb.model.reduced,xgb.train.data.reduced, type="binary", base_score = 0.5, trees_idx = NULL)
pred.breakdown = explainPredictions(xgb.model.reduced, explainer, xgb.test.data.reduced)
```

## Converting log-odds to probabilities

``` r
# Get weights for log-odds conversion
weights = rowSums(pred.breakdown)

# Convert log-odds to probabilities
pred.xgb.odds = 1/(1+exp(-weights))

# List the maximum predicted win probability
cat(max(xgb.preds.reduced-pred.xgb.odds),'\n')
```

    ## 9.640165e-08

## Produce a waterfall chart for the last predicted record

``` r
# Define row index
idx_to_get = as.integer(nrow(test.reduced))

# Create waterfall plot for a specific record and it's associated prediction weights by feature
showWaterfall(xgb.model.reduced, explainer, xgb.test.data.reduced,test.reduced,idx_to_get, type = "binary")
```

    ##
    ##
    ## Extracting the breakdown of each prediction...
    ##
    ## DONE!
    ##
    ## Prediction:  0.04847498
    ## Weight:  -2.977018
    ## Breakdown
    ##        intercept           deaths          assists      trinket3340
    ##      0.003203373     -3.353259779      2.079234101     -0.979936364
    ##      trinket3364            kills enemyjunglekills      visionscore
    ##     -0.901846784      0.666465659     -0.597400484      0.432727868
    ##   ownjunglekills       firstblood        item11055
    ##     -0.148696027     -0.128740917     -0.048768889

![](res/figure-gfm/creating%20waterfall%20chart-1.png)<!-- -->

<span style="text-align:center; font-weight:bold;">This plot shows an
individual view of ‘weightiness’ for each feature in the model on a
specific prediction</span>

## Variable Impact on log-odds

``` r
# Create scatter plot for the deaths feature where 'x' is the number of deaths and 'y' is impact on log-odds
plot(test.reduced[,"deaths"], as.matrix(pred.breakdown[,"deaths"]), main="Impact on logg-odds (Deaths)", cex=0.4, pch=16, xlab = "number of Deaths", ylab = "impact on log-odds")
```

![](res/figure-gfm/plotting%20log-odds%20impact%20deaths-1.png)<!-- -->

<span style="text-align:center; font-weight:bold;">This plot shows the
correlation between deaths and impact on log-odds</span>

``` r
# Create scatter plot for the assists feature where 'x' is the number of assists and 'y' is impact on log-odds
plot(test.reduced[,"assists"], as.matrix(pred.breakdown[,"assists"]), main="Impact on logg-odds (Assists)", cex=0.4, pch=16, xlab = "number of assists", ylab = "impact on log-odds")
```

![](res/figure-gfm/plotting%20log-odds%20impact%20assists-1.png)<!-- -->

<span style="text-align:center; font-weight:bold;">This plot shows the
correlation between assists and impact on log-odds</span>

``` r
# Create scatter plot for the kills feature where 'x' is the number of kills and 'y' is impact on log-odds
plot(test.reduced[,'kills'], as.matrix(pred.breakdown[,"kills"]), main="Impact on logg-odds (Kills)", cex=0.4, pch=16, xlab = "number of kills", ylab = "kills impact on log-odds")
```

![](res/figure-gfm/plotting%20log-odds%20impact%20kills-1.png)<!-- -->

<span style="text-align:center; font-weight:bold;">This plot shows the
correlation between kills and impact on log-odds</span>

``` r
# Create scatter plot for the actual and predicted deaths feature where 'x' is both the actual(blue) and predicted(red) with 'y' being impact on log-odds
clr <- c("blue","red")
plot(test.reduced[,'deaths'], as.matrix(pred.breakdown[,"deaths"]),
main="Actual vs Predicted logg-odds (Deaths)",
xlab = "number of deaths",
ylab="impact on logg-odds",
type="p",col=clr,cex=0.4, pch=16)
legend("topright",
c("actual","predicted"),
fill=clr)
```

![](res/figure-gfm/plotting%20log-odds%20actual%20vs%20predicted%20(deaths)-1.png)<!-- -->

<span style="text-align:center; font-weight:bold;">This plot shows the
correlation between deaths and impact on log-odds, with the additional
context of actual values vs predicted values impact.</span>
