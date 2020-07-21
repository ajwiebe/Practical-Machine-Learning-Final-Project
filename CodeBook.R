url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url1, destfile = "training.csv")
download.file(url2, destfile = "testing.csv")
training <- read.csv("training.csv")
testing <- read.csv("testing.csv")
library(caret)
library(AppliedPredictiveModeling)
library(ggplot2)
library(rpart)
library(gbm)

training <- training[, colSums(is.na(training)) == 0]
testing <- testing[, colSums(is.na(testing))== 0]
testing <- testing[, -c(1:7)]
training <- training[, -c(1:7)]

set.seed(15)
inTrain <- createDataPartition(training$classe, p = 0.7, list =F)
train <- training[inTrain, ]
test <- training[-inTrain, ]
validation <- testing
nearzero <- nearZeroVar(train)
nearzero2 <- nearZeroVar(test)
train <- train[, -nearzero]
test <- test[, -nearzero2]

##rf, gbm, lda
set.seed(15)
rfControl <- trainControl(method = "cv", number = 5, verboseIter = F)
model1 <- train(classe~., data = train, method = "rf", trControl = rfControl)
set.seed(15)
gbmControl <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
model2 <- train(classe~., data = train, method = "gbm", trControl = gbmControl, verbose = F)
set.seed(15)
ldaControl <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
model3 <- train(classe~., data = train, method = "lda", trControl = ldaControl)
pred1 <- predict(model1, newadta= test)
pred2 <- predict(model2, newdata = test)
pred3 <- predict(model3, newdata = test)
predDF <- data.frame(pred1, pred2, pred3, classe = test$classe, cache = T)
combModel <- train(classe~., method = "rf", data = predDF)
combPred <- predict(combModel, test)
m1 <- confusionMatrix(pred1, as.factor(test$classe))$overall
m1
m2 <- confusionMatrix(pred2, as.factor(test$classe))$overall
m2
m3 <- confusionMatrix(pred3, as.factor(test$classe))$overall
m3
## clearly the worst one 
cm <- confusionMatrix(combPred, as.factor(test$classe))$overall

acs <- c(0.9918437, 0.9651657, 0.7033135, 0.9918437)
OutSampError <- 1 - acs
print(OutSampError)

plot(model1, main = "Random Forest Error vs Trees", xlab = "Trees")
plot(model2, main = "GBM Accuracy vs Number of Boosting Iterations")

predValidation <- predict(model1, newdata = validation) 
predValidation 