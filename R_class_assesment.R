
###########################################################################
###########################################################################
###                                                                     ###
###                         R CLASS ASSESSMENT                          ###
###                                                                     ###
###########################################################################
###########################################################################



#################################################################
##                     Importing Libraries                     ##
#################################################################

library(psych)
library(RColorBrewer)
library(VIM)
library(tree)
library(randomForest)
library(caret)
library(ROCR)
library(dplyr)
library(class)
library(corrplot)
library(Amelia)
library(mlbench)


##################################################################
##                      Importing Dataset                       ##
##################################################################
df = read.csv('C:/Users/Darshan/Desktop/R Logistic Regression + DT + KNN + RF + SVM/biodeg.csv',
              sep = ';', stringsAsFactors = T, header = F)



#################################################################
##                  Exploratory Data Analysis                  ##
#################################################################

View(df)

str(df)

dim(df)

describe(df)

correlations <- cor(df[,1:41])
corrplot(correlations, method="circle")

# Missing values
sapply(df, function(x) sum(is.na(x)))
sapply(df, function(x) sum(is.null(x)))
missmap(df, col=c("blue", "red"), legend=FALSE)

# There are no missing values

##################################################################
##                       Train test split                       ##
##################################################################

set.seed(121)
select_rows = sample(1:nrow(df), round(0.8*nrow(df)), replace = F)
train = df[select_rows,]
test = df[-select_rows,]

dim(train)
dim(test)

#################################################################
##                        Decision Tree                        ##
#################################################################


dt_model = tree(V42~.-V42, data=train)

plot(dt_model, type='uniform')
text(dt_model, pretty=0, cex=0.6)

summary(dt_model)

pred_train = predict(dt_model, train, type = 'class')
confusionMatrix(pred_train, train$V42)

pred_test = predict(dt_model, test, type = 'class')
confusionMatrix(pred_test, test$V42)


#################################################################
##                     Logistic regression                     ##
#################################################################

# Top variables used for decision tree construction are: 
# [1] "V36" "V34" "V1"  "V10" "V39" "V9"  "V13" "V30" "V35" "V40" "V12" "V3"  "V11" "V27" "V7" 

# we can use these variables for logistic regression

log_train = train[,c("V42", "V36", "V34", "V1",  "V10", "V39", "V9",  "V13", "V30",
                     "V35", "V40", "V12" ,"V3" , "V11", "V27" ,"V7" )]

log_test = test[,c("V42", "V36", "V34", "V1",  "V10", "V39", "V9",  "V13", "V30",
                   "V35", "V40", "V12" ,"V3" , "V11", "V27" ,"V7" )]
dim(log_train)
dim(log_test)

glm.fit <- glm(V42 ~ .-V42, data = log_train, family = binomial)

summary(glm.fit)

new_log_train = train[, c('V42', 'V34', 'V1', 'V10', 'V9',
                          'V40', 'V3', 'V11', 'V7')]
new_log_test = test[, c('V42', 'V34', 'V1', 'V10', 'V9',
                        'V40', 'V3', 'V11', 'V7')]

log_model <- glm(V42 ~ .-V42,
               data = new_log_train, family = binomial)
summary(log_model)

pred <- predict(log_model, type = 'response')

table(new_log_train$V42, pred > 0.5)

pred_train


#################################################################
##                        Random Forest                        ##
#################################################################

rf <- randomForest(V42~., data = train)

pred = predict(rf, newdata = test)

table(test$V42, pred)



#################################################################
##                     K Nearest Neighbors                     ##
#################################################################

train_knn <- train %>%
  mutate(V42 = ifelse(V42 == "NRB",0,1))

test_knn <- test %>%
  mutate(V42 = ifelse(V42 == "NRB",0,1))


#Build KNN Model - from class library
model = knn(train = train_knn, test = test_knn, 
            cl = train_knn$V42, k = 5)

pred_tab = table(model, test_knn$V42)
print(pred_tab)

#Accuracy
sum(diag(pred_tab))/sum(pred_tab)

#To find the optimal value of K
Accuracy = NULL;

for (i in seq(1,25,2)){
  model = knn(train = train_knn, test = test_knn, 
              cl = train_knn$V42, k = i)
  pred_tab = table(model, test_knn$V42)
  y = round(sum(diag(pred_tab))/sum(pred_tab),4)
  Accuracy = rbind(Accuracy,y) #appending 
  print(paste('for K is',i,'accuracy is',y))
}


plot(x = seq(1,25,2), y = Accuracy, 
     xlab='K Values', ylab = 'Accuracy', 
     col='red', type='b', pch=19,
     main='Accuracy vs K Values',xaxt="none", mgp=c(3,1,0), 
     panel.first = grid(), las=1)
axis(1,seq(1,25,2))


##################################################################
##                    Support Vector Machine                    ##
##################################################################


library(e1071)

#Building the Model - Default - Radial Basis (RBF)
model1 = svm(V42~.-V42, data = train)

#Prediction & Accuracy
pred1 = predict(model1, test[,-42])
pred_tab1 = table(pred1, test$V42)
print(pred_tab1)


sum(diag(pred_tab1))/sum(pred_tab1)

#_________________2nd...._______Building the Model2 - Linear
model2 = svm(V42~.-V42, data = train, kernel='linear')
#Prediction & Accuracy
pred2 = predict(model2, test[,-42])
pred_tab2 = table(pred2, test$V42)
print(pred_tab2)


sum(diag(pred_tab2))/sum(pred_tab2)

#_____________3rd.........Building the Model3 - Polynomial
model3 = svm(V42~.-V42, data = train, kernel='polynomial')
#Prediction & Accuracy
pred3 = predict(model3, test[,-42])
pred_tab3 = table(pred3, test$V42)
print(pred_tab3)


sum(diag(pred_tab3))/sum(pred_tab3)

#_____________4th...Building the Model4 - Sigmoid
model4 = svm(V42~.-V42, data = train, kernel='sigmoid')
#Prediction & Accuracy
pred4 = predict(model4, test[,-42])
pred_tab4 = table(pred4, test$V42)
print(pred_tab4)


sum(diag(pred_tab4))/sum(pred_tab4)

