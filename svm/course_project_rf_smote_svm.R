library('class')
library('smotefamily')
library('caret')
library('randomForest')
library('dplyr')
library('ROSE')
library('e1071')

library('VIM')     #for.KNN.imputation 

#reading csv file -------
data<-read.csv('Churn_Modelling.csv')

#converting from int to factor ----------
data$Exited<-as.numeric(data$Exited)
data$IsActiveMember<-as.numeric(data$IsActiveMember)
data$HasCrCard<-as.numeric(data$HasCrCard)

data$RowNumber<-as.numeric(data$RowNumber)
data$CustomerId<-as.numeric(data$CustomerId)
data$CreditScore<-as.numeric(data$CreditScore)
data$Age<-as.numeric(data$Age)
data$Tenure<-as.numeric(data$Tenure)
data$NumOfProducts<-as.numeric(data$NumOfProducts)


#converting 0 balance to NA ---------

data$Balance[data$Balance == 0 ] = NA
# data.numeric <- select_if(data, is.numeric)


# randomly reorder the data-------

set.seed(1234)                                        # setting a seed for sample function
new.data<-data[sample(1:nrow(data)),]

#removing outliers --------
new.data.modified<-subset(new.data,new.data$CreditScore>405 & Age<65)

#rf impute imputation
df2 <- new.data.modified[,-c(3,6,5)] # removing three columns having char type
rf.data<-new.data.modified 
set.seed(123)
new.rf.data<-rfImpute( Exited~.,data = df2)   # imputing empty values






#correcting class imbalance ---------
table(new.rf.data$Exited)
prop.table(table(new.rf.data$Exited))
smote_out=SMOTE(X=new.rf.data,target=new.rf.data$Exited,K=3,dup_size =3)
new.rf.data=smote_out$data
table(new.rf.data$Exited)
prop.table(table(new.rf.data$Exited))


new.rf.data<-new.rf.data[,-12]
new.rf.data$Exited=as.factor(new.rf.data$Exited)


#train test-------
set.seed(12345)
sample.size<-sample(2,nrow(new.rf.data),replace=TRUE , prob = c(0.7,0.3))
train<- new.rf.data[sample.size==1,]
test<- new.rf.data[sample.size==2,]



# setting classifier


classifier=svm(formula=Exited~.,
               data=train,
               type='C-classification',
               kernel = 'radial'
)

#prediction using test 
pred=predict(classifier,newdata = train)






#evaluation

cm=table(train$Exited,pred)


prec1<- (cm[1,1])/sum(cm[1,1],cm[2,1])
rec1<- (cm[1,1])/sum(cm[1,1],cm[1,2])








confusionMatrix(train$Exited,pred)          #Accuracy : 
cat("\n precision",prec1)               #precision : 
cat("\n recall ",rec1)                  #recall : 
cat("\n f-score:",2*(prec1*rec1)/(prec1+rec1))   #f-score : 









#setting classifier - linear



classifier.1=svm(formula=Exited~.,
                 data=train,
                 type='C-classification',
                 kernel = 'linear'
)

#prediction using test 
pred.1=predict(classifier.1,newdata = train)






#evaluation

cm.1=table(train$Exited,pred.1)


prec1.1<- (cm.1[1,1])/sum(cm.1[1,1],cm.1[2,1])
rec1.1<- (cm.1[1,1])/sum(cm.1[1,1],cm.1[1,2])




confusionMatrix(train$Exited,pred.1)          #Accuracy :  
cat("\n precision",prec1.1)               #precision : 
cat("\n recall ",rec1.1)                  #recall : 
cat("\n f-score:",2*(prec1.1*rec1.1)/(prec1.1+rec1.1))   #f-score :








