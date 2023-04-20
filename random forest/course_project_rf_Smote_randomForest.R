library('class')
library('smotefamily')
library('caret')
library('randomForest')
library('dplyr')
library('ROSE')

library('VIM')     #for.KNN.imputation 

#reading csv file -------S
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
rf.data$Balance<-new.rf.data$Balance         # adding imputed balance column to the data having all variables


#correcting class imbalance ---------
prop.table(table(new.rf.data$Exited))
smote_out=SMOTE(X=new.rf.data,target=new.rf.data$Exited,K=3,dup_size =3)
new.rf.data=smote_out$data
table(new.rf.data$Exited)
prop.table(table(new.rf.data$Exited))


new.rf.data<-new.rf.data[,-12]
new.rf.data$Exited=as.factor(new.rf.data$Exited)
new.rf.data$Exited=as.factor(new.rf.data$Exited)





#train test-------
set.seed(12345)
sample.size<-sample(2,nrow(new.rf.data),replace=TRUE , prob = c(0.7,0.3))
train<- new.rf.data[sample.size==1,]
test<- new.rf.data[sample.size==2,]

#checking class imbalance--
summary(train$Exited)







#model----------
rf.train <-randomForest(Exited~.,data=train)


#evaluation------
df1<-table(predict(rf.train,test),test$Exited)
prec<- df1[1,1]/sum(df1[1,1],df1[2,1])
rec<- df1[1,1]/sum(df1[1,1],df1[1,2])

confusionMatrix(predict(rf.train,test),test$Exited ) 
cat("\n precision",prec)               
cat("\n recall ",rec)                  
cat("\n f-score:",2*(prec*rec)/(prec+rec))   







