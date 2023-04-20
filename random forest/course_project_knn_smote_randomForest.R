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


#KNN imputation-----
new.knn.data <- new.data.modified[,-c(3,6,5)]
new.knn.imputed<-kNN(new.knn.data , variable= c('Balance') , k=7)
new.knn.imputed<-subset(new.knn.imputed , select = RowNumber : Exited )

#correcting class imbalance ---------
prop.table(table(new.knn.imputed$Exited))
smote_out=SMOTE(X=new.knn.imputed,target=new.knn.imputed$Exited,K=3,dup_size =3)
new.knn.imputed=smote_out$data
table(new.knn.imputed$Exited)
prop.table(table(new.knn.imputed$Exited))


new.knn.imputed<-new.knn.imputed[,-12]
new.knn.imputed$Exited=as.factor(new.knn.imputed$Exited)
new.knn.imputed$Exited=as.factor(new.knn.imputed$Exited)





#train test-------
set.seed(12345)
sample.size<-sample(2,nrow(new.knn.imputed),replace=TRUE , prob = c(0.7,0.3))
train<- new.knn.imputed[sample.size==1,]
test<- new.knn.imputed[sample.size==2,]

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







