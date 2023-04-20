library('class')
library('smotefamily')
library('caret')
library('randomForest')
library('dplyr')
library('ROSE')

library('VIM')     #for.KNN.imputation 

#reading csv file -------
data<-read.csv('Churn_Modelling.csv')

#converting from int to factor ----------
data$Exited<-as.factor(data$Exited)
data$IsActiveMember<-as.factor(data$IsActiveMember)
data$HasCrCard<-as.factor(data$HasCrCard)




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
over<-ovun.sample(Exited~.,data=new.rf.data , method = 'over' , N=15446)$data
under<-ovun.sample(Exited~.,data=new.rf.data , method = 'under' , N=3944)$data
both<-ovun.sample(Exited~.,data=new.rf.data , method = 'both' , 
                  p=0.5,  seed=123,  N=7723)$data


#train test-------

#no
set.seed(12345)
sample.size<-sample(2,nrow(new.rf.data),replace=TRUE , prob = c(0.7,0.3))
train<- new.rf.data[sample.size==1,]
test<- new.rf.data[sample.size==2,]


#over
set.seed(12345)
sample.size.over<-sample(2,nrow(over),replace=TRUE , prob = c(0.7,0.3))
train.over<- over[sample.size.over==1,]
test.over<- over[sample.size.over==2,]



#under
set.seed(12345)
sample.size.under<-sample(2,nrow(under),replace=TRUE , prob = c(0.7,0.3))
train.under<- under[sample.size.under==1,]
test.under<- under[sample.size.under==2,]


#both
set.seed(12345)
sample.size.both<-sample(2,nrow(both),replace=TRUE , prob = c(0.7,0.3))
train.both<- both[sample.size.both==1,]
test.both<- both[sample.size.both==2,]


#checking class imbalance--
summary(train$Exited)

#model----------
rf.train <-randomForest(Exited~.,data=train)
rf.over <-randomForest(Exited~.,data=train.over)
rf.under <-randomForest(Exited~.,data=train.under)
rf.both <-randomForest(Exited~.,data=train.both)



#evaluation------

df1<-table(predict(rf.train,test),test$Exited)
prec<- df1[1,1]/sum(df1[1,1],df1[2,1])
rec<- df1[1,1]/sum(df1[1,1],df1[1,2])

confusionMatrix(predict(rf.train,test),test$Exited ) #Accuracy :   Sensitivity :   Specificity :    
cat("\n precision",prec)               
cat("\n recall ",rec)                  
cat("\n f-score:",2*(prec*rec)/(prec+rec))   




df2<-table(predict(rf.over,test),test$Exited)
prec2<- df2[1,1]/sum(df2[1,1],df2[2,1])
rec2<- df2[1,1]/sum(df2[1,1],df2[1,2])

confusionMatrix(predict(rf.over,test),test$Exited )  #Accuracy :  Sensitivity : Specificity : 
df2<-table(predict(rf.over,test),test$Exited)
cat("\n precision",prec2)               
cat("\n recall ",rec2)                 
cat("\n f-score:",2*(prec2*rec2)/(prec2+rec2))   





df3<-table(predict(rf.under,test),test$Exited)
prec3<- df3[1,1]/sum(df3[1,1],df3[2,1])
rec3<- df3[1,1]/sum(df3[1,1],df3[1,2])

confusionMatrix(predict(rf.under,test),test$Exited )  #Accuracy :   Sensitivity :  Specificity :   
cat("\n precision",prec3)                
cat("\n recall ",rec3)                  
cat("\n f-score:",2*(prec3*rec3)/(prec3+rec3))   







df4<-table(predict(rf.both,test),test$Exited)
prec4<- df4[1,1]/sum(df4[1,1],df4[2,1])
rec4<- df4[1,1]/sum(df4[1,1],df4[1,2])


confusionMatrix(predict(rf.both,test),test$Exited )   #Accuracy :   Sensitivity :  Specificity : 
cat("\n precision",prec4)                
cat("\n recall ",rec4)                   
cat("\n f-score:",2*(prec4*rec4)/(prec4+rec4))   

