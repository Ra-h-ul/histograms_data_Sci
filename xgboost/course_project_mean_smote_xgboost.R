# packages
library('xgboost')
library('dplyr')
library('magrittr')
library('Matrix')
library('smotefamily')
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
data.numeric <- select_if(data, is.numeric)


# randomly reorder the data-------

set.seed(1234)                                        # setting a seed for sample function
new.data<-data[sample(1:nrow(data)),]


#removing outliers --------
new.data.modified<-subset(new.data,new.data$CreditScore>405 & Age<65)


#------- mean imputation
new.mean.data <- new.data[,-c(3,6,5)]
new.mean.data$Balance[which(is.na(new.mean.data$Balance))]=mean(new.mean.data$Balance,na.rm=TRUE)

 




#correcting class imbalance ---------
table(new.mean.data$Exited)
prop.table(table(new.mean.data$Exited))
smote_out=SMOTE(X=new.mean.data,target=new.mean.data$Exited,K=3,dup_size =3)
new.mean.data=smote_out$data
table(new.mean.data$Exited)
prop.table(table(new.mean.data$Exited))


new.mean.data<-new.mean.data[,-12]

#train test-------
set.seed(12345)
sample.size<-sample(2,nrow(new.mean.data),replace=TRUE , prob = c(0.7,0.3))
train<- new.mean.data[sample.size==1,]
test<- new.mean.data[sample.size==2,]


#create matrix ----------------------------------------------------------------

#no sampling
trainm<- sparse.model.matrix(Exited~. , -11 , data=train)
train_label <-train[,"Exited"]
train_matrix<-xgb.DMatrix( data = as.matrix(trainm), label=train_label)

testm <- sparse.model.matrix(Exited~. , -11, data=test)
test_label<-test[,"Exited"]
test_matrix<-xgb.DMatrix(data=as.matrix(testm)  , label = test_label)


#parameters--------------------------------


nc<-length(unique(train_label))
xgb_params<-list("objective" = "multi:softprob",
                 "eval_metric" = "mlogloss" ,
                 "num_class"=nc)
watchlist<-list(train = train_matrix , test=test_matrix )




#model--------------------------------------------


bst_model <-xgb.train(params = xgb_params ,
                      data=train_matrix,
                      nround = 100 ,
                      watchlist = watchlist,
                      eta = 0.05,
                      max.depth = 6,
                      gamma = 2,)

#confusion matrix----------------------------------------------------------


p<-predict(bst_model , newdata = test_matrix)
pred<-matrix(p,nrow=nc,ncol=length(p)/nc) %>%
  t()%>%
  data.frame()%>%
  mutate(label=test_label , max_prob =max.col(.,"last")-1)

t<-table(prediction = pred$max_prob , Actual = pred$label)

acc<-sum(t[1,1],t[2,2])/sum(t[1,1],t[1,2],t[2,2],t[2,1])
sen<-t[1,1]/sum(t[1,1],t[2,1])
spe<-t[2,2]/sum(t[2,2],t[1,2])
pre<-t[1,1]/sum(t[1,1],t[1,2])
rec<-t[1,1]/sum(t[1,1],t[2,1])
f_s<- 2*pre*rec/sum(pre,rec);






#printing result
cat("accuracy : sensitivity : specificity : precision : recall : f_Score")
cat(acc,sen,spe,pre,rec,f_s)
cat("\n")




#analysis
#plot
e<-data.frame(bst_model$evaluation_log)
plot(e$iter , e$train_mlogloss , col='blue')
lines(e$iter , e$test_mlogloss , col = 'red')

