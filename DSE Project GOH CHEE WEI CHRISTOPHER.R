rm(list = ls()) #clear all         

library(readxl) #readexcel
library(tree) #tree
library(MASS)
library(kknn) # allows us to do KNN for regression and classification
data = read_excel("~/Desktop/DSE1101/HDB_resale_prices/HDB_data_2021_sample.xlsx")

###############
#Data Cleaning#
###############

sum(is.na(data)) #locate NA values

data=na.omit(data) #remove NA values

data$resale_price <- data$resale_price / 1000 #divide resale_price by 1000

##################
#Train-Test split#
##################

ntrain=3000 # 50/50 split

set.seed(1234) # for reproducibility
tr = sample(1:nrow(data),ntrain)  # draw ntrain observations from original data
train = data[tr,]   # Training sample
test = data[-tr,]   # Testing sample

####################
#Variable selection#
####################

#Use trees for variable selection:

big.tree = tree(resale_price~.,data=train,mindev=0.0001)

#determine best tree size based on CV:

cv.bigtree = cv.tree(big.tree, , prune.tree) #10-fold cross-validation
bestcp = cv.bigtree$size[max(which(cv.bigtree$dev == min(cv.bigtree$dev)))]

bestcp #13 leaves is best

#prune the tree using the cross-validated size choice:
final.tree=prune.tree(big.tree,best=bestcp)

#plot the tree
plot(final.tree, type = "uniform")
text(final.tree, col = "blue", label = "yval", cex = 0.8)

treefit=predict(final.tree, newdata=test)

#check OOS MSE
mean((test$resale_price-treefit)^2) #MSE is 6496.478

#Variables shown are:
#1. floor_area_sqm
#2. max_floor_lvl
#3. Dist_CBD
#4. Remaining_lease
#5. flat_model_dbss

###############
#Model Fitting#
###############

#Fitting Multiple Linear Regression
lmfit1 = lm(resale_price ~ floor_area_sqm + max_floor_lvl + Dist_CBD + Remaining_lease + flat_model_dbss, data = train)

# Diagnostic plots
summary(lmfit1)
par(mfrow = c(2, 2))
plot(lmfit1)
dev.off()

#1st Check linearity
#Check residuals vs fitted plot if red line is horizontal then relation follows a linear pattern
#2nd Check normality
#Check Q-Q residuals plot if points follow a straight diagonal line, assumption for normality is met
#3rd Constant variablity
#Check scale-location plot for horizontal red line
#Check for influential values
#Check reiduals vs leverage if got outlying values

#Compute MSE
lmpred1 = predict(lmfit1,newdata=test) #prediction on test data
mean((test$resale_price-lmpred1)^2) #MSE is 4964.732

#Fitting Multiple Linear Regression on variables that I thought was crucial
lmfit2 = lm(resale_price ~ floor_area_sqm + Remaining_lease + Dist_nearest_station, data = train)
summary(lmfit2)

#Compute MSE
lmpred2 = predict(lmfit2,newdata=test) #prediction on test data
mean((test$resale_price-lmpred2)^2) #MSE is 13697.36

#Fitting Multiple Linear Regression on factors listed on HDB website
lmfit3 = lm(resale_price ~ floor_area_sqm + max_floor_lvl + Dist_CBD + Dist_nearest_station + Dist_nearest_mall, data = train)
summary(lmfit3)

#Compute MSE
lmpred3 = predict(lmfit3,newdata=test) #prediction on test data
mean((test$resale_price-lmpred3)^2) #MSE is 6885.307

#Fitting Multiple Linear Regression on all variables 
lmfit4 = lm(resale_price ~ ., data = train)
summary(lmfit4) 

#Compute MSE
lmpred4 = predict(lmfit4,newdata=test) #prediction on test data
mean((test$resale_price-lmpred4)^2) # MSE is 425400.6, worse than before

#Perform KNN using the variables that we selected, kmax = 100
resalecv=train.kknn(resale_price ~ floor_area_sqm + max_floor_lvl + Dist_CBD + Remaining_lease + flat_model_dbss, data = train,kmax=100, kernel = "rectangular")

#Examine the behavior of the LOOCV MSE:
plot((1:100),resalecv$MEAN.SQU, type="l", col = "blue", main="LOOCV MSE", xlab="Complexity: K", ylab="MSE")

#Find the best K:
kbest=resalecv$best.parameters$k
kbest

#We find K=4 works best according to LOOCV

#Fit for the selected K:
knnreg = kknn(resale_price ~ floor_area_sqm + max_floor_lvl + Dist_CBD + Remaining_lease + flat_model_dbss,train,test,k=kbest,kernel = "rectangular")

knnmse=mean((test$resale_price-knnreg$fitted.values)^2) #test set MSE
knnmse #MSE is 3129.765

#Perform KNN on all the variables
resalecv2=train.kknn(resale_price ~ ., data = train,kmax=100, kernel = "rectangular")

#Examine the behavior of the LOOCV MSE:
plot((1:100),resalecv2$MEAN.SQU, type="l", col = "blue", main="LOOCV MSE", xlab="Complexity: K", ylab="MSE")

#Find the best K:
kbest=resalecv2$best.parameters$k
kbest

#We find K=2 works best according to LOOCV

#Fit for the selected K:
knnreg2 = kknn(resale_price ~ .,train,test,k=kbest,kernel = "rectangular")

knnmse2=mean((test$resale_price-knnreg2$fitted.values)^2) #test set MSE
knnmse2 # MSE is 5818.863, worse than before

#Predicting resale price
hdbpred=predict(lmfit1, data.frame(floor_area_sqm = 146.0436, max_floor_lvl = 12, Dist_CBD = 9.4, Remaining_lease = 64, flat_model_dbss = 0))

hdbpred #predicted price is $705,932.2, actual price is $ 1,150,000
