# The highway management and control center in Muniville monitors the traffic state of a highway
# segment shown as Figure 1. Traffic volume and speed data were automatically and continuously
# collected from location 1, 2 and 3 when the loop detectors work well. However, the loop
# detectors are sometimes offline due to either a regular maintenance or malfunction. Usually, it
# takes about three business days for the loop detectors to be back online. For that reason, you are
# asked to use supervised learning method to estimate traffic state (i.e. volume and speed) for a
# location when its loop detector is absent.
# Given one-month historical data from location- 1, 2, and 3 (the data can be found in a separate
# spreadsheet.), you are asked to design a Deep Neural Network (DNN) for the purpose of traffic
# state estimation for location-3 when its loop detector is absent/malfunctional for a 3-day period.
# It is suggested that you split the historical data into three data sets; the first set is used to train the
# Deep Neural Network; and the second set (e.g. 15-20%) is used to validate your training; the
# third set which contains the last 3 days of data is used to test prediction accuracy of your neural
# network. The time interval is 60 minutes.


# Clear all ---------------------------------------------------------------
rm(list=ls())
gc()
cat("\014") 


# Libraries ---------------------------------------------------------------
library(tidyverse)
library(nnet)
library(neuralnet)
library(h2o)
library(profvis)
# Read data from CSV ------------------------------------------------------

# Crude_Data <- h2o.importFile("Problemset 2 data for problem 2.csv", 
#                              col.names = c("Volume.1", "Volume.2", "Volume.3", "Speed.1", "Speed.2", "Speed.3")
#                              )
# 
# Crude_Data <- Crude_Data[-c(1,2),];

Crude_Data <- read_csv("Problemset 2 data for problem 2.csv", skip = 1, col_names = TRUE);


for (i in 1:length(Crude_Data)) {
  if (i <= 3) {
    colnames(Crude_Data)[i] <- paste("Volume", i, sep = ".");
  }
  if (i > 3) {
    colnames(Crude_Data)[i] <- paste("Speed", i-3, sep = ".");
  }

}

Input_Data <- as.data.frame(c(Crude_Data[,1:2], Crude_Data[, 4:5]), check.names = FALSE);
Output_Data <- as.data.frame(c(Crude_Data[, 3], Crude_Data[, 6]), check.names = FALSE);

# Separate Datasets  ------------------------------------------------------

## The last 3 days of data correspond to the last (24 x 3) = 72 rows of the data set
## The test data is (744 - 72) x 20% =~ 135
## The train data is what's left (744 - 72 - 135) = 537
Input_Data_Training <- Input_Data[1:537, ];
Input_Data_Testing <- Input_Data[(537+1):(537+135), ];
Input_Data_Forecast <- Input_Data[(537 + 135 + 1):nrow(Input_Data), ];


Output_Data_Training <- Output_Data[1:537, ];
Output_Data_Testing <- Output_Data[(537+1):(537+135), ];
Output_Data_Forecast <- Output_Data[(537 + 135 + 1):nrow(Output_Data), ];

Training_Data <- cbind(Input_Data_Training, Output_Data_Training);
Validation_Data <- cbind(Input_Data_Testing, Output_Data_Testing);
Test_Data <- cbind(Input_Data_Forecast, Output_Data_Forecast);
# Training_Data <- Training_Data[, c(1,2,5,3,4,6)]
# Training_Data <- as.h2o(Training_Data);

Predictors <- names(Input_Data_Training);
Response <- names(Output_Data_Training);

profvis({
# Neural network Train ----------------------------------------------------
h2o.init(nthreads = -1)
# h2o.no_progress();
Model_DL_Speed3 <- list()
Model_DL_Speed3_Performance <- list()
Model_DL_Volume3 <- list()
Model_DL_Volume3_Performance <- list()

n <- 0;
for (i in 1:3) {
  if (i==1) {
    for (j in 5:10) {
      n <- n + 1;
      Model_DL_Speed3[n] <- list(h2o.deeplearning(x = Predictors, y = "Speed.3",
                                          training_frame = Training_Data %>% as.h2o(), 
                                          model_id = paste(i, "hidden layer with", j, "nodes", sep = " "), 
                                          hidden = c(j), validation_frame = Validation_Data %>% as.h2o()));
      Model_DL_Speed3_Performance[n] <- list(h2o.performance(Model_DL_Speed3[[n]], valid = TRUE))
      
      Model_DL_Volume3[n] <- list(h2o.deeplearning(x = Predictors, y = "Volume.3",
                                                  training_frame = Training_Data %>% as.h2o(), 
                                                  model_id = paste(i, "hidden layer with", j, "nodes", sep = " "), 
                                                  hidden = c(j), validation_frame = Validation_Data %>% as.h2o()));
      Model_DL_Volume3_Performance[n] <- list(h2o.performance(Model_DL_Volume3[[n]], valid = TRUE))
    }
  }
  if (i==2) {
    for (j in 5:10) {
      for (k in 5:10) {
        n <- n + 1;
        Model_DL_Speed3[n] <- list(h2o.deeplearning(x = Predictors, y = "Speed.3",
                                                    training_frame = Training_Data %>% as.h2o(), 
                                                    model_id = paste(i, "hidden layers with", j, "and", k, "nodes", sep = " "), 
                                                    hidden = c(j,k), validation_frame = Validation_Data %>% as.h2o()));
        Model_DL_Speed3_Performance[n] <- list(h2o.performance(Model_DL_Speed3[[n]], valid = TRUE))
        
        Model_DL_Volume3[n] <- list(h2o.deeplearning(x = Predictors, y = "Volume.3",
                                                    training_frame = Training_Data %>% as.h2o(), 
                                                    model_id = paste(i, "hidden layers with", j, "and", k, "nodes", sep = " "), 
                                                    hidden = c(j,k), validation_frame = Validation_Data %>% as.h2o()));
        Model_DL_Volume3_Performance[n] <- list(h2o.performance(Model_DL_Volume3[[n]], valid = TRUE))
      }
    }
  }
  if (i==3) {
    for (j in 5:10) {
      for (k in 5:10) {
        for (l in 5:10) {
          n <- n + 1;
          Model_DL_Speed3[n] <- list(h2o.deeplearning(x = Predictors, y = "Speed.3",
                                                      training_frame = Training_Data %>% as.h2o(), 
                                                      model_id = paste(i, "hidden layers with", j, ",", k, "and", l, "nodes", sep = " "), 
                                                      hidden = c(j, k, l), validation_frame = Validation_Data %>% as.h2o()));
          Model_DL_Speed3_Performance[n] <- list(h2o.performance(Model_DL_Speed3[[n]], valid = TRUE))
          
          Model_DL_Volume3[n] <- list(h2o.deeplearning(x = Predictors, y = "Volume.3",
                                                      training_frame = Training_Data %>% as.h2o(), 
                                                      model_id = paste(i, "hidden layers with", j, ",", k, "and", l, "nodes", sep = " "), 
                                                      hidden = c(j, k, l), validation_frame = Validation_Data %>% as.h2o()));
          Model_DL_Volume3_Performance[n] <- list(h2o.performance(Model_DL_Volume3[[n]], valid = TRUE))
          
          
          #print(paste(i, "hidden layers with", j, ",", k, "and", l, "nodes", sep = " "));
          
          
        }
      }
    }
  }
}



})

# Plotting MSE values for each Neural Network -----------------------------


Metrics_Speed3 <- data.frame(matrix(data = 0,nrow = length(Model_DL_Speed3_Performance), ncol = 5))
colnames(Metrics_Speed3) <- c("MSE.Train", "MSE.Valid", "RMSE.Train", "RMSE.Valid", "ModelID")
#Initializing the variable MSE_Speed3

for (i in 1:length(Model_DL_Speed3_Performance)) {
  Metrics_Speed3[i,] <- c(h2o.mse(Model_DL_Speed3[[i]], valid = TRUE, train = TRUE), 
                          h2o.rmse(Model_DL_Speed3[[i]], valid = TRUE, train = TRUE), 
                          Model_DL_Speed3[[i]]@model_id)
  ## Put the MSE values of every model into the variable MSE_Speed3 for a graph
}

ggplot(data = Metrics_Speed3, mapping = aes(x = as.character(ModelID))) + 
  geom_point(aes(y = as.numeric(MSE.Train), colour = "MSE.Train")) +
  geom_point(aes(y = as.numeric(MSE.Valid), colour = "MSE.Valid")) +
  ylim(0,50) + 
  theme(axis.ticks = element_line(size  =  0.2, linetype  =  'blank'),
        panel.grid.major = element_line(linetype  =  'blank'), 
        panel.grid.minor = element_line(colour  =  'ivory4'),
        axis.text.x = element_text(angle = 90, size = 7)) + 
  xlab("Number of layers and nodes") +
  ylab("MSE Error")  +
  scale_x_discrete(breaks = Metrics_Speed3$ModelID[c(T, F, F, F)])

ggplot(data = Metrics_Speed3, mapping = aes(x = as.character(ModelID))) + 
  geom_point(aes(y = as.numeric(RMSE.Train), colour = "MSE.Train")) +
  geom_point(aes(y = as.numeric(RMSE.Valid), colour = "MSE.Valid")) +
  ylim(0,10) + 
  theme(axis.ticks = element_line(size  =  0.2, linetype  =  'blank'),
        panel.grid.major = element_line(linetype  =  'blank'), 
        panel.grid.minor = element_line(colour  =  'ivory4'),
        axis.text.x = element_text(angle = 90, size = 7)) + 
  xlab("Number of layers and nodes") +
  ylab("RMSE Error")  +
  scale_x_discrete(breaks = Metrics_Speed3$ModelID[c(T, F, F, F)])

## Metrics for the Volume3 neural networks

Metrics_Volume3 <- data.frame(matrix(data = 0,nrow = length(Model_DL_Volume3_Performance), ncol = 5))
colnames(Metrics_Volume3) <- c("MSE.Train", "MSE.Valid", "RMSE.Train", "RMSE.Valid", "ModelID")
#Initializing the variable MSE_Speed3

for (i in 1:length(Model_DL_Volume3_Performance)) {
  Metrics_Volume3[i,] <- c(h2o.mse(Model_DL_Volume3[[i]], valid = TRUE, train = TRUE), 
                          h2o.rmse(Model_DL_Volume3[[i]], valid = TRUE, train = TRUE), 
                          Model_DL_Volume3[[i]]@model_id)
  ## Put the MSE values of every model into the variable MSE_Speed3 for a graph
}

ggplot(data = Metrics_Volume3, mapping = aes(x = as.character(ModelID))) + 
  geom_point(aes(y = as.numeric(MSE.Train), colour = "MSE.Train")) +
  geom_point(aes(y = as.numeric(MSE.Valid), colour = "MSE.Valid")) +
  ylim(0, 800000) + 
  theme(axis.ticks = element_line(size  =  0.2, linetype  =  'blank'),
        panel.grid.major = element_line(linetype  =  'blank'), 
        panel.grid.minor = element_line(colour  =  'ivory4'),
        axis.text.x = element_text(angle = 90, size = 7)) + 
  xlab("Number of layers and nodes") +
  ylab("MSE Error")  +
  scale_x_discrete(breaks = Metrics_Volume3$ModelID[c(T, F, F, F)])

ggplot(data = Metrics_Volume3, mapping = aes(x = as.character(ModelID))) + 
  geom_point(aes(y = as.numeric(RMSE.Train), colour = "RMSE.Train")) +
  geom_point(aes(y = as.numeric(RMSE.Valid), colour = "RMSE.Valid")) +
  ylim(0,1300) + 
  theme(axis.ticks = element_line(size  =  0.2, linetype  =  'blank'),
        panel.grid.major = element_line(linetype  =  'blank'), 
        panel.grid.minor = element_line(colour  =  'ivory4'),
        axis.text.x = element_text(angle = 90, size = 7)) + 
  xlab("Number of layers and nodes") +
  ylab("RMSE Error")  +
  scale_x_discrete(breaks = Metrics_Volume3$ModelID[c(T, F, F, F)])






# rm(Model_Test)
# Model_2 <- deepnet::nn.train(x = Input_Data_Training %>% as.matrix(), y = Output_Data_Training %>% as.matrix(),
#                              hidden = c(20,20,200),
#                              activationfun = "sigm",
#                              numepochs = 500, batchsize = 40, output = "sigm")
# 
# Model_Test <- nn.test(nn = Model_2, x = Input_Data_Testing, y = Output_Data_Training)

#   
# model <- neuralnet(formula = Volume.3 + Speed.3 ~ Volume.1 + Volume.2 + Speed.1 + Speed.2,
#                   data = Training_Data, hidden = c(100,100,100), rep = 4, lifesign = 'full',
#                   act.fct = 'logistic',  linear.output = TRUE, 
#                   learningrate = 0.0001, algorithm = 'rprop+', err.fct = "sse")
# 
# 
# model_2 <- nnet(formula = Volume.3 + Speed.3 ~ Volume.1 + Volume.2 + Speed.1 + Speed.2, data = Training_Data,
#                 size = 10, linout = TRUE, decay = 0.2)
# 
# 
# model <- neuralnet(formula = Volume.3 ~ Volume.1 + Volume.2 + Speed.1 + Speed.2,
#                    data = Training_Data, hidden = c(10,10,10), rep = 50, lifesign = 'minimal',
#                    act.fct = 'logistic', err.fct = 'sse', linear.output = TRUE)


