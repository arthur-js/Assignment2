Run_Volume_NN <- function(Model_File, x, y, Training_Frame, Validation_Frame)
{
  Model_DL_Volume <- Model_File;
  rm(Model_File)
  n <- 0;
  for (i in 1:3) {
    if (i==1) {
      for (j in 5:10) {
        n <- n + 1;
        Model_DL_Volume[n] <- list(h2o.deeplearning(x = x, y = y,
                                                    training_frame = Training_Frame %>% as.h2o(), 
                                                    model_id = paste(i, "hidden layer with", j, "nodes", sep = " "), 
                                                    hidden = c(j), validation_frame = Validation_Frame %>% as.h2o(),
                                                    export_weights_and_biases = TRUE));
        
      }
    }
    
    if (i==2) {
      for (j in 5:10) {
        for (k in 5:10) {
          n <- n + 1;
          
          Model_DL_Volume[n] <- list(h2o.deeplearning(x = x, y = y,
                                                      training_frame = Training_Frame %>% as.h2o(), 
                                                      model_id = paste(i, "hidden layers with", j, "and", 
                                                                       k, "nodes", sep = " "), 
                                                      hidden = c(j,k), validation_frame = Validation_Frame %>% as.h2o(),
                                                      export_weights_and_biases = TRUE));
          
        }
      }
    }
    if (i==3) { 
      for (j in 5:10) {
        for (k in 5:10) {
          for (l in 5:10) {
            n <- n + 1;
            Model_DL_Volume[n] <- list(h2o.deeplearning(x = x, y = y,
                                                        training_frame = Training_Frame %>% as.h2o(), 
                                                        model_id = paste(i, "hidden layers with", j, ",", k, 
                                                                         "and", l, "nodes", sep = " "), 
                                                        hidden = c(j, k, l), validation_frame = Validation_Frame %>% 
                                                          as.h2o(),
                                                        export_weights_and_biases = TRUE));
            
            
            
          }
        }
      }
    }
  }
  return(Model_DL_Volume)
}

