Input1 <- cbind(5, 0.5, 2)
Desired_Output <- 1
learn_rate <- 5
Hidden_Nodes <- 2
Number_Iterations <- 2
Weights_Input <- matrix(0.1, nrow = ncol(Input1), ncol = Hidden_Nodes)
Weights_Output <- matrix(0.5, Hidden_Nodes, 1)



train <- function(x, y, hidden, learn_rate, iterations, Weights1, Weights2) {
  d <- ncol(x)
  w1 <- matrix(Weights1, d, hidden)
  w2 <- matrix(Weights2, nrow = hidden , 1)
  for (i in 1:iterations) {
    ff <- feed_forward(x, w1, w2)
    bp <- feed_backward(x, y,
                        y_hat = ff$output,
                        w1, w2,
                        h = ff$h,
                        learn_rate = learn_rate)
    w1 <- bp$w1; w2 <- bp$w2
  }
  list(output = ff$output, w1 = w1, w2 = w2)
  
}


feed_forward <- function(x = Input1, w1, w2){
  z1 <- cbind(x) %*% w1
  h <- sigmoid(z1)
  z2 <- cbind(h) %*% w2
  list(output = sigmoid(z2), h = h)
}

sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

feed_backward <- function(x, y = Desired_Output, y_hat, w1, w2, h, learn_rate) {
  
  dw2 <- (y_hat - y) * y_hat * (1 - y_hat) * as.vector(h)
  
  dh <- (as.double(y_hat) - y) * as.double(y_hat) * (1 - as.double(y_hat)) %*% w2[1]
  dw1 <- t(x) %*% (h *  (1 - h) * c(dh))
  
  
  
  w1 <- w1 - learn_rate * dw1
  w2 <- w2 - learn_rate * dw2
  
  list(w1 = w1, w2 = w2)
}

nnet <- train(x = Input1, y = Desired_Output, hidden = Hidden_Nodes, learn_rate = learn_rate, iterations = Number_Iterations, 
              Weights1 = Weights_Input, Weights2 = Weights_Output)

