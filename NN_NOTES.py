#A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b*c*d, a) is to use:
X_flatten = X.reshape(X.shape[0], -1).T
# so a matrix of 209 images, 64px in size: (209, 64, 64, 3) would become shaped (12288, 209)

# GENERAL THINGS TO REMEMBER AND COME BACK TO......
np.dot() = # impliments matrix multiplication where two matrices [a,b],[b,c] must share same dimension of y. the resulting shape will always be of dimensions (a,c)
* # impliments elementwise multiplication

'''' For a 3 layer network...

     ############################
     ######## FORWARD PROP ######
     ############################

     W[l].shape = (n[l],n[l-1])     <-- [l] = the current layer in the network and n[l] is the node count 
                                              in that layer

     b[l].shape = (n[l],1)          <-- remember that even though one of the dimensions is 1 and is being added 
                                        to a matrix of higher dimension, it is broadcast across all dimensions.

     Z[l].shape = (n[l],m)


     A[l].shape = (n[l],m)


     ----------     ----------     ----------     ----------     ----------
          np.dot(W,X) = np.dot((node_count,nx),(nx,m))  =  (node_count,m)
     ----------     ----------     ----------     ----------     ----------

     
     
     
     

     ############################
     ####### BACKWARD PROP ######
     ############################
     
     dZ2 = A2-Y
     dW2 = np.dot(dZ2,A1.T)/m
     db2 = np.sum(dZ2,axis=1,keepdims=True)/m
     ....
     dZ1 = np.dot(W[2].T,dZ2) * dgz[1] (or in other words, the derivative of the activation function of z1, or 
                                        still in more words, the derivative of A1, which is given on page 12 of 
                                        your notes that you have been taking. its 1-a^2 for a tanh activation
                                        function)


     dW[l].shape = (n[l],n[l-1]) 

     db[l].shape = (n[l],1)      

     dZ[l].shape = (n[l],m)

     dA[l].shape = (n[l],m)



















'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                           
                                           LOGISTC SHIT
                                           
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''                                           






# ------ What you need to remember:

# Common steps for pre-processing a new dataset are:

# 1) Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
# 2) Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
# 3) "Standardize" the data (substract the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole numpy array), or just divide b 255 for pixel values



#Reminder: The general methodology to build a Neural Network is to:

# 1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
# 2. Initialize the model's parameters
# 3. Loop:
#    - Implement forward propagation
#    - Compute loss
#    - Implement backward propagation to get the gradients
#    - Update parameters (gradient descent)




# why average the deriviates of the weights?

#Summing the gradients due to individual samples you get a much smoother gradient. The larger the batch the smoother the resulting gradient used in updating the weight.

#Dividing the sum by the batch size and taking the average gradient has the effect of:

#The magnitude of the weight does not grow out of proportion. Adding L2 regularization to the weight update penalizes large weight values. This often leads to improved generalization performance. Taking the average, especially if the gradients happen to point in the same direction, keep the weights from getting too large.
#The magnitude of the gradient is independent of the batch size. This allows comparison of weights from other experiments using different batch sizes.
#Countering the effect of the batch size with the learning rate can be numerically equivalent but you end up with a learning rate that is implementation specific. It makes it difficult to communicate your results and experimental setup if people cannot relate to the scale of parameters you're using and they'll have trouble reproducing your experiment.
#Averaging enables clearer comparability and keeping gradient magnitudes independent of batch size. Choosing a batch size is sometimes constrained by the computational resources you have and you want to mitigate the effect of this when evaluating your model.





Z = np.dot(w.T,x)+b

def sigmoid(z):
    s = 1/(1+np.exp(-z))    
    return s
    # to compute it without Z given, note the order of parentheses here.... 1/(1+np.exp(-(np.dot(w.T,X)+b))) # <--- validated to be accurate, so use this.

# dim = number of pixels i believe
def initialize_with_zeros(dim):
    w = np.zeros((dim,1)) 
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b
    
    
    
    
    


# so the setup is like this...
# w = np.array([[1.],[2.]])
# b = 2.
# X = np.array([[1.,2.,-1.],
#               [3.,4.,-3.2]])
# Y = np.array([[1,0,1]])

# in other words, X has 3 columns. (1,3) are the two features for training example(1). 2,4 are the two features for training example(2) etc..

# X.shape = (2,3)
# Y.shape = (1,3)

def propagate(w, b, X, Y):

    m = X.shape[1]
    
    ##############################       FORWARD PROPAGATION (FROM X TO COST) ########################
    A = 1/(1+np.exp(-(np.dot(w.T,X)+b)))  # compute activation, or in other words, run the parameters with their weights through a linear model that gets squishified through a sigmoid function which if >.5 "activates" the neuron or causes it to fire, or read the value as true, or however you want to interpret it
    cost = -((Y*np.log(A))+(1-Y)*np.log(1-A))   # compute cost. again... since Y,A are vectors (or in other cases matrices), and np.log is capable of vectorization, what happens here is the loss function is computed across all example at once and you are returned with a vector of 3 losses, which will be averaged in the line below.
    cost = cost.sum()/m
    
    
    ########################      BACKWARD PROPAGATION (TO FIND GRADIENT with respect to weights and bias) ########################
    deriviative_with_respect_to_weights = np.dot(X,  (A - Y).T)/m    # remember that np.dot(X,(A-Y).T) is multiplying matrices of shapes (2,3) & (3,1) together.
    deriviative_with_respect_to_bias = ((A - Y).sum())/m
    

    assert(deriviative_with_respect_to_weights.shape == w.shape)
    assert(deriviative_with_respect_to_bias.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": deriviative_with_respect_to_weights,
             "db": deriviative_with_respect_to_bias}
    
    return grads, cost 







def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads, cost = propagate(w, b, X, Y)

        
        
        # print(w,dw,w-dw,learning_rate*dw,w-(learning_rate*dw)) # <-- you can kind of use this to just see what's happening. and if you change the values of w, you can watch to see how it still descends toward the same answer
        # Retrieve derivatives from variable "grads"
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        w = w-(learning_rate*dw)
        b = b-(learning_rate*db)
        
        # Record the costs
        if i % 100 == 0: # % returns the remainder after a division. so only when i=100 will i%100 be 0 since 100/100 leaves no remainder after division
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0: # i still have no fucking clue how print_cost turns to true.
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
    
    
    
    
    
    
def predict(w, b, X):

    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m)) # we are just initalizing here
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = 1/(1+np.exp(-(np.dot(w.T,X)+b)))
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0][i]>.5:
            Y_prediction[0][i]=1
        else:
            Y_prediction[0][i]=0

    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction    
    
    
    
    
# What to remember: You've implemented several functions that:

# * Initialize (w,b)
# * Optimize the loss iteratively to learn parameters (w,b):
#      * computing the cost and its gradient
#      * updating the parameters using gradient descent
# * Use the learned (w,b) to predict the labels for a given set of examples    
    
    
# so just to type this out... the propogate function is like a yoyo, it goes out to the loss function ("cost" function for all of them) and comes back with information about how bad those params were. it executes a single time. then what the optimize function does is it takes that propogate function and runs it many many many times, gradually updating the params according to what information the yoyo, or messenger pigeon came back with. so the propogate function just sits happily inside the optimize function. then what happens from there is the predict function takes those results and does a 1-time evaluation of how good the optimize function and the propogate function together performed. heyy!

# the optimize function returns params, grads, costs
    
    
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False): 
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(train_set_x.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_train)
    Y_prediction_train = predict(w, b, X_test)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                           
                                     2 LAYER NETWORK (1 HIDDEN LAYER)
                                           
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''                                               
    
    
    
    
    
    
https://github.com/fanghao6666/neural-networks-and-deep-learning/blob/master/py/Building%20your%20Deep%20Neural%20Network%20Step%20by%20Step%20v3.py    
    
    
    
    
    
    
    
    
    
# MY FIRST NEURAL NETWORK...........

# 4.1 Defining the neural network structure
# Exercise: Define three variables (output literally just numbers):

# - n_x: the size of the input layer X is shaped where rows[0] are features and columns [1] and training examples. this means that X[0] will be the size of the input layer since that's the number of features, or parameters
# - n_h: the size of the hidden layer (set this to 4) # this would be anything you want but i think it needs to be >= X[0]?? <-- literally not true, hm, lol
# - n_y: the size of the output layer # in the example this was set to 2 but I don't understand why. should be 1 for binary classification?
    



# 4.2 - Initialize the model's parameters
# Exercise: Implement the function initialize_parameters().

# Make sure your parameters' sizes are right. Refer to the neural network figure above if needed.
# You will initialize the weights matrices with random values.
# Use: np.random.randn(a,b) * 0.01 to randomly initialize a matrix of shape (a,b).
# You will initialize the bias vectors as zeros.
# Use: np.zeros((a,b)) to initialize a matrix of shape (a,b) with zeros.



# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = np.random.randn(a,b) * 0.01
    b1 = None
    W2 = None
    b2 = None
    ### END CODE HERE ###
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h)) # <-- for a 3 layer network
    assert (b2.shape == (n_y, 1)) # <-- for a 3 layer network
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters





# 4.3 - The Loop
# Question: Implement forward_propagation().

# The steps you have to implement are:
# Retrieve each parameter from the dictionary "parameters" (which is the output of initialize_parameters()) by using parameters[".."].
# Implement Forward Propagation. Compute  Z[1],A[1],Z[2]Z[1],A[1],Z[2]  and  A[2]A[2]  (the vector of all your predictions on all the examples in the training set).
# Values needed in the backpropagation are stored in "cache". The cache will be given as an input to the backpropagation function.

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    ### START CODE HERE ### (≈ 4 lines of code)
    Z1 = np.dot(W1,X)+b1 # <--- note that we do not need to transpose?
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    ### END CODE HERE ###
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache



    
# 4.4 Exercise: Implement compute_cost() to compute the value of the cost  JJ .

# There are many ways to implement the cross-entropy loss. To help you, we give you how we would have implemented  −∑i=0my(i)log(a[2](i))−∑i=0my(i)log⁡(a[2](i)) :
# logprobs = np.multiply(np.log(A2),Y)
# cost = - np.sum(logprobs)                # no need to use a for loop!
# (you can use either np.multiply() and then np.sum() or directly np.dot()).    


# GRADED FUNCTION: compute_cost

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    ### START CODE HERE ### (≈ 2 lines of code)
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1 - A2),1 - Y)
    cost = - np.sum(logprobs) * (1 / m) 
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost
    
    
    
# 4.5 Question: Implement the function backward_propagation().

# Instructions: Backpropagation is usually the hardest (most mathematical) part in deep learning. To help you, here again is the slide from the lecture on backpropagation. You'll want to use the six equations on the right of this slide, since you are building a vectorized implementation.    

# Tips:
# To compute dZ1 you'll need to compute  g[1]′(Z[1])g[1]′(Z[1]) . Since  g[1](.)g[1](.)  is the tanh activation function, if  a=g[1](z)a=g[1](z)  then  g[1]′(z)=1−a2g[1]′(z)=1−a2 . So you can compute  g[1]′(Z[1])g[1]′(Z[1])  using (1 - np.power(A1, 2)).    
    

        
# GRADED FUNCTION: backward_propagation

# GRADED FUNCTION: backward_propagation

def backward_propagation(parameters, Zs_and_As, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters W
    Zs_and_As -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    ### START CODE HERE ### (≈ 2 lines of code)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    ### END CODE HERE ###
        
    # Retrieve also A1 and A2 from dictionary "Zs_and_As".
    ### START CODE HERE ### (≈ 2 lines of code)
    A1 = Zs_and_As["A1"]
    A2 = Zs_and_As["A2"]
    ### END CODE HERE ###
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    ### START CODE HERE ### (≈ 6 lines of code, corresponding to 6 equations on slide above)
    dZ2 = A2-Y                  # shape = (1,m)
    dW2 = np.dot(dZ2,A1.T)/m    # shape = (1,node_count) i think....
    db2 = np.sum(dZ2,axis=1,keepdims=True)/m  #
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m
    ### END CODE HERE ###
    
    gradients = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return gradients
    
    
    
    
    
# GRADED FUNCTION: linear_backward

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    ### START CODE HERE ### (≈ 3 lines of code)
    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)
    ### END CODE HERE ###
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db    
    
    
    
    
    
    
    
# GRADED FUNCTION: update_parameters

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###
    
    # Retrieve each gradient from the dictionary "grads"
    ### START CODE HERE ### (≈ 4 lines of code)
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    ## END CODE HERE ###
    
    # Update rule for each parameter
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = W1-(learning_rate*dW1)
    b1 = b1-(learning_rate*db1)
    W2 = W2-(learning_rate*dW2)
    b2 = b2-(learning_rate*db2)
    ### END CODE HERE ###
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters    
    
    
    
    
    
    
    
    
    
    
# GRADED FUNCTION: nn_model

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    ### START CODE HERE ### (≈ 5 lines of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):
         
        ### START CODE HERE ### (≈ 4 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate = 1.2)
        
        ### END CODE HERE ###
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters    
    
    
    
    
    
    
# GRADED FUNCTION: predict

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    ### START CODE HERE ### (≈ 2 lines of code)
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    ### END CODE HERE ###
    
    return predictions    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                           
                                            DEEP NETWORK
                                           
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''    



''' To build your neural network, you will be implementing several "helper functions". These helper functions will be used in the next assignment to build a two-layer neural network and an L-layer neural network. Each small helper function you will implement will have detailed instructions that will walk you through the necessary steps. Here is an outline of this assignment, you will:

Initialize the parameters for a two-layer network and for an  LL -layer neural network.
Implement the forward propagation module (shown in purple in the figure below).
Complete the LINEAR part of a layer's forward propagation step (resulting in  Z[l]Z[l] ).
We give you the ACTIVATION function (relu/sigmoid).
Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer  LL ). This gives you a new L_model_forward function.
Compute the loss.
Implement the backward propagation module (denoted in red in the figure below).
Complete the LINEAR part of a layer's backward propagation step.
We give you the gradient of the ACTIVATE function (relu_backward/sigmoid_backward)
Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
Finally update the parameters. '''



import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v4 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

np.random.seed(1)







# L-layer Neural Network
# The initialization for a deeper L-layer neural network is more complicated because there are many more weight matrices and bias vectors. When completing the  initialize_parameters_deep, you should make sure that your dimensions match between each layer. Recall that  n[l]n[l]  is the number of units in layer  ll . Thus for example if the size of our input  XX  is  (12288,209)(12288,209)  (with  m=209m=209  examples) then:

# GRADED FUNCTION: initialize_parameters_deep

# the following function will work when the input values like this:
# GRADED FUNCTION: initialize_parameters_deep



def initialize_parameters_deep(layer_dims):
    # so the way this function works is that layer_dims is a list like [5,2,6] where each element 
    # corresponds to a layer of the network, and it works sequentially so that the first element
    # (in this example, 5) is the input parameter layer X, then 2 is W1, then 3 is b1, then each element
    # pair after that is w2,b2, wn,bn all the way forward. pretty cool. 
    
    # Returns:
    # parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
    #                Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
    #                bl -- bias vector of shape (layer_dims[l], 1)

    
    np.random.seed(3)  # not needed when doing on your own obviously
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L): # ok so remember python indices DO start at 0, but we intentionally start at 1 because
                          # the first element of the input list is the layer X, which we do not need to initialize
                          # parameters for so we skip it, but we include it in the input list because we base
                          # the dimensions of W1 and consequently everything else on it.

        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        ### END CODE HERE ###

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

#^^^^^^^^^^^^^^
# OK MOTHERFUCKING GOT IT!!!!!!!!!! 







# FORWARD PROPOGATION MODULE....... YOU WILL COMPUTE THESE THREE FUNCTIONS
# LINEAR (linear_forward)                                                                   -- COMPUTER Z, RETURNS [z,cache(A, W, b)]
# LINEAR -> ACTIVATION (linear_activation_forward) where ACTIVATION will be either ReLU or Sigmoid.  -- COMPUTES A from the z provided by linear_forward above, RETURNS [A, activation_cache(z only?)] so simply but so confusing name
# [LINEAR -> RELU]  ××  (L-1) -> LINEAR -> SIGMOID (whole model)             -- ok bitches so this just simply takes the above function (which on it's own takes the function above it) and impliments for l:L, pretty fucking simple







# GRADED FUNCTION: linear_forward

def linear_forward(A, W, b): # COMPUTES Z GIVEN W,X(A),b, RETURNS (Z,CACHE).... cache = (A, W, b)
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    Z = np.dot(W,A)+b
    ### END CODE HERE ###
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache
    
#^^^^^^^^^^^^^^
# OK MOTHERFUCKING GOT IT!!!!!!!!!!     
    
    
    
# GRADED FUNCTION: linear_activation_forward

def linear_activation_forward(A_prev, W, b, activation):  # RUNS Z THROUGH THE ACTIVATION FUNCTION, EITHER RELU OR SIGMOID BASED ON WHATEVER YOU SPECIFY. 
                                                          # RETURNS (A, cache).... cache = (A, W, b), yes this is redudant, A is returned twice
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
        ### END CODE HERE ###
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
        ### END CODE HERE ###
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache    

 #^^^^^^^^^^^^^^
 # OK MOTHERFUCKING GOT IT!!!!!!!!!!   
    
    
    
    
# GRADED FUNCTION: L_model_forward

def L_model_forward(X, parameters):    # FOR ANY GIVEN NUMBER OF LAYERS, COMPUTES  THE SAME AS linear_activation_forward BASICALLY IT JUST DOES IT FOR ALL LAYERS, SO SAME AS ABOVE, IT STILL RETURNS (AL, cache).... cache = (A, W, b) BUT DOES IT FOR ALL LAYERS, BUT ONLY AL IS RETURNED ALONE, NOT ANY OF THE OTHER A'S

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        ### START CODE HERE ### (≈ 2 lines of code)
        A, cache = linear_activation_forward(A_prev, 
                                             parameters['W' + str(l)], 
                                             parameters['b' + str(l)], 
                                             activation='relu')
        caches.append(cache)
        
        ### END CODE HERE ###
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    ### START CODE HERE ### (≈ 2 lines of code)
    AL, cache = linear_activation_forward(A,  # <--- i still don't fucking understand how A, or X is the input for the final layer. A-1 should be the input right?
                                          parameters['W' + str(L)], 
                                          parameters['b' + str(L)], 
                                          activation='sigmoid')
    caches.append(cache)
    
    ### END CODE HERE ###
    
    assert(AL.shape == (1, X.shape[1]))
            
    return AL, caches    
    
 #^^^^^^^^^^^^^^
 # OK MOTHERFUCKING GOT IT!!!!!!!!!!       
    
    
    
    
    
    
    
    
# GRADED FUNCTION: compute_cost

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 lines of code)
    cost = - np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1 - AL),1 - Y)) * (1 / m) 
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost    
    
    
    
    
    
#Now, similar to forward propagation, you are going to build the backward propagation in three steps:

#LINEAR backward                 -- COMPUTES and RETURNS dA_prev, Dw (current layer), db (current layer)
#LINEAR -> ACTIVATION backward where ACTIVATION computes the derivative of either the ReLU or sigmoid activation
#[LINEAR -> RELU]  ××  (L-1) -> LINEAR -> SIGMOID backward (whole model)        
    
    
    
# GRADED FUNCTION: linear_backward

def linear_backward(dZ, cache):   # cache = the linear output modules needed to compute the gradients of the function, a_prev,w,b... apparently z is not needed here.
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    ### START CODE HERE ### (≈ 3 lines of code)
    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)
    ### END CODE HERE ###
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db    
    
 #^^^^^^^^^^^^^^
 # ..pretty sure i got it but please don't test me on it    
    
    
# GRADED FUNCTION: linear_activation_backward

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###
        
    elif activation == "sigmoid":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###
        
        
    
    return dA_prev, dW, db    
    
    
#^^^^^^^^^^^^^^
# sure why fucking not









# GRADED FUNCTION: L_model_backward

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    ### START CODE HERE ### (1 line of code)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    ### END CODE HERE ###
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    ### START CODE HERE ### (approx. 2 lines)
    current_cache = caches[-1] # gets the last element, in this case the cache, or as he is calling it: the cache of linear_activation_forward() with "sigmoid" 
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    ### END CODE HERE ###
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads
        
        
        
        
        
        
        
# GRADED FUNCTION: update_parameters

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    ### START CODE HERE ### (≈ 3 lines of code)
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-(learning_rate*grads["dW" + str(l+1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-(learning_rate*grads["db" + str(l+1)])
    ### END CODE HERE ###
    return parameters        
    
    
    
    
    
    
    
    
    
    
    