""""# A well chosen initialization can: 
#    * Speed up the convergence of gradient descent
#    * Increase the odds of gradient descent converging to a lower training (and generalization) error 


Different initializations lead to different results
Random initialization is used to break symmetry and make sure different hidden units can learn different things
Don't intialize to values that are too large
He initialization works well for networks with ReLU activations.

!!! INITIALIZING TO ZEROS !!!
In general, initializing all the weights to zero results in the network failing to break symmetry. 
This means that every neuron in each layer will learn the same thing, and you might as well be training a n
eural network with  n[l]=1n[l]=1  for every layer, and the network is no more powerful than a linear classifier 
such as logistic regression.


What you should remember:

The weights  W[l]W[l]  should be initialized randomly to break symmetry.
It is however okay to initialize the biases  b[l]b[l]  to zeros. Symmetry is still broken so long as  W[l]W[l]  is initialized randomly.



!!! INITIALIZING TO LARGE RANDOM NUMBERS !!! ... like np.random.randn(layers_dims[l],layers_dims[l-1]) * 10 

The cost starts very high. This is because with large random-valued weights, the last activation (sigmoid) outputs results that are very close to 0 or 1 for some examples, and when it gets that example wrong it incurs a very high loss for that example. Indeed, when  log(a[3])=log(0)log⁡(a[3])=log⁡(0) , the loss goes to infinity.
Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm.
If you train this network longer you will see better results, but initializing with overly large random numbers slows down the optimization.
In summary:

Initializing weights to very large random values does not work well.
Hopefully intializing with small random values does better. The important question is: how small should be these random values be? Lets find out in the next part!
"""



def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        ### END CODE HERE ###
        
    return parameters
    
    
    
    

def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.
    
    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
    
    ### START CODE HERE ### (approx. 1 line)
    L2_regularization_cost = (lambd/(2*m))*np.sum((np.sum(np.square(W1)),np.sum(np.square(W2)),np.sum(np.square(W3))))
    
    
    
    ### END CODER HERE ###
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost    
    
    
    
# GRADED FUNCTION: backward_propagation_with_regularization


"""The value of  λ  is a hyperparameter that you can tune using a dev set.
L2 regularization makes your decision boundary smoother. If  λ  is too large, it is also possible to "oversmooth", resulting in a model with high bias.
What is L2-regularization actually doing?:

L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.

What you should remember -- the implications of L2-regularization on:

The cost computation:
A regularization term is added to the cost
The backpropagation function:
There are extra terms in the gradients with respect to weight matrices
Weights end up smaller ("weight decay"):
Weights are pushed to smaller values."""

def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    
    ### START CODE HERE ### (approx. 1 line)
    dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd/m)*W3
    ### END CODE HERE ###
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd/m)*W2
    ### END CODE HERE ###
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW1 = 1./m * np.dot(dZ1, X.T) + (lambd/m)*W1
    ### END CODE HERE ###
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients    
    
    
    
    
    
def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """
    
    np.random.seed(1)
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    ### START CODE HERE ### (approx. 4 lines)         # Steps 1-4 below correspond to the Steps 1-4 described above. 
    D1 = np.random.randn(A1.shape[0],A1.shape[1])     # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    D1 = (D1 > keep_prob)                             # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = A1 * D1                                      # Step 3: shut down some neurons of A1
    A1 = A1 / keep_prob                               # Step 4: scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    ### START CODE HERE ### (approx. 4 lines)
    D2 = np.random.randn(A2.shape[0],A2.shape[1])     # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    D2 = (D2 > keep_prob)                             # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A2 = A2 * D2                                      # Step 3: shut down some neurons of A2
    A2 = A2 / keep_prob                               # Step 4: scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache    
    
    
    
    
    
# GRADED FUNCTION: backward_propagation_with_dropout

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Implements the backward propagation of our baseline model to which we added dropout.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation_with_dropout()
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)
    ### START CODE HERE ### (≈ 2 lines of code)
    dA2 = dA2*D2              # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 = dA2/keep_prob              # Step 2: Scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    ### START CODE HERE ### (≈ 2 lines of code)
    dA1 = dA1*D1              # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 = dA1/keep_prob              # Step 2: Scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients    
    
    
    
    
    
"""As in 1) and 2), you want to compare "gradapprox" to the gradient computed by backpropagation. The formula is still:

∂J∂θ=limε→0J(θ+ε)−J(θ−ε)2ε(1)
(1)∂J∂θ=limε→0J(θ+ε)−J(θ−ε)2ε
 
However,  θθ  is not a scalar anymore. It is a dictionary called "parameters". We implemented a function "dictionary_to_vector()" for you. It converts the "parameters" dictionary into a vector called "values", obtained by reshaping all parameters (W1, b1, W2, b2, W3, b3) into vectors and concatenating them.

The inverse function is "vector_to_dictionary" which outputs back the "parameters" dictionary.    """
    
    
    
    
    
# GRADED FUNCTION: gradient_check_n
# holy hell i actually understand this. like for real, now granted he is the best teacher ever, but i can tell im getting better at learning as well.
def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    print(len(parameters_values))
    print(len(grad))
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        ### START CODE HERE ### (approx. 3 lines)
        thetaplus = np.copy(parameters_values)                                      # Step 1
        thetaplus[i] = thetaplus[i]+epsilon                               # Step 2
        
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))                                   # Step 3
        ### END CODE HERE ###
        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        ### START CODE HERE ### (approx. 3 lines)
        thetaminus = np.copy(parameters_values)                                     # Step 1
        thetaminus[i] = thetaminus[i]-epsilon                               # Step 2        
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))                                  # Step 3
        ### END CODE HERE ###
        
        # Compute gradapprox[i]
        ### START CODE HERE ### (approx. 1 line)
        gradapprox[i] = (J_plus[i]-J_minus[i])/(2*epsilon)
        ### END CODE HERE ###
    
    # Compare gradapprox to backward propagation gradients by computing difference.
    ### START CODE HERE ### (approx. 1 line)
    numerator = np.linalg.norm(grad-gradapprox)                                           # Step 1'
    denominator = np.linalg.norm(grad)+np.linalg.norm(gradapprox)                                         # Step 2'
    difference = numerator/denominator                                          # Step 3'
    ### END CODE HERE ###

    if difference > 2e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference    
    
    
    
    
    
    
#### OPTIMIZATION..........    
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........    
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........
#### OPTIMIZATION..........


"""" A variant of this is Stochastic Gradient Descent (SGD), which is equivalent to mini-batch gradient descent 
where each mini-batch has just 1 example. The update rule that you have just implemented does not change. What 
changes is that you would be computing gradients on just one training example at a time, rather than on the whole 
training set. The code examples below illustrate the difference between stochastic gradient descent and (batch) gradient descent. """



"""(Batch) Gradient Descent:"""
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    # Forward propagation
    a, caches = forward_propagation(X, parameters)
    # Compute cost.
    cost = compute_cost(a, Y)
    # Backward propagation.
    grads = backward_propagation(a, caches, parameters)
    # Update parameters.
    parameters = update_parameters(parameters, grads)

"""Stochastic Gradient Descent:"""
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    for j in range(0, m):
        # Forward propagation
        a, caches = forward_propagation(X[:,j], parameters)
        # Compute cost
        cost = compute_cost(a, Y[:,j])
        # Backward propagation
        grads = backward_propagation(a, caches, parameters)
        # Update parameters.
        parameters = update_parameters(parameters, grads)

#- With a well-turned mini-batch size, usually it outperforms either gradient descent or stochastic gradient descent (particularly when the training set is large).
""""2 - Mini-Batch Gradient descent
Let's learn how to build mini-batches from the training set (X, Y).

There are two steps:

Shuffle: Create a shuffled version of the training set (X, Y) as shown below. 
Each column of X and Y represents a training example. Note that the random shuffling 
is done synchronously between X and Y. Such that after the shuffling the  ithith  column 
of X is the example corresponding to the  ithith  label in Y. The shuffling step 
ensures that examples will be split randomly into different mini-batches.

Partition: Partition the shuffled (X, Y) into mini-batches of size mini_batch_size 
(here 64). Note that the number of training examples is not always divisible by 
mini_batch_size. The last mini batch might be smaller, but you don't need to worry 
about this. When the final mini-batch is smaller than the full mini_batch_size, 
it will look like this:

"""



# GRADED FUNCTION: random_mini_batches
# hellLLLlllLLLllllLLllll ya i understand this shit (actually its not hard... :( .... )
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, (mini_batch_size * k):mini_batch_size * (k + 1)-1]
        mini_batch_Y = shuffled_Y[:, (mini_batch_size * k):mini_batch_size * (k + 1)-1]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:,m-(m-mini_batch_size*(math.floor(m/mini_batch_size))):m]
        mini_batch_Y = shuffled_Y[:,m-(m-mini_batch_size*(math.floor(m/mini_batch_size))):m]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches



"""3 - Momentum
Because mini-batch gradient descent makes a parameter update after seeing just a subset of 
    examples, the direction of the update has some variance, and so the path taken by mini-batch 
    gradient descent will "oscillate" toward convergence. Using momentum can reduce these oscillations.

Momentum takes into account the past gradients to smooth out the update. We will store the 
    'direction' of the previous gradients in the variable  vv . Formally, this will be the 
    exponentially weighted average of the gradient on previous steps. You can also think of  
    vv  as the "velocity" of a ball rolling downhill, building up speed (and momentum) according 
    to the direction of the gradient/slope of the hill.  
    
    
    
Exercise: Initialize the velocity. The velocity,  vv , is a python dictionary that needs to be 
    initialized with arrays of zeros. Its keys are the same as those in the grads dictionary, that is: for  l=1,...,Ll=1,...,L :

v["dW" + str(l+1)] = ... #(numpy array of zeros with the same shape as parameters["W" + str(l+1)])
v["db" + str(l+1)] = ... #(numpy array of zeros with the same shape as parameters["b" + str(l+1)])
Note that the iterator l starts at 0 in the for loop while the first parameters are v["dW1"] and 
    v["db1"] (that's a "one" on the superscript). This is why we are shifting l to l+1 in the for loop.    
    
    
"""
    
    
    
# GRADED FUNCTION: initialize_velocity

def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(L):
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape[0],parameters["W" + str(l+1)].shape[1]))
        v["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape[0],1))
        ### END CODE HERE ###
        
    return v






# GRADED FUNCTION: update_parameters_with_momentum

"""Note that:

The velocity is initialized with zeros. So the algorithm will take a few iterations to "build up" velocity a
nd start to take bigger steps. If  β=0β=0 , then this just becomes standard gradient descent without momentum.
How do you choose  ββ ?

The larger the momentum  ββ  is, the smoother the update because the more we take the past gradients into account. 
But if  ββ  is too big, it could also smooth out the updates too much.
Common values for  ββ  range from 0.8 to 0.999. If you don't feel inclined to tune this,  β=0.9β=0.9  is often a 
reasonable default.
Tuning the optimal  ββ  for your model might need trying several values to see what works best in term of reducing 
the value of the cost function  JJ .

"""
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2 # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(L):
        
        ### START CODE HERE ### (approx. 4 lines)
        # compute velocities
        v["dW" + str(l+1)] = beta*v["dW" + str(l+1)]+(1-beta)*grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta*v["db" + str(l+1)]+(1-beta)*grads["db" + str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v["db" + str(l+1)]
        ### END CODE HERE ###
        
    return parameters, v


"""
4 - Adam
Adam is one of the most effective optimization algorithms for training neural networks. It combines 
ideas from RMSProp (described in lecture) and Momentum.

How does Adam work?

It calculates an exponentially weighted average of past gradients, and stores it in variables  
    vv  (before bias correction) and  vcorrectedvcorrected  (with bias correction).
It calculates an exponentially weighted average of the squares of the past gradients, and stores 
    it in variables  ss  (before bias correction) and  scorrectedscorrected  (with bias correction).
It updates parameters in a direction based on combining information from "1" and "2".
    
    
where:

t counts the number of steps taken of Adam
L is the number of layers
β1β1  and  β2β2  are hyperparameters that control the two exponentially weighted averages.
αα  is the learning rate
εε  is a very small number to avoid dividing by zero
As usual, we will store all parameters in the parameters dictionary    
"""

# GRADED FUNCTION: initialize_adam

def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
    ### START CODE HERE ### (approx. 4 lines)
        v["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape[0],parameters["W" + str(l+1)].shape[1]))
        v["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape[0],parameters["b" + str(l+1)].shape[1]))
        s["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape[0],parameters["W" + str(l+1)].shape[1]))
        s["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape[0],parameters["b" + str(l+1)].shape[1]))
    ### END CODE HERE ###
    
    return v, s    
    
    
    
    
    
# GRADED FUNCTION: update_parameters_with_adam

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)]+(1-beta1)*grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1*v["db" + str(l+1)]+(1-beta1)*grads["db" + str(l+1)]
        ### END CODE HERE ###

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-np.power(beta1,t))
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-np.power(beta1,t))
        ### END CODE HERE ###

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        ### START CODE HERE ### (approx. 2 lines)
        s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)]+(1-beta2)*np.power(grads["dW" + str(l+1)],2)
        s["db" + str(l+1)] = beta2*s["db" + str(l+1)]+(1-beta2)*np.power(grads["db" + str(l+1)],2)
        ### END CODE HERE ###

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-np.power(beta2,t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1-np.power(beta2,t))
        ### END CODE HERE ###

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        ### START CODE HERE ### (approx. 2 lines)
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*(v_corrected["dW" + str(l+1)]/np.sqrt(s_corrected["dW" + str(l+1)]+epsilon))
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*(v_corrected["db" + str(l+1)]/np.sqrt(s_corrected["db" + str(l+1)]+epsilon))
        ### END CODE HERE ###

    return parameters, v, s    
    
    
    
    
    
    
    
    
    
    
    
    
    
# OK NOW ACTUALLY USING TENSORFLOW    
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW    
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW    
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW    
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW    
# OK NOW ACTUALLY USING TENSORFLOW    
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW    
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW    
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW    
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW
# OK NOW ACTUALLY USING TENSORFLOW


"""Writing and running programs in TensorFlow has the following steps:

Create Tensors (variables) that are not yet executed/evaluated.
Write operations between those Tensors.
Initialize your Tensors.
Create a Session.
Run the Session. This will run the operations you'd written above.

Therefore, when we created a variable for the loss, we simply defined the loss as a 
function of other quantities, but did not evaluate its value. To evaluate it, we had 
to run init=tf.global_variables_initializer(). That initialized the loss variable, and 
in the last line we were finally able to evaluate the value of loss and print its value.



Only after running tf.global_variables_initializer() in a session will your variables 
hold the values you told them to hold when you declare them (tf.Variable(tf.zeros(...)), 
tf.Variable(tf.random_normal(...)),...).


... " a tensor is a map. and it's a map that takes elements from a cartesian product of 
vector spaces and returns a real number. tensors come in many forms because there are many 
ways of making these cartesian products.
"""




y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')                    # Define y. Set to 39

loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                 # the loss variable will be initialized and ready to be computed
with tf.Session() as session:                    # Create a session and print the output
    session.run(init)                            # Initializes the variables
    print(session.run(loss))                     # Prints the loss
    
    
"""Now let us look at an easy example. Run the cell below:    """    
####################    
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)
print(c)
####################

"""RESULT:"""
Tensor("Mul:0", shape=(), dtype=int32)

"""As expected, you will not see 20! You got a tensor saying that the result is 
a tensor that does not have the shape attribute, and is of type "int32". All you 
did was put in the 'computation graph', but you have not run this computation yet. 
In order to actually multiply the two numbers, you will have to create a session and run it.    """


sess = tf.Session()
print(sess.run(c))
20

"""Great! To summarize, remember to initialize your variables, create a session 
and run the operations inside the session.

Next, you'll also have to know about placeholders. A placeholder is an object 
whose value you can specify only later. To specify values for a placeholder, 
you can pass in values by using a "feed dictionary" (feed_dict variable). 
Below, we created a placeholder for x. This allows us to pass in a number later 
when we run the session."""



# Change the value of x in the feed_dict

x = tf.placeholder(tf.int64, name = 'x')
print(sess.run(2 * x, feed_dict = {x: 3}))
sess.close()

"""When you first defined x you did not have to specify a value for it. A placeholder 
is simply a variable that you will assign data to only later, when running the session. 
We say that you feed data to these placeholders when running the session.

Here's what's happening: When you specify the operations needed for a computation, 
you are telling TensorFlow how to construct a computation graph. The computation graph 
can have some placeholders whose values you will specify only later. Finally, when you 
run the session, you are telling TensorFlow to execute the computation graph. """



# GRADED FUNCTION: linear_function

def linear_function():
    """
    Implements a linear function: 
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns: 
    result -- runs the session for Y = WX + b 
    """
    
    np.random.seed(1)
    
    ### START CODE HERE ### (4 lines of code)
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name = "b")
    Y = tf.add(tf.matmul(W,X),b)
    ### END CODE HERE ### 
    
    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate
    
    ### START CODE HERE ###
    sess = tf.Session()
    result = sess.run(Y) 
       
    ### END CODE HERE ### 
    
    # close the session 
    sess.close()

    return result
    
    
    
    
    
    
"""1.2 - Computing the sigmoid
Great! You just implemented a linear function. Tensorflow offers a variety of commonly 
    used neural network functions like tf.sigmoid and tf.softmax. For this exercise lets 
    compute the sigmoid function of an input.

You will do this exercise using a placeholder variable x. When running the session, 
    you should use the feed dictionary to pass in the input z. In this exercise, you 
    will have to (i) create a placeholder x, (ii) define the operations needed to compute 
    the sigmoid using tf.sigmoid, and then (iii) run the session.

Note that there are two typical ways to create and use sessions in tensorflow:"""


"""Method 1: """
sess = tf.Session()
# Run the variables initialization (if needed), run the operations
result = sess.run(..., feed_dict = {...})
sess.close() # Close the session

"""Method 2:"""
with tf.Session() as sess: 
    # run the variables initialization (if needed), run the operations
    result = sess.run(..., feed_dict = {...})
    # This takes care of closing the session for you :)    

def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    results -- the sigmoid of z
    """
    
    ### START CODE HERE ### ( approx. 4 lines of code)
    # Create a placeholder for x. Name it 'x'.
    x = tf.placeholder(tf.float32, name="x")

    # compute sigmoid(x)
    sigmoid = tf.sigmoid(x)

    # Create a session, and run it. Please use the method 2 explained above. 
    # You should use a feed_dict to pass z's value to x. 
    with tf.Session() as sess:
        # Run session and call the output "result"
        result = sess.run(sigmoid, feed_dict={x: z})
    
    ### END CODE HERE ###
    
    return result



def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0) 
    
    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
    in the TensorFlow documentation. So logits will feed into z, and labels into y. 
    
    Returns:
    cost -- runs the session of the cost (formula (2))
    """
    
    ### START CODE HERE ### 
    
    # Create the placeholders for "logits" (z) and "labels" (y) (approx. 2 lines)
    z = tf.placeholder(tf.float32, name="z")
    y = tf.placeholder(tf.float32, name="y")
    
    # Use the loss function (approx. 1 line)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)
    
    # Create a session (approx. 1 line). See method 1 above.
    sess = tf.Session()
    
    # Run the session (approx. 1 line).
    cost = sess.run(cost, feed_dict={z: logits, y: labels})
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    
    ### END CODE HERE ###
    
    return cost







# GRADED FUNCTION: one_hot_matrix

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    
    ### START CODE HERE ###
    
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name = "C")
    
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels,C,axis=0)
    
    # Create the session (approx. 1 line)
    sess = tf.Session()
    
    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close
    
    ### END CODE HERE ###
    
    return one_hot





# GRADED FUNCTION: ones

def ones(shape):
    """
    Creates an array of ones of dimension shape
    
    Arguments:
    shape -- shape of the array you want to create
        
    Returns: 
    ones -- array containing only ones
    """
    
    ### START CODE HERE ###
    
    # Create "ones" tensor using tf.ones(...). (approx. 1 line)
    ones = tf.ones(shape)
    
    # Create the session (approx. 1 line)
    sess = tf.Session()
    
    # Run the session to compute 'ones' (approx. 1 line)
    ones = sess.run(ones)
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close
    
    ### END CODE HERE ###
    return ones




# GRADED FUNCTION: create_placeholders
#2.1 - Create placeholders
#Your first task is to create placeholders for X and Y. This will allow you to later pass your training data in when you run your session.

#sExercise: Implement the function below to create the placeholders in tensorflow.
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(dtype=tf.float32, shape=[n_x, None], name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=[n_y, None], name="y")
    ### END CODE HERE ###
    
    return X, Y

X, Y = create_placeholders(12288, 6)

#2.2 - Initializing the parameters
#Your second task is to initialize the parameters in tensorflow.

#Exercise: Implement the function below to initialize the parameters in tensorflow. You are going use Xavier Initialization for weights and Zero Initialization for biases. The shapes are given below. As an example, to help you, for W1 and b1 you could use:



# GRADED FUNCTION: initialize_parameters

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters
    


tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
# GRADED FUNCTION: forward_propagation
"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
#It is important to note that the forward propagation stops at z3. The reason is that in tensorflow the last linear layer output is given as input to the function computing the loss. Therefore, you don't need a3!
"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                                # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                               # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                               # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###
    
    return Z3    

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))





def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels= labels))
    ### END CODE HERE ###
    
    return cost
    
    
    
tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))
    
    
    
    
"""2.5 - Backward propagation & parameter updates
This is where you become grateful to programming frameworks. All the backpropagation and the 
    parameters update is taken care of in 1 line of code. It is very easy to incorporate this l
    ine in the model.

After you compute the cost function. You will create an "optimizer" object. You have to 
    call this object along with the cost when running the tf.session. When called, it 
    will perform an optimization on the given cost with the chosen method and learning rate.

For instance, for gradient descent the optimizer would be:

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
To make the optimization you would do:

_ , c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
This computes the backpropagation by passing through the tensorflow graph in the reverse order. 
    From cost to inputs.

Note When coding, we often use _ as a "throwaway" variable to store values that we won't need 
    to use later. Here, _ takes on the evaluated value of optimizer, which we don't need (and 
    c takes the value of the cost variable).        """




def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_x, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X, parameters)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters







