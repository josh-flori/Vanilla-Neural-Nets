# Vanilla-Neural-Nets

The purpose of this repository is to demonstrate a hand built network and explore basic questions relating to architecture and performance. Data for the networks are pictures of a slinky and an erasor and I trained the networks to recognize the difference between the two. This repository contains three basic networks with several variants, or batches. Training data is available here:
https://drive.google.com/file/d/1dK0JhkUuWg-oTfPda8DOg_UDiLsMcVXb/view?usp=sharing


# Network 1

## Batch 1
Batch 1 tested a learning rate of .05 at 100,000 iterations on 9 different networks with the following number of hidden layers and units per layer:

[5,5,5,5,5], [5,5,5,5], [5,5,5], [5,5], [10,5], [50,5], [50,10], [5,50], [50,50]

#### Take aways:

1) Performance for more than 2 hidden layers is just bad. They take a long time to optimize or (seemingly) never optimize at all as with the first 3 models

    a) We see large upward spikes for models 7:9 before they leave the plateau. I don't know what's different about those than models 4:6 since they have similar numbers of layers/nodes
    
    b) It looks like anything more than 4 layers like ([n_x,n_h_1_2,n_h_2,n_y]) either doesn't optimize or takes forever to optimize
    
    c) But even 3 layers with the smallest number of total nodes out of all models still takes MORE time to optimize than 4 layer functions with more nodes!! Why is that? The only improvement is that it does not experience a huge spike before optimizing
    
    d) Although a different run with different initializations moved it a bit, I'm surprised the 9th model optimizes before the 8th considering it is so much smaller. ..
    
      i) layer_dims_8 = [2100, 5, 50, 1]
          
      ii)layer_dims_9 = [2100, 50, 50, 1]
          
The plots are unlabeled. Training cost is on the vertical axis. Iterations are on X. Individual models are identified by color as highlighted in the top right legend.

![alt_text](https://imgur.com/hsTR7YV.png)


## Batch 2
Batch 1 tested a higher learning rate of .1 with everything else being the same as Batch 1

#### Take aways:

1) When you increase the learning rate it speeds up learning. My rate is .1 which seems very large compared to stuff I've seen online so I'm not sure what the difference is in terms of why I can use it without seeming consequence but it's a general bad idea.

So... im curious with batch gradient descent how... cost can be flat for thousands of iterations then suddenly get not flat. how does it move if it's completely flat? but it looks like the more layers the longer it takes to descend, or rather, the longer the plateus are... but what about same number of layers and larger number of neurons?
