# Vanilla-Neural-Nets

The purpose of this repository is to demonstrate a hand built network and explore basic questions relating to architecture and performance. This repository contains the following models:

### Network 1

##### Overview: 
The first network varies the number of hidden layers, units per layer, learning rate and number of iterations enough to get a good feel for different behavior and in different circumstances.

##### Take aways:

1) Performance for more than 2 hidden layers is just bad. They take a long time to optimize or (seemingly) never optimize at all as with the first 3 models

       a) We see large upward spikes for models 7:9 before they leave the plateau. I don't know what's different about those than models 4:6 since they have similar numbers of layers/nodes
    
       b) It looks like anything more than 4 layers like ([n_x,n_h_1_2,n_h_2,n_y]) either doesn't optimize or takes forever to optimize
    
       c) But even 3 layers with the smallest number of total nodes out of all models still takes MORE time to optimize than 4 layer functions with more nodes!! Why is that? The only improvement is that it does not experience a huge spike before optimizing
    
       d) Although a different run with different initializations moved it a bit, I'm surprised the 9th model optimizes before the 8th considering it is so much smaller. ..
    
          i) layer_dims_8 = [2100, 5, 50, 1]
          
          ii)layer_dims_9 = [2100, 50, 50, 1]



All data can be viewed here
https://drive.google.com/file/d/1dK0JhkUuWg-oTfPda8DOg_UDiLsMcVXb/view?usp=sharing
