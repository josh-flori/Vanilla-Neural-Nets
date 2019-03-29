# Vanilla-Neural-Nets

The purpose of this repository is to demonstrate a hand built network and explore basic questions relating to architecture and performance. Data for the networks are pictures of a slinky and an erasor and I trained the networks to recognize the difference between the two. This repository contains three basic networks with several variants, or batches. Training data is available here:
https://drive.google.com/file/d/1dK0JhkUuWg-oTfPda8DOg_UDiLsMcVXb/view?usp=sharing


# Network 1

## Batch 1
*Batch Settings: Batch 1 tested a learning rate of .05 at 100,000 iterations on 9 different networks with the following number of hidden layers and units per layer:*

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
*Batch Settings: This batch tests a higher learning rate of .1 with everything else being the same as Batch 1*

#### Take aways:

1) When you increase the learning rate it speeds up learning. My rate is .1 which seems very large compared to stuff I've seen online so I'm not sure what the difference is in terms of why I can use it without seeming consequence but it's a general bad idea.

So... im curious with batch gradient descent how... cost can be flat for thousands of iterations then suddenly get not flat. how does it move if it's completely flat? but it looks like the more layers the longer it takes to descend, or rather, the longer the plateus are... but what about same number of layers and larger number of neurons?


![alt_text](https://imgur.com/alM8OiT.png)


## Batch 3
* Batch Settings: This batch tests a .05 learning over 100,000 iterations on 6 different networks with the following number of hidden layers and units per layer, with the last having no hidden layers, just a single logistic unit:*

[5], [20], [50], [100], [1000], [] <-- single logistic unit, no hidden layers


#### Take aways:

1) Some really interesting things happening here...
    a) First of all, all of these are billions of times faster than the previous models... and this isn't even a fast learning rate or highly optimized! (I drastically shortened the x axis in the plot to account for this)
    b) So let me get this right.... when it comes to having 2 hidden layers, the MORE neurons there are, the FEWER iterations it takes to begin descending but the longer the iterations take, with 1,000 neurons taking a hella long time, many many times longer than the others
    c) But NONE of the models are NEARLY as fast at descending as just the logistic unit alone. But it has some weird noise, I wonder if it's because it's dumber
    d) Perhaps similar to the logistic unit, the model with only 5 neurons had a weird squiggle, maybe more neurons help smooth out the function, or something...

2) I would be curious if the whole "more neurons in a layer = fewer iterations to descend" is true across the board, or in what circumstances

![alt_text](https://imgur.com/Z78ISTG.png)



## Batch 4
*Batch Settings: This batch tests a higher learning rate of .1 with everything else being the same as Batch 3*


#### Take aways:

1) Convergence was only slightly faster for all but one, but VERY noisy, which is not great.

   a) The really interesting thing is that a .1 learning rate (seems) to have had a better effect for larger than smaller networks.
      It seems to have created less noise (though maybe I wasn't zoomed in far enough to see it). But it definitely sped up learning more
      (though I think it was already at some optimally fast level for the smaller networks).
      
2) Overall, not preferable to the .05 learning rate on the same networks, but still good to see this visually.


![alt_text](https://imgur.com/mV41Al0.png)







