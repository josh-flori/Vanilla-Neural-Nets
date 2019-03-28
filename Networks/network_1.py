"""The purpose of this network is to use the functions created in the class
   on my own data. And with those functions and data, explore a variety of
   basic architectures. Vary the number of hidden layers, units per layer,
   learning rate and number of iterations enough to get a good feel for
   different behavior and in different circumstances"""

import pandas as pd
import numpy as np
import imageio
import importlib.machinery
import os
from bokeh.palettes import Spectral11
from bokeh.plotting import figure, show
from bokeh.models import Legend, LegendItem


# load my custom built neural network functions
loader = importlib.machinery.SourceFileLoader("functions", "/users/josh.flori/drive_backup/drive_backup/pychrm_networks/homebrewed_nn_functions/functions.py")
hmbw_func = loader.load_module()




##################
#      PATH      #
##################
path='/users/josh.flori/drive_backup/drive_backup/pychrm_networks/data/images resized/'
m=len([i for i in os.listdir(path) if 'jpg' in i])


##################
#    load x/y    #
##################
X = np.array([imageio.imread('/users/josh.flori/drive_backup/drive_backup/pychrm_networks/data/images resized/'+str(i)+'.jpg').flatten() for i in range(1,m+1)]).transpose()           # So things to note here... mostly just... os.listdir returns some .DS_Store bullshit, so we sorted the list and it pops out in front...  then we get everything BUT that. It reads the matrix the wrong way so we flip it so that each column is an entire image, which I think is what we want. # 208 must be the total count of images + 1 for python to do it's thing
Y=np.array(pd.read_csv('/users/josh.flori/drive_backup/drive_backup/pychrm_networks/data/Y.csv')['Y'].values.tolist()).reshape(1,m)
X.shape       # X.shape[1], Y.shape[1] should both be equal to m, or the number of training examples
Y.shape


##########################
#    STANDARDIZE DATA    #
##########################
X = X/255



##########################
#   ASSERTION CHECKING   #
##########################
assert(X.shape[1]==m)
assert(Y.shape[0]==1)
assert(Y.shape[1]==m)


####################################################
#   DEFINE LAYER SIZES AND INITIALIZE PARAMETERS   #
####################################################
n_x = X.shape[0]     # num_px * num_px * 3   len(pixels)
n_h_1 = 5
n_h_1_2 = 10
n_h_1_3 = 50
n_h_2 = 5
n_h_2_2 = 10
n_h_2_3 = 50
n_h_3 = 5
n_h_4 = 5
n_h_5 = 5
n_y = 1


##################################
#     BATCH1+2 INITIALIZATIONS   #
##################################
# Testing number and size of hidden layers, learning rate .05 iterations 10,000
layer_dims_1 = [n_x,n_h_1,n_h_2,n_h_3,n_h_4,n_h_5,n_y]
layer_dims_2 = [n_x,n_h_1,n_h_2,n_h_3,n_h_4,n_y]
layer_dims_3 = [n_x,n_h_1,n_h_2,n_h_3,n_y]
layer_dims_4 = [n_x,n_h_1,n_h_2,n_y]
layer_dims_5 = [n_x,n_h_1_2,n_h_2,n_y]
layer_dims_6 = [n_x,n_h_1_3,n_h_2,n_y]
layer_dims_7 = [n_x,n_h_1,n_h_2_2,n_y]
layer_dims_8 = [n_x,n_h_1,n_h_2_3,n_y]
layer_dims_9 = [n_x,n_h_1_3,n_h_2_3,n_y]


""""""""""""""""""""""""
#     BATCH_1 MODELS   #
""""""""""""""""""""""""
# larger networks, smaller learning rates
original_params_1,updated_parameters_1,cost_list_1,accuracy_1=hmbw_func.L_layer_model(X,Y, layer_dims_1, learning_rate=0.05, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(1,original_params_1,updated_parameters_1,cost_list_1,accuracy_1,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

original_params_2,updated_parameters_2,cost_list_2,accuracy_2=hmbw_func.L_layer_model(X,Y, layer_dims_2, learning_rate=0.05, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(2,original_params_2,updated_parameters_2,cost_list_2,accuracy_2,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

original_params_3,updated_parameters_3,cost_list_3,accuracy_3=hmbw_func.L_layer_model(X,Y, layer_dims_3, learning_rate=0.05, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(3,original_params_3,updated_parameters_3,cost_list_3,accuracy_3,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

original_params_4,updated_parameters_4,cost_list_4,accuracy_4=hmbw_func.L_layer_model(X,Y, layer_dims_4, learning_rate=0.05, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(4,original_params_4,updated_parameters_4,cost_list_4,accuracy_4,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

original_params_5,updated_parameters_5,cost_list_5,accuracy_5=hmbw_func.L_layer_model(X,Y, layer_dims_5, learning_rate=0.05, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(5,original_params_5,updated_parameters_5,cost_list_5,accuracy_5,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

original_params_6,updated_parameters_6,cost_list_6,accuracy_6=hmbw_func.L_layer_model(X,Y, layer_dims_6, learning_rate=0.05, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(6,original_params_6,updated_parameters_6,cost_list_6,accuracy_6,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

original_params_7,updated_parameters_7,cost_list_7,accuracy_7=hmbw_func.L_layer_model(X,Y, layer_dims_7, learning_rate=0.05, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(7,original_params_7,updated_parameters_7,cost_list_7,accuracy_7,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

original_params_8,updated_parameters_8,cost_list_8,accuracy_8=hmbw_func.L_layer_model(X,Y, layer_dims_8, learning_rate=0.05, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(8,original_params_8,updated_parameters_8,cost_list_8,accuracy_8,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

original_params_9,updated_parameters_9,cost_list_9,accuracy_9=hmbw_func.L_layer_model(X,Y, layer_dims_9, learning_rate=0.05, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(9,original_params_9,updated_parameters_9,cost_list_9,accuracy_9,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

############################
#     BATCH1 ASSERTIONS    #
############################
hmbw_func.check_model_parameter_updates(original_params_1,updated_parameters_1)
hmbw_func.check_model_parameter_updates(original_params_2,updated_parameters_2)
hmbw_func.check_model_parameter_updates(original_params_3,updated_parameters_3)
hmbw_func.check_model_parameter_updates(original_params_4,updated_parameters_4)
hmbw_func.check_model_parameter_updates(original_params_5,updated_parameters_5)
hmbw_func.check_model_parameter_updates(original_params_6,updated_parameters_6)
hmbw_func.check_model_parameter_updates(original_params_7,updated_parameters_7)
hmbw_func.check_model_parameter_updates(original_params_8,updated_parameters_8)
hmbw_func.check_model_parameter_updates(original_params_9,updated_parameters_9)

######################
#     BATCH1 PLOT    #
######################
batch_1_df = pd.DataFrame({'[2100, 5, 5, 5, 5, 5, 1]':cost_list_1,'[2100, 5, 5, 5, 5, 1]':cost_list_2,'[2100, 5, 5, 5, 1]': cost_list_3,'[2100, 5, 5, 1]': cost_list_4,'[2100, 10, 5, 1]': cost_list_5,'[2100, 50, 5, 1]': cost_list_6,'[2100, 5, 10, 1]': cost_list_7,'[2100, 5, 50, 1]': cost_list_8,'[2100, 50, 50, 1]': cost_list_9})
numlines=len(batch_1_df.columns)
mypalette=Spectral11[0:numlines]
p = figure(width=1920, height=1080)
r=p.multi_line(xs=[batch_1_df.index.values]*numlines,
             ys=[batch_1_df[name].values for name in batch_1_df],
             line_color=mypalette,
             line_width=5)
legend = Legend(items=[
    LegendItem(label=batch_1_df.columns[0], renderers=[r], index=0),
    LegendItem(label=batch_1_df.columns[1], renderers=[r], index=1),
    LegendItem(label=batch_1_df.columns[2], renderers=[r], index=2),
    LegendItem(label=batch_1_df.columns[3], renderers=[r], index=3),
    LegendItem(label=batch_1_df.columns[4], renderers=[r], index=4),
    LegendItem(label=batch_1_df.columns[5], renderers=[r], index=5),
    LegendItem(label=batch_1_df.columns[6], renderers=[r], index=6),
    LegendItem(label=batch_1_df.columns[7], renderers=[r], index=7),
    LegendItem(label=batch_1_df.columns[8], renderers=[r], index=8)
])
p.add_layout(legend)
p.y_range.start,p.y_range.end,p.x_range.start,p.x_range.end=(0,1,0,50000) # 1:3 go flat all the way to 100,000
show(p)


###################################
#    TAKE AWAYS FROM THIS MODEL   #
###################################

# 1) Performance for more than 2 hidden layers (as with this batch of models where we have 3+) is just bad. They take a long time to optimize or (seemingly) never optimize at all as with the first 3 models
#    a) We see large upward spikes for models 7:9 before they leave the plateau. I don't know what's different about those than models 4:6 since they have similar numbers of layers/nodes
#    b) It looks like anything more than 4 layers like ([n_x,n_h_1_2,n_h_2,n_y]) either doesn't optimize or takes forever to optimize
#    c) But even 3 layers with the smallest number of total nodes out of all models still takes MORE time to optimize than 4 layer functions with more nodes!! Why is that? The only improvement is that it does not experience a huge spike before optimizing
#    d) Although a different run with different initializations moved it a bit, I'm surprised the 9th model optimizes before the 8th considering it is so much smaller. ..
#          i) layer_dims_8 = [2100, 5, 50, 1]
#          ii)layer_dims_9 = [2100, 50, 50, 1]









""""""""""""""""""""""""
#     BATCH_2 MODELS   #
""""""""""""""""""""""""
# larger networks, larger learning rates
batch_2_original_params_1,batch_2_updated_parameters_1,batch_2_cost_list_1,batch_2_accuracy_1=hmbw_func.L_layer_model(X,Y, layer_dims_1, learning_rate=0.1, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(1,2,batch_2_original_params_1,batch_2_updated_parameters_1,batch_2_cost_list_1,batch_2_accuracy_1,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_2_original_params_2,batch_2_updated_parameters_2,batch_2_cost_list_2,batch_2_accuracy_2=hmbw_func.L_layer_model(X,Y, layer_dims_2, learning_rate=0.1, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(2,2,batch_2_original_params_2,batch_2_updated_parameters_2,batch_2_cost_list_2,batch_2_accuracy_2,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_2_original_params_3,batch_2_updated_parameters_3,batch_2_cost_list_3,batch_2_accuracy_3=hmbw_func.L_layer_model(X,Y, layer_dims_3, learning_rate=0.1, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(3,2,batch_2_original_params_3,batch_2_updated_parameters_3,batch_2_cost_list_3,batch_2_accuracy_3,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_2_original_params_4,batch_2_updated_parameters_4,batch_2_cost_list_4,batch_2_accuracy_4=hmbw_func.L_layer_model(X,Y, layer_dims_4, learning_rate=0.1, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(4,2,batch_2_original_params_4,batch_2_updated_parameters_4,batch_2_cost_list_4,accuracy_4,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_2_original_params_5,batch_2_updated_parameters_5,batch_2_cost_list_5,batch_2_accuracy_5=hmbw_func.L_layer_model(X,Y, layer_dims_5, learning_rate=0.1, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(5,2,batch_2_original_params_5,batch_2_updated_parameters_5,batch_2_cost_list_5,batch_2_accuracy_5,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_2_original_params_6,batch_2_updated_parameters_6,batch_2_cost_list_6,batch_2_accuracy_6=hmbw_func.L_layer_model(X,Y, layer_dims_6, learning_rate=0.1, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(6,2,batch_2_original_params_6,batch_2_updated_parameters_6,batch_2_cost_list_6,batch_2_accuracy_6,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_2_original_params_7,batch_2_updated_parameters_7,batch_2_cost_list_7,batch_2_accuracy_7=hmbw_func.L_layer_model(X,Y, layer_dims_7, learning_rate=0.1, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(7,2,batch_2_original_params_7,batch_2_updated_parameters_7,batch_2_cost_list_7,batch_2_accuracy_7,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_2_original_params_8,batch_2_updated_parameters_8,batch_2_cost_list_8,batch_2_accuracy_8=hmbw_func.L_layer_model(X,Y, layer_dims_8, learning_rate=0.1, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(8,2,batch_2_original_params_8,batch_2_updated_parameters_8,batch_2_cost_list_8,batch_2_accuracy_8,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_2_original_params_9,batch_2_updated_parameters_9,batch_2_cost_list_9,batch_2_accuracy_9=hmbw_func.L_layer_model(X,Y, layer_dims_9, learning_rate=0.1, num_iterations=100000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(9,2,batch_2_original_params_9,batch_2_updated_parameters_9,batch_2_cost_list_9,batch_2_accuracy_9,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

############################
#     BATCH2 ASSERTIONS    #
############################
hmbw_func.check_model_parameter_updates(batch_2_original_params_4,batch_2_updated_parameters_1)
hmbw_func.check_model_parameter_updates(batch_2_original_params_4,batch_2_updated_parameters_2)
hmbw_func.check_model_parameter_updates(batch_2_original_params_4,batch_2_updated_parameters_3)
hmbw_func.check_model_parameter_updates(batch_2_original_params_4,batch_2_updated_parameters_4)
hmbw_func.check_model_parameter_updates(batch_2_original_params_4,batch_2_updated_parameters_5)
hmbw_func.check_model_parameter_updates(batch_2_original_params_4,batch_2_updated_parameters_6)
hmbw_func.check_model_parameter_updates(batch_2_original_params_4,batch_2_updated_parameters_7)
hmbw_func.check_model_parameter_updates(batch_2_original_params_4,batch_2_updated_parameters_8)
hmbw_func.check_model_parameter_updates(batch_2_original_params_4,batch_2_updated_parameters_9)

######################
#     BATCH2 PLOT    #
######################
batch_2_df = pd.DataFrame({'[2100, 5, 5, 5, 5, 5, 1]':batch_2_cost_list_1,'[2100, 5, 5, 5, 5, 1]':batch_2_cost_list_2,'[2100, 5, 5, 5, 1]': batch_2_cost_list_3,'[2100, 5, 5, 1]': batch_2_cost_list_4,'[2100, 10, 5, 1]': batch_2_cost_list_5,'[2100, 50, 5, 1]': batch_2_cost_list_6,'[2100, 5, 10, 1]': batch_2_cost_list_7,'[2100, 5, 50, 1]': batch_2_cost_list_8,'[2100, 50, 50, 1]': batch_2_cost_list_9})
numlines=len(batch_2_df.columns)
mypalette=Spectral11[0:numlines]
p = figure(width=1420, height=800)
r=p.multi_line(xs=[batch_2_df.index.values]*numlines,
             ys=[batch_2_df[name].values for name in batch_2_df],
             line_color=mypalette,
             line_width=5)
legend = Legend(items=[
    LegendItem(label=batch_2_df.columns[0], renderers=[r], index=0),
    LegendItem(label=batch_2_df.columns[1], renderers=[r], index=1),
    LegendItem(label=batch_2_df.columns[2], renderers=[r], index=2),
    LegendItem(label=batch_2_df.columns[3], renderers=[r], index=3),
    LegendItem(label=batch_2_df.columns[4], renderers=[r], index=4),
    LegendItem(label=batch_2_df.columns[5], renderers=[r], index=5),
    LegendItem(label=batch_2_df.columns[6], renderers=[r], index=6),
    LegendItem(label=batch_2_df.columns[7], renderers=[r], index=7),
    LegendItem(label=batch_2_df.columns[8], renderers=[r], index=8)

])
p.add_layout(legend)
#p.y_range.start,p.y_range.end,p.x_range.start,p.x_range.end=(0,1,0,50000) # 1:3 go flat all the way to 100,000
show(p)


###################################
#    TAKE AWAYS FROM THIS MODEL   #
###################################

# um idk off the top of my head other than when you increase the learning rate it speeds up learning.
# My rate is .1 which seems very large compared to stuff I've seen online so I'm not sure what the difference
# is in terms of why I can use it without seeming consequence but it's a general bad idea.

# so... im curious with batch gradient descent how... cost can be flat for thousands of iterations then
# suddenly get not flat. how does it move if it's completely flat?
# but it looks like the more layers the longer it takes to descend, or rather, the longer the plateus
# are... but what about same number of layers and larger number of neurons?








""""""""""""""""""""""""
#     BATCH_3 MODELS   #
""""""""""""""""""""""""
# smaller networks, smaller learning rates
####################################################
#   DEFINE LAYER SIZES AND INITIALIZE PARAMETERS   #
####################################################
n_x = X.shape[0]     # num_px * num_px * 3   len(pixels)
n_h_1 = 5
n_h_1_2 = 10
n_h_1_3 = 50
n_h_1_4 = 100
n_h_1_5 = 1000
n_y = 1


#################################
#     BATCH_3 INITIALIZATIONS   #
#################################
batch3_layer_dims_1 = [n_x,n_h_1,n_y]
batch3_layer_dims_2 = [n_x,n_h_1_2,n_y]
batch3_layer_dims_3 = [n_x,n_h_1_3,n_y]
batch3_layer_dims_4 = [n_x,n_h_1_4,n_y]
batch3_layer_dims_5 = [n_x,n_h_1_5,n_y]
batch3_layer_dims_6 = [n_x,n_y]


batch_3_original_params_1,batch_3_updated_parameters_1,batch_3_cost_list_1,batch_3_accuracy_1=hmbw_func.L_layer_model(X,Y, batch3_layer_dims_1, learning_rate=0.05, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(1,3,batch_3_original_params_1,batch_3_updated_parameters_1,batch_3_cost_list_1,batch_3_accuracy_1,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')


batch_3_original_params_2,batch_3_updated_parameters_2,batch_3_cost_list_2,batch_3_accuracy_2=hmbw_func.L_layer_model(X,Y, batch3_layer_dims_2, learning_rate=0.05, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(2,3,batch_3_original_params_2,batch_3_updated_parameters_2,batch_3_cost_list_2,batch_3_accuracy_2,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_3_original_params_3,batch_3_updated_parameters_3,batch_3_cost_list_3,batch_3_accuracy_3=hmbw_func.L_layer_model(X,Y, batch3_layer_dims_3, learning_rate=0.05, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(3,3,batch_3_original_params_3,batch_3_updated_parameters_3,batch_3_cost_list_3,batch_3_accuracy_3,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_3_original_params_4,batch_3_updated_parameters_4,batch_3_cost_list_4,batch_3_accuracy_4=hmbw_func.L_layer_model(X,Y, batch3_layer_dims_4, learning_rate=0.05, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(4,3,batch_3_original_params_4,batch_3_updated_parameters_4,batch_3_cost_list_4,batch_3_accuracy_4,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_3_original_params_5,batch_3_updated_parameters_5,batch_3_cost_list_5,batch_3_accuracy_5=hmbw_func.L_layer_model(X,Y, batch3_layer_dims_5, learning_rate=0.05, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(5,3,batch_3_original_params_5,batch_3_updated_parameters_5,batch_3_cost_list_5,batch_3_accuracy_5,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_3_original_params_6,batch_3_updated_parameters_6,batch_3_cost_list_6,batch_3_accuracy_6=hmbw_func.L_layer_model(X,Y, batch3_layer_dims_6, learning_rate=0.05, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(6,3,batch_3_original_params_6,batch_3_updated_parameters_6,batch_3_cost_list_6,batch_3_accuracy_6,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')



############################
#     BATCH3 ASSERTIONS    #
############################
hmbw_func.check_model_parameter_updates(batch_3_original_params_1,batch_3_updated_parameters_1)
hmbw_func.check_model_parameter_updates(batch_3_original_params_2,batch_3_updated_parameters_2)
hmbw_func.check_model_parameter_updates(batch_3_original_params_3,batch_3_updated_parameters_3)
hmbw_func.check_model_parameter_updates(batch_3_original_params_4,batch_3_updated_parameters_4)
hmbw_func.check_model_parameter_updates(batch_3_original_params_5,batch_3_updated_parameters_5)
hmbw_func.check_model_parameter_updates(batch_3_original_params_6,batch_3_updated_parameters_6)

######################
#     BATCH3 PLOT    #
######################
batch_3_df = pd.DataFrame({'[2100, 5, 1]':batch_3_cost_list_1,'[2100, 10, 1]':batch_3_cost_list_2,'[2100, 50, 1]': batch_3_cost_list_3,'[2100, 100, 1]': batch_3_cost_list_4,'[2100, 1000, 1]': batch_3_cost_list_5,'[2100, 1]': batch_3_cost_list_6})
numlines=len(batch_3_df.columns)
mypalette=Spectral11[0:numlines]
p = figure(width=1420, height=800)
r=p.multi_line(xs=[batch_3_df.index.values]*numlines,
             ys=[batch_3_df[name].values for name in batch_3_df],
             line_color=mypalette,
             line_width=5)
legend = Legend(items=[
    LegendItem(label=batch_3_df.columns[0], renderers=[r], index=0),
    LegendItem(label=batch_3_df.columns[1], renderers=[r], index=1),
    LegendItem(label=batch_3_df.columns[2], renderers=[r], index=2),
    LegendItem(label=batch_3_df.columns[3], renderers=[r], index=3),
    LegendItem(label=batch_3_df.columns[4], renderers=[r], index=4),
    LegendItem(label=batch_3_df.columns[5], renderers=[r], index=5)

])
p.add_layout(legend)
p.y_range.start,p.y_range.end,p.x_range.start,p.x_range.end=(0,3,0,900) # 1:3 go flat all the way to 100,000
show(p)


###################################
#    TAKE AWAYS FROM THIS MODEL   #
###################################
# 1) Some really interesting things happening here...
#    a) First of all, all of these are billions of times faster than the previous models... and this isn't even a fast learning rate or highly optimized!
#    b) So let me get this right.... when it comes to having 2 hidden layers, the MORE neurons there are, the FEWER iterations it takes to begin descending
#       but the longer the iterations take, with 1,000 neurons taking a hella long time, many many times longer than the others
#    c) But NONE of the models are NEARLY as fast at descending as just the logistic unit alone. But it has some weird noise, I wonder if it's because it's dumber
#    d) Perhaps similar to the logistic unit, the model with only 5 neurons had a weird squiggle, maybe more neurons help smooth out the function, or something...
# 2) I would be curious if the whole "more neurons in a layer = fewer iterations to descend" is true across the board, or in what circumstances





""""""""""""""""""""""""
#     BATCH_4 MODELS   #
""""""""""""""""""""""""
# smaller networks, higher learning rates
batch_4_original_params_1,batch_4_updated_parameters_1,batch_4_cost_list_1,batch_4_accuracy_1=hmbw_func.L_layer_model(X,Y, batch3_layer_dims_1, learning_rate=0.1, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(1,4,batch_4_original_params_1,batch_4_updated_parameters_1,batch_4_cost_list_1,batch_4_accuracy_1,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')


batch_4_original_params_2,batch_4_updated_parameters_2,batch_4_cost_list_2,batch_4_accuracy_2=hmbw_func.L_layer_model(X,Y, batch3_layer_dims_2, learning_rate=0.1, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(2,4,batch_4_original_params_2,batch_4_updated_parameters_2,batch_4_cost_list_2,batch_4_accuracy_2,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_4_original_params_3,batch_4_updated_parameters_3,batch_4_cost_list_3,batch_4_accuracy_3=hmbw_func.L_layer_model(X,Y, batch3_layer_dims_3, learning_rate=0.1, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(3,4,batch_4_original_params_3,batch_4_updated_parameters_3,batch_4_cost_list_3,batch_4_accuracy_3,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_4_original_params_4,batch_4_updated_parameters_4,batch_4_cost_list_4,batch_4_accuracy_4=hmbw_func.L_layer_model(X,Y, batch3_layer_dims_4, learning_rate=0.1, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(4,4,batch_4_original_params_4,batch_4_updated_parameters_4,batch_4_cost_list_4,batch_4_accuracy_4,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_4_original_params_5,batch_4_updated_parameters_5,batch_4_cost_list_5,batch_4_accuracy_5=hmbw_func.L_layer_model(X,Y, batch3_layer_dims_5, learning_rate=0.1, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(5,4,batch_4_original_params_5,batch_4_updated_parameters_5,batch_4_cost_list_5,batch_4_accuracy_5,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')

batch_4_original_params_6,batch_4_updated_parameters_6,batch_4_cost_list_6,batch_4_accuracy_6=hmbw_func.L_layer_model(X,Y, batch3_layer_dims_6, learning_rate=0.1, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(6,4,batch_4_original_params_6,batch_4_updated_parameters_6,batch_4_cost_list_6,batch_4_accuracy_6,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_1_outputs/')



############################
#     BATCH4 ASSERTIONS    #
############################
hmbw_func.check_model_parameter_updates(batch_4_original_params_1,batch_4_updated_parameters_1)
hmbw_func.check_model_parameter_updates(batch_4_original_params_2,batch_4_updated_parameters_2)
hmbw_func.check_model_parameter_updates(batch_4_original_params_3,batch_4_updated_parameters_3)
hmbw_func.check_model_parameter_updates(batch_4_original_params_4,batch_4_updated_parameters_4)
hmbw_func.check_model_parameter_updates(batch_4_original_params_6,batch_4_updated_parameters_6)

######################
#     BATCH4 PLOT    #
######################
batch_4_df = pd.DataFrame({'[2100, 5, 1]':batch_4_cost_list_1,'[2100, 10, 1]':batch_4_cost_list_2,'[2100, 50, 1]': batch_4_cost_list_3,'[2100, 100, 1]': batch_4_cost_list_4,'[2100, 1]': batch_4_cost_list_6})
numlines=len(batch_4_df.columns)
mypalette=Spectral11[0:numlines]
p = figure(width=1420, height=800)
r=p.multi_line(xs=[batch_4_df.index.values]*numlines,
             ys=[batch_4_df[name].values for name in batch_4_df],
             line_color=mypalette,
             line_width=5)
legend = Legend(items=[
    LegendItem(label=batch_4_df.columns[0], renderers=[r], index=0),
    LegendItem(label=batch_4_df.columns[1], renderers=[r], index=1),
    LegendItem(label=batch_4_df.columns[2], renderers=[r], index=2),
    LegendItem(label=batch_4_df.columns[3], renderers=[r], index=3),
    LegendItem(label=batch_4_df.columns[4], renderers=[r], index=4)

])
p.add_layout(legend)
p.y_range.start,p.y_range.end,p.x_range.start,p.x_range.end=(0,3,0,900) # 1:3 go flat all the way to 100,000
show(p)


###################################
#    TAKE AWAYS FROM THIS MODEL   #
###################################
# 1) Convergence was only slightly faster for all but one, but VERY noisy, which is not great.
#    a) The really interesting thing is that a .1 learning rate (seems) to have had a better effect for larger than smaller networks.
#       It seems to have created less noise (though maybe I wasn't zoomed in far enough to see it). But it definitely sped up learning more
#       (though I think it was already at some optimally fast level for the smaller networks).
# 2) Overall, not preferable to the .05 learning rate on the same networks, but still good to see this visually.





