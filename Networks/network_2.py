"""The purpose of this network is to take a smattering of models from network_1
   and explore train/dev/test sets, vary splits and view performance. Do different
   models inherently generalize better? But then.... if all models converge to the
   same cost... should we expect performance to be different between any of them??
   Gosh that's a good question. Lots of really interesting questions going on here"""


import pandas as pd
import numpy as np
import imageio
import importlib.machinery
import os
from bokeh.palettes import Spectral11
from bokeh.plotting import figure, show
from bokeh.models import Legend, LegendItem
from itertools import chain
from PIL import Image

# load my custom built neural network functions
loader = importlib.machinery.SourceFileLoader("functions", "/users/josh.flori/drive_backup/drive_backup/pychrm_networks/homebrewed_nn_functions/functions.py")
hmbw_func = loader.load_module()



""""""""""""""""""
#    VERSION 1   #
""""""""""""""""""
# Smaller color

##################
#      PATH      #
##################
path='/users/josh.flori/drive_backup/drive_backup/pychrm_networks/data/resized_smaller_color/'
m=len([i for i in os.listdir(path) if 'jpg' in i])


##################
#    load x/y    #
##################
X = np.array([imageio.imread('/users/josh.flori/drive_backup/drive_backup/pychrm_networks/data/resized_smaller_color/'+str(i)+'.jpg').flatten() for i in range(1,m+1)]).transpose()           # So things to note here... mostly just... os.listdir returns some .DS_Store bullshit, so we sorted the list and it pops out in front...  then we get everything BUT that. It reads the matrix the wrong way so we flip it so that each column is an entire image, which I think is what we want. # 208 must be the total count of images + 1 for python to do it's thing
Y=np.array(pd.read_csv('/users/josh.flori/drive_backup/drive_backup/pychrm_networks/data/Y.csv')['Y'].values.tolist()).reshape(1,m)
X.shape       # X.shape[1], Y.shape[1] should both be equal to m, or the number of training examples
Y.shape

##########################
#    STANDARDIZE DATA    #
##########################
X = X/255


##################
#   TRAIN TEST   #
##################
indices = np.random.permutation(m)
train_end=int(m*.8)
dev_begin=int(m*.8)
dev_end=train_end+int(m*.1)
test_begin=dev_end

train_idx, dev_idx, test_idx = indices[:train_end], indices[dev_begin:dev_end], indices[test_begin:]
train_x, dev_x, test_x, train_y, dev_y, test_y = X[:,train_idx], X[:,dev_idx], X[:,test_idx], Y[0,train_idx].reshape(1,len(train_idx)), Y[0,dev_idx].reshape(1,len(dev_idx)) , Y[0,test_idx].reshape(1,len(test_idx))  # .reshape(1,len(training_idx)) is necessary because otherwise it just flips the gosh garn dimensions around and completely removes the demension 1, so dimensions are m by fucking nothing like what the heck


##########################
#   ASSERTION CHECKING   #
##########################
assert(len(train_idx)+len(dev_idx)+len(test_idx)==m)
assert(len(set(list(chain.from_iterable([train_idx.tolist(),dev_idx.tolist(),test_idx.tolist()]))))==len(list(chain.from_iterable([train_idx.tolist(),dev_idx.tolist(),test_idx.tolist()])))) # <-- make sure that no index is repeated
assert(X.shape[1]==m)
assert(Y.shape[0]==1)
assert(Y.shape[1]==m)
assert(train_x.shape[1]+dev_x.shape[1]+test_x.shape[1]==m)
assert(train_y.shape[1]+dev_y.shape[1]+test_y.shape[1]==m)
assert(train_y.shape[0]==1 and test_y.shape[0]==1)



####################################################
#   DEFINE LAYER SIZES AND INITIALIZE PARAMETERS   #
####################################################
n_x = X.shape[0]     # num_px * num_px * 3   len(pixels)
n_h_1 = 5
n_h_1_2 = 10
n_h_1_3 = 50
n_h_1_4 = 100
n_y = 1


#######################
#   INITIALIZATIONS   #
#######################
model_1_dims = [n_x,n_h_1,n_y]
model_2_dims = [n_x,n_h_1_2,n_y]
model_3_dims = [n_x,n_h_1_3,n_y]
model_4_dims = [n_x,n_h_1_4,n_y]
model_5_dims = [n_x,n_y]



###############
#    MODELS   #
###############
original_params_1,updated_parameters_1,cost_list_1,accuracy_1=hmbw_func.L_layer_model(train_x,train_y, model_1_dims, learning_rate=0.05, num_iterations=10000, print_cost=True,show_plot=True)
hmbw_func.save_outputs(1,1,original_params_1,updated_parameters_1,cost_list_1,accuracy_1,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_2_outputs/')

original_params_2,updated_parameters_2,cost_list_2,accuracy_2=hmbw_func.L_layer_model(train_x,train_y, model_2_dims, learning_rate=0.05, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(2,1,original_params_2,updated_parameters_2,cost_list_2,accuracy_2,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_2_outputs/')

original_params_3,updated_parameters_3,cost_list_3,accuracy_3=hmbw_func.L_layer_model(train_x,train_y, model_3_dims, learning_rate=0.05, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(3,1,original_params_3,updated_parameters_3,cost_list_3,accuracy_3,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_2_outputs/')

original_params_4,updated_parameters_4,cost_list_4,accuracy_4=hmbw_func.L_layer_model(train_x,train_y, model_4_dims, learning_rate=0.05, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(4,1,original_params_4,updated_parameters_4,cost_list_4,accuracy_4,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_2_outputs/')

original_params_5,updated_parameters_5,cost_list_5,accuracy_5=hmbw_func.L_layer_model(train_x,train_y, model_5_dims, learning_rate=0.05, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(5,1,original_params_5,updated_parameters_5,cost_list_5,accuracy_5,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_2_outputs/')

#############
#    PLOT   #
#############
model_df = pd.DataFrame({'[2100, 5, 1]':cost_list_1,'[2100, 10, 1]':cost_list_2,'[2100, 50, 1]': cost_list_3,'[2100, 100, 1]': cost_list_4,'[2100, 1]':cost_list_5})
numlines=len(model_df.columns)
mypalette=Spectral11[0:numlines]
p = figure(width=1420, height=800)
r=p.multi_line(xs=[model_df.index.values]*numlines,
             ys=[model_df[name].values for name in model_df],
             line_color=mypalette,
             line_width=5)
legend = Legend(items=[
    LegendItem(label=model_df.columns[0], renderers=[r], index=0),
    LegendItem(label=model_df.columns[1], renderers=[r], index=1),
    LegendItem(label=model_df.columns[2], renderers=[r], index=2),
    LegendItem(label=model_df.columns[3], renderers=[r], index=3),
    LegendItem(label=model_df.columns[4], renderers=[r], index=4)
])
p.add_layout(legend)
p.y_range.start,p.y_range.end=(0,1)
show(p)
p.y_range.start,p.y_range.end,p.x_range.start,p.x_range.end=(0,1,0,5000)
show(p)


# DEV CHECK
model_1_predictions=hmbw_func.predict_from_output(updated_parameters_1,dev_x)
hmbw_func.accuracy(model_1_predictions,dev_y)

model_2_predictions=hmbw_func.predict_from_output(updated_parameters_2,dev_x)
hmbw_func.accuracy(model_2_predictions,dev_y)

model_3_predictions=hmbw_func.predict_from_output(updated_parameters_3,dev_x)
hmbw_func.accuracy(model_3_predictions,dev_y)

model_4_predictions=hmbw_func.predict_from_output(updated_parameters_4,dev_x)
hmbw_func.accuracy(model_4_predictions,dev_y)

model_5_predictions=hmbw_func.predict_from_output(updated_parameters_5,dev_x)
hmbw_func.accuracy(model_5_predictions,dev_y)



# TEST CHECK
model_1_predictions=hmbw_func.predict_from_output(updated_parameters_1,test_x)
hmbw_func.accuracy(model_1_predictions,test_y)

model_2_predictions=hmbw_func.predict_from_output(updated_parameters_2,test_x)
hmbw_func.accuracy(model_2_predictions,test_y)

model_3_predictions=hmbw_func.predict_from_output(updated_parameters_3,test_x)
hmbw_func.accuracy(model_3_predictions,test_y)

model_4_predictions=hmbw_func.predict_from_output(updated_parameters_4,test_x)
hmbw_func.accuracy(model_4_predictions,test_y)

model_5_predictions=hmbw_func.predict_from_output(updated_parameters_5,test_x)
hmbw_func.accuracy(model_5_predictions,test_y)


###################################
#    TAKE AWAYS FROM THIS MODEL   #
###################################
# 1) Um, pretty much perfect performance. I may see some different things if I used a larger and more varied data set





















""""""""""""""""""
#    VERSION 2   #
""""""""""""""""""
# Smaller black and white

##################
#    load x/y    #
##################
X=np.array([np.array(Image.open('/users/josh.flori/drive_backup/drive_backup/pychrm_networks/data/resized_smaller_color/'+str(i)+'.jpg').convert('LA')).flatten() for i in range(1,m+1)]).transpose()
Y=np.array(pd.read_csv('/users/josh.flori/drive_backup/drive_backup/pychrm_networks/data/Y.csv')['Y'].values.tolist()).reshape(1,m)
X.shape       # X.shape[1], Y.shape[1] should both be equal to m, or the number of training examples
Y.shape



##################
#   TRAIN TEST   #
##################
indices = np.random.permutation(m)
train_end=int(m*.8)
dev_begin=int(m*.8)
dev_end=train_end+int(m*.1)
test_begin=dev_end

train_idx, dev_idx, test_idx = indices[:train_end], indices[dev_begin:dev_end], indices[test_begin:]
train_x, dev_x, test_x, train_y, dev_y, test_y = X[:,train_idx], X[:,dev_idx], X[:,test_idx], Y[0,train_idx].reshape(1,len(train_idx)), Y[0,dev_idx].reshape(1,len(dev_idx)) , Y[0,test_idx].reshape(1,len(test_idx))  # .reshape(1,len(training_idx)) is necessary because otherwise it just flips the gosh garn dimensions around and completely removes the demension 1, so dimensions are m by fucking nothing like what the heck


##########################
#   ASSERTION CHECKING   #
##########################
assert(len(train_idx)+len(dev_idx)+len(test_idx)==m)
assert(len(set(list(chain.from_iterable([train_idx.tolist(),dev_idx.tolist(),test_idx.tolist()]))))==len(list(chain.from_iterable([train_idx.tolist(),dev_idx.tolist(),test_idx.tolist()])))) # <-- make sure that no index is repeated
assert(X.shape[1]==m)
assert(Y.shape[0]==1)
assert(Y.shape[1]==m)
assert(train_x.shape[1]+dev_x.shape[1]+test_x.shape[1]==m)
assert(train_y.shape[1]+dev_y.shape[1]+test_y.shape[1]==m)
assert(train_y.shape[0]==1 and test_y.shape[0]==1)



####################################################
#   DEFINE LAYER SIZES AND INITIALIZE PARAMETERS   #
####################################################
n_x = X.shape[0]     # num_px * num_px * 3   len(pixels)
n_h_1 = 5
n_h_1_2 = 10
n_h_1_3 = 50
n_h_1_4 = 100
n_y = 1


#######################
#   INITIALIZATIONS   #
#######################
model_1_dims = [n_x,n_h_1,n_y]
model_2_dims = [n_x,n_h_1_2,n_y]
model_3_dims = [n_x,n_h_1_3,n_y]
model_4_dims = [n_x,n_h_1_4,n_y]
model_5_dims = [n_x,n_y]



###############
#    MODELS   #
###############
original_params_1,updated_parameters_1,cost_list_1,accuracy_1=hmbw_func.L_layer_model(train_x,train_y, model_1_dims, learning_rate=0.00001, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(1,2,original_params_1,updated_parameters_1,cost_list_1,accuracy_1,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_2_outputs/')

original_params_2,updated_parameters_2,cost_list_2,accuracy_2=hmbw_func.L_layer_model(train_x,train_y, model_2_dims, learning_rate=0.00001, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(2,2,original_params_2,updated_parameters_2,cost_list_2,accuracy_2,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_2_outputs/')

original_params_3,updated_parameters_3,cost_list_3,accuracy_3=hmbw_func.L_layer_model(train_x,train_y, model_3_dims, learning_rate=0.00001, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(3,2,original_params_3,updated_parameters_3,cost_list_3,accuracy_3,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_2_outputs/')

original_params_4,updated_parameters_4,cost_list_4,accuracy_4=hmbw_func.L_layer_model(train_x,train_y, model_4_dims, learning_rate=0.00001, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(4,2,original_params_4,updated_parameters_4,cost_list_4,accuracy_4,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_2_outputs/')

original_params_5,updated_parameters_5,cost_list_5,accuracy_5=hmbw_func.L_layer_model(train_x,train_y, model_5_dims, learning_rate=0.0000005, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(5,2,original_params_5,updated_parameters_5,cost_list_5,accuracy_5,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_2_outputs/')





#############
#    PLOT   #
#############
model_df = pd.DataFrame({'[2100, 5, 1]':cost_list_1,'[2100, 10, 1]':cost_list_2,'[2100, 50, 1]': cost_list_3,'[2100, 100, 1]': cost_list_4,'[2100, 1]':cost_list_5})
numlines=len(model_df.columns)
mypalette=Spectral11[0:numlines]
p = figure(width=1420, height=800)
r=p.multi_line(xs=[model_df.index.values]*numlines,
             ys=[model_df[name].values for name in model_df],
             line_color=mypalette,
             line_width=5)
legend = Legend(items=[
    LegendItem(label=model_df.columns[0], renderers=[r], index=0),
    LegendItem(label=model_df.columns[1], renderers=[r], index=1),
    LegendItem(label=model_df.columns[2], renderers=[r], index=2),
    LegendItem(label=model_df.columns[3], renderers=[r], index=3),
    LegendItem(label=model_df.columns[4], renderers=[r], index=4)
])
p.add_layout(legend)
p.y_range.start,p.y_range.end=(0,1)
show(p)
p.y_range.start,p.y_range.end,p.x_range.start,p.x_range.end=(0,1,0,5000)
show(p)


# DEV CHECK
model_1_predictions=hmbw_func.predict_from_output(updated_parameters_1,dev_x)
hmbw_func.accuracy(model_1_predictions,dev_y)

model_2_predictions=hmbw_func.predict_from_output(updated_parameters_2,dev_x)
hmbw_func.accuracy(model_2_predictions,dev_y)

model_3_predictions=hmbw_func.predict_from_output(updated_parameters_3,dev_x)
hmbw_func.accuracy(model_3_predictions,dev_y)

model_4_predictions=hmbw_func.predict_from_output(updated_parameters_4,dev_x)
hmbw_func.accuracy(model_4_predictions,dev_y)

model_5_predictions=hmbw_func.predict_from_output(updated_parameters_5,dev_x)
hmbw_func.accuracy(model_5_predictions,dev_y)



# TEST CHECK
model_1_predictions=hmbw_func.predict_from_output(updated_parameters_1,test_x)
hmbw_func.accuracy(model_1_predictions,test_y)

model_2_predictions=hmbw_func.predict_from_output(updated_parameters_2,test_x)
hmbw_func.accuracy(model_2_predictions,test_y)

model_3_predictions=hmbw_func.predict_from_output(updated_parameters_3,test_x)
hmbw_func.accuracy(model_3_predictions,test_y)

model_4_predictions=hmbw_func.predict_from_output(updated_parameters_4,test_x)
hmbw_func.accuracy(model_4_predictions,test_y)

model_5_predictions=hmbw_func.predict_from_output(updated_parameters_5,test_x)
hmbw_func.accuracy(model_5_predictions,test_y)


###################################
#    TAKE AWAYS FROM THIS MODEL   #
###################################
# 1) Learning rates...
#    a) First of all, ALL of these require much smaller learning rates than the colored images. That makes me think that smaller dimensional inputs are simpler solution
#       spaces requiring less time to descend or something like that idk.
#    b) For model a, 0001 is too large (it wastes a lot of time jittering around the descent). 000001 is too small but 00001 is juuust right. Crazy!
#    c) The others are about the same but the logistic unit needs to be at .0000005 or lower which is MUCH smaller than the other ones. So not only is input smaller but
#       the number of weights are smaller, which makes the... solution space smaller? What is a solution space? How do inputs and weights effect the time it takes to descent?
#       ^^^^ that's an important question.
# 2) Train/Dev/Test performance..
#    a) Again, all perfect performance. I think this would be different if the images were more varied.











""""""""""""""""""
#    VERSION 3   #
""""""""""""""""""
# Larger black and white # only tried this because the smallers were getting nanned at first, but we may as well continue and see what we can get from it, see how size plays in.

##################
#    load x/y    #
##################
X=np.array([np.array(Image.open('/users/josh.flori/drive_backup/drive_backup/pychrm_networks/data/resized_larger_black_white/'+str(i)+'.jpg')).flatten() for i in range(1,m+1)]).transpose()
Y=np.array(pd.read_csv('/users/josh.flori/drive_backup/drive_backup/pychrm_networks/data/Y.csv')['Y'].values.tolist()).reshape(1,m)
X.shape       # X.shape[1], Y.shape[1] should both be equal to m, or the number of training examples
Y.shape



##################
#   TRAIN TEST   #
##################
indices = np.random.permutation(m)
train_end=int(m*.8)
dev_begin=int(m*.8)
dev_end=train_end+int(m*.1)
test_begin=dev_end

train_idx, dev_idx, test_idx = indices[:train_end], indices[dev_begin:dev_end], indices[test_begin:]
train_x, dev_x, test_x, train_y, dev_y, test_y = X[:,train_idx], X[:,dev_idx], X[:,test_idx], Y[0,train_idx].reshape(1,len(train_idx)), Y[0,dev_idx].reshape(1,len(dev_idx)) , Y[0,test_idx].reshape(1,len(test_idx))  # .reshape(1,len(training_idx)) is necessary because otherwise it just flips the gosh garn dimensions around and completely removes the demension 1, so dimensions are m by fucking nothing like what the heck


##########################
#   ASSERTION CHECKING   #
##########################
assert(len(train_idx)+len(dev_idx)+len(test_idx)==m)
assert(len(set(list(chain.from_iterable([train_idx.tolist(),dev_idx.tolist(),test_idx.tolist()]))))==len(list(chain.from_iterable([train_idx.tolist(),dev_idx.tolist(),test_idx.tolist()])))) # <-- make sure that no index is repeated
assert(X.shape[1]==m)
assert(Y.shape[0]==1)
assert(Y.shape[1]==m)
assert(train_x.shape[1]+dev_x.shape[1]+test_x.shape[1]==m)
assert(train_y.shape[1]+dev_y.shape[1]+test_y.shape[1]==m)
assert(train_y.shape[0]==1 and test_y.shape[0]==1)



####################################################
#   DEFINE LAYER SIZES AND INITIALIZE PARAMETERS   #
####################################################
n_x = X.shape[0]     # num_px * num_px * 3   len(pixels)
n_h_1 = 5
n_h_1_2 = 10
n_h_1_3 = 50
n_h_1_4 = 100
n_y = 1


#######################
#   INITIALIZATIONS   #
#######################
model_1_dims = [n_x,n_h_1,n_y]
model_2_dims = [n_x,n_h_1_2,n_y]
model_3_dims = [n_x,n_h_1_3,n_y]
model_4_dims = [n_x,n_h_1_4,n_y]
model_5_dims = [n_x,n_y]



###############
#    MODELS   #
###############
original_params_1,updated_parameters_1,cost_list_1,accuracy_1=hmbw_func.L_layer_model(train_x,train_y, model_1_dims, learning_rate=0.00001, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(1,3,original_params_1,updated_parameters_1,cost_list_1,accuracy_1,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_2_outputs/')
original_params_2,updated_parameters_2,cost_list_2,accuracy_2=hmbw_func.L_layer_model(train_x,train_y, model_2_dims, learning_rate=0.00001, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(2,3,original_params_2,updated_parameters_2,cost_list_2,accuracy_2,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_2_outputs/')
original_params_3,updated_parameters_3,cost_list_3,accuracy_3=hmbw_func.L_layer_model(train_x,train_y, model_3_dims, learning_rate=0.00001, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(3,3,original_params_3,updated_parameters_3,cost_list_3,accuracy_3,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_2_outputs/')
original_params_4,updated_parameters_4,cost_list_4,accuracy_4=hmbw_func.L_layer_model(train_x,train_y, model_4_dims, learning_rate=0.00001, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(4,3,original_params_4,updated_parameters_4,cost_list_4,accuracy_4,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_2_outputs/')
original_params_5,updated_parameters_5,cost_list_5,accuracy_5=hmbw_func.L_layer_model(train_x,train_y, model_5_dims, learning_rate=0.00000005, num_iterations=10000, print_cost=True,show_plot=False)
hmbw_func.save_outputs(5,3,original_params_5,updated_parameters_5,cost_list_5,accuracy_5,'/users/josh.flori/drive_backup/drive_backup/pychrm_networks/network versions/network_2_outputs/')



#############
#    PLOT   #
#############
model_df = pd.DataFrame({'[2100, 5, 1]':cost_list_1,'[2100, 10, 1]':cost_list_2,'[2100, 50, 1]': cost_list_3,'[2100, 100, 1]': cost_list_4,'[2100, 1]':cost_list_5})
numlines=len(model_df.columns)
mypalette=Spectral11[0:numlines]
p = figure(width=1420, height=800)
r=p.multi_line(xs=[model_df.index.values]*numlines,
             ys=[model_df[name].values for name in model_df],
             line_color=mypalette,
             line_width=5)
legend = Legend(items=[
    LegendItem(label=model_df.columns[0], renderers=[r], index=0),
    LegendItem(label=model_df.columns[1], renderers=[r], index=1),
    LegendItem(label=model_df.columns[2], renderers=[r], index=2),
    LegendItem(label=model_df.columns[3], renderers=[r], index=3),
    LegendItem(label=model_df.columns[4], renderers=[r], index=4)
])
p.add_layout(legend)
p.y_range.start,p.y_range.end=(0,1)
show(p)



# DEV CHECK
model_1_predictions=hmbw_func.predict_from_output(updated_parameters_1,dev_x)
hmbw_func.accuracy(model_1_predictions,dev_y)

model_2_predictions=hmbw_func.predict_from_output(updated_parameters_2,dev_x)
hmbw_func.accuracy(model_2_predictions,dev_y)

model_3_predictions=hmbw_func.predict_from_output(updated_parameters_3,dev_x)
hmbw_func.accuracy(model_3_predictions,dev_y)

model_4_predictions=hmbw_func.predict_from_output(updated_parameters_4,dev_x)
hmbw_func.accuracy(model_4_predictions,dev_y)

model_5_predictions=hmbw_func.predict_from_output(updated_parameters_5,dev_x)
hmbw_func.accuracy(model_5_predictions,dev_y)



# TEST CHECK
model_1_predictions=hmbw_func.predict_from_output(updated_parameters_1,test_x)
hmbw_func.accuracy(model_1_predictions,test_y)

model_2_predictions=hmbw_func.predict_from_output(updated_parameters_2,test_x)
hmbw_func.accuracy(model_2_predictions,test_y)

model_3_predictions=hmbw_func.predict_from_output(updated_parameters_3,test_x)
hmbw_func.accuracy(model_3_predictions,test_y)

model_4_predictions=hmbw_func.predict_from_output(updated_parameters_4,test_x)
hmbw_func.accuracy(model_4_predictions,test_y)

model_5_predictions=hmbw_func.predict_from_output(updated_parameters_5,test_x)
hmbw_func.accuracy(model_5_predictions,test_y)


###################################
#    TAKE AWAYS FROM THIS MODEL   #
###################################
# 1) Learning rates...
#    a) Nothing new here other than the logistic unit rate had to be reduced by a decimal place to .00000005 which is weird and contradicts my other findings which are:
# 2) Learning time
#    a) As shown in the previous models, it looks like the larger a solution space is the longer a time it takes to learn. From network_1 which had large networks, those took forever.
#       Then from this network with fewer layers, the colored images took way longer still than the black and white images. So why the logistic unit was so noisy at the same learning
#       rate as version2 (which had smaller images) I don't know. May it had to do with the intializations but somehow I doubt that.
# 3) Unlike the smaller black and white images, a couple of these smaller networks had a different solution space, you can see a plateau then a quick drop, so that's interesting.. you
#    can sort of visualize them learning different functions. WOW I'm typing more now than I usually do that's for sure. It actually feels pretty good, like loosening up old joints
# 4) Performance? Perfect, again. Small errors on dev but I chalk that up to something we can't define given the small number of images.








#####################
#   GOING FORWARD  #
#####################
# It looks like that on this dataset I will not be able to use regularization since performance was so good... so I will need to use new data.
# I could use optimization on the larger networks though. I think I will start there so I can wrap up this dataset, and then look into
# Regularization on different data


