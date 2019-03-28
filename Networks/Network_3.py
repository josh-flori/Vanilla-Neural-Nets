"""The purpose of this network is to try to speed up
   learning on bigger images using the last version
   of network_2"""


import pandas as pd
import numpy as np
import importlib.machinery
from bokeh.palettes import Spectral11
from bokeh.plotting import figure, show
from bokeh.models import Legend, LegendItem
from itertools import chain
from PIL import Image

loader = importlib.machinery.SourceFileLoader("functions", "/users/josh.flori/drive_backup/drive_backup/pychrm_networks/homebrewed_nn_functions/functions.py")
hmbw_func = loader.load_module()




""""""""""""""""""
#    VERSION 3   #
""""""""""""""""""
# Larger black and white # only tried this because the smallers were getting nanned at first, but we may as well continue and see what we can get from it, see how size plays in.
path='/users/josh.flori/drive_backup/drive_backup/pychrm_networks/data/resized_larger_black_white/'
m=len([i for i in os.listdir(path) if 'jpg' in i])


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
n_y = 1


#######################
#   INITIALIZATIONS   #
#######################
model_1_dims = [n_x,n_h_1_2,n_y]



###############
#    MODELS   #
###############
# base initialization
# average alpha
original_params_1,updated_parameters_1,cost_list_1,accuracy_1=hmbw_func.L_layer_model(train_x,train_y, model_1_dims, learning_rate=0.00001, num_iterations=10000, print_cost=True,show_plot=False)
# smaller alpha
original_params_2,updated_parameters_2,cost_list_1,accuracy_2=hmbw_func.L_layer_model(train_x,train_y, model_1_dims, learning_rate=0.0001, num_iterations=10000, print_cost=True,show_plot=False)
# larger alpha
original_params_3,updated_parameters_3,cost_list_3,accuracy_3=hmbw_func.L_layer_model(train_x,train_y, model_1_dims, learning_rate=0.000001, num_iterations=10000, print_cost=True,show_plot=False)
# much smaller alpha
original_params_4,updated_parameters_4,cost_list_4,accuracy_4=hmbw_func.L_layer_model(train_x,train_y, model_1_dims, learning_rate=0.001, num_iterations=10000, print_cost=True,show_plot=False)
# so i tried HE initialization and mostly got a bunch of nans. I either had to set the learning rate to some absurdly small number and watch it fail to learn quickly
# or set high and watch it go from 6 cost to nan the next iteration every time, and some infs thrown in for good measure. looks like there's nothing i can do with this
# for now but that's ok, I don't even remember how it worked or what it did on a technical level
original_params_5,cost_list_5 = hmbw_func.optimized_model(train_x, train_y, model_1_dims, learning_rate=0.0001, num_epochs=10000, print_cost=True, optimizer = "gd")
original_params_6,cost_list_6 = hmbw_func.optimized_model(train_x, train_y, model_1_dims, learning_rate=0.0001, num_epochs=10000, print_cost=True, optimizer = "momentum")
original_params_7,cost_list_7 = hmbw_func.optimized_model(train_x, train_y, model_1_dims, learning_rate=0.0001, num_epochs=10000, print_cost=True, optimizer = "adam")
original_params_8,cost_list_8 = hmbw_func.optimized_model(train_x, train_y, model_1_dims, learning_rate=0.00001, num_epochs=10000, print_cost=True, optimizer = "adam")


model_df_1 = pd.DataFrame({'.00001 alpha, no optimization':cost_list_1[0:10000],'.0001 alpha, no optimization':cost_list_2[0:10000],'.000001 alpha, no optimization':cost_list_3[0:10000],'.001 alpha, no optimization':cost_list_4[0:10000],'.0001 minibatch':cost_list_5,'.0001 momentum':cost_list_6,'.0001 adam???':cost_list_7,'.00001 adam!':cost_list_8})
numlines=len(model_df_1.columns)
mypalette=Spectral11[0:numlines]
p = figure(width=1420, height=800)
r=p.multi_line(xs=[model_df_1.index.values]*numlines,
             ys=[model_df_1[name].values for name in model_df_1],
             line_color=mypalette,
             line_width=5)
legend = Legend(items=[
    LegendItem(label=model_df_1.columns[0], renderers=[r], index=0),
    LegendItem(label=model_df_1.columns[1], renderers=[r], index=1),
    LegendItem(label=model_df_1.columns[2], renderers=[r], index=2),
    LegendItem(label=model_df_1.columns[3], renderers=[r], index=3),
    LegendItem(label=model_df_1.columns[4], renderers=[r], index=4),
    LegendItem(label=model_df_1.columns[5], renderers=[r], index=5),
    LegendItem(label=model_df_1.columns[6], renderers=[r], index=6),
    LegendItem(label=model_df_1.columns[7], renderers=[r], index=7)
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