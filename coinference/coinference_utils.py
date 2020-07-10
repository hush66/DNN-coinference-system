import torch
from co_inference_config import *

def load_model(branchyNet):
    params = torch.load(MODEL_LOCATION)
    # if load on raspberry, since there is no gpu support on raspberry
    #params = torch.load(MODEL_LOCATION, map_location = {'cuda:0': 'cpu'})
    
    main_branch = params["model_main_state_dict"]
    branchyNet.main.load_state_dict(main_branch)

    for i, model in enumerate(branchyNet.models):
        model.load_state_dict(params["model_branch_%s_state_dict" % (i+1)])
    return