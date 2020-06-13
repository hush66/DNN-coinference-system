import torch
from co_inference_config import *

def load_model(branchyNet):
    params = torch.load(MODEL_LOCATION)
    main_branch = params["model_main_state_dict"]
    branchyNet.main.load_state_dict(main_branch)

    for i, model in enumerate(branchyNet.models):
        model.load_state_dict(params["model_branch_%s_state_dict" % (i+1)])
    return