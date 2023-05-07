import torch
import numpy as np

from douzero.env.env import get_obs
from douzero.env.env_douzero import get_obs_douzero
from douzero.env.env_res import _get_obs_resnet

def _load_model(position, model_path, model_type):
    from douzero.dmc.models import model_dict, model_dict_douzero
    if model_type == "test":
        model = model_dict_douzero[position]()
    elif model_type == "best":
        from douzero.dmc.models_res import model_dict_resnet
        model = model_dict_resnet[position]()
    else:
        model = model_dict[position]()
    model_state_dict = model.state_dict()
    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    # torch.save(model.state_dict(), model_path.replace(".ckpt", "_nobn.ckpt"))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

class DeepAgent:

    def __init__(self, position, model_path):
        if "test" in model_path:
            self.model_type = "test"
        elif "best" in model_path:
            self.model_type = "best"
        else:
            self.model_type = "new"
        self.model = _load_model(position, model_path, self.model_type)
        self.EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                            8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                            13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}

    def act(self, infoset):
        if self.model_type == "test":
            obs = get_obs_douzero(infoset)
        elif self.model_type == "best":
            obs = _get_obs_resnet(infoset, infoset.player_position)
        else:
            obs = get_obs(infoset, bid_over=infoset.bid_over, new_model=True)

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]
        if obs["position"] == "landlord":
            ccc = 1
        return best_action
