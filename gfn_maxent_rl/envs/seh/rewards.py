from typing import List
from torch import Tensor
import torch_geometric.data as gd

from gflownet.models import bengio2021flow
from gflownet.utils.misc import get_worker_device
from rdkit.Chem.rdchem import Mol as RDMol


class RewardProxy():
    def __init__(self) -> None:
        self.model = self._load_task_models()

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model.to(get_worker_device())
        # this is used in the original repo to optionally send model to separate workers
        # model = self._wrap_model(model)
        return model
    
    def flat_reward_transform(self, reward):
        # ?? why do they do this
        return reward/8
    
    def compute_flat_reward(self, mols: List[RDMol]) -> Tensor:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        preds = self.compute_reward_from_graph(graphs).reshape((-1, 1))

        return preds
        
    def compute_reward_from_graph(self, graphs: List[gd.Data]) -> Tensor:
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.model.device if hasattr(self.model, 'device') else get_worker_device())
        preds = self.model(batch).reshape((-1,)).data.cpu()
        preds[preds.isnan()] = 0
        return self.flat_reward_transform(preds).clip(1e-4, 100).reshape((-1,))