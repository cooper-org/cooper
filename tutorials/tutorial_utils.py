from collections import OrderedDict
from copy import deepcopy
from typing import List


class StateLogger:
    def __init__(self, save_metrics: List[str]):
        self.logger = OrderedDict()
        self.save_metrics = save_metrics

    def store_metrics(self, formulation, step_id, partial_dict: dict = None):

        aux_dict = {}

        for metric in self.save_metrics:
            if metric == "loss":
                aux_dict[metric] = formulation.cmp.state.loss.item()
            elif metric == "ineq_defect":
                aux_dict[metric] = deepcopy(formulation.cmp.state.ineq_defect.data)
            elif metric == "eq_defect":
                aux_dict[metric] = deepcopy(formulation.cmp.state.eq_defect.data)
            elif metric == "ineq_multipliers":
                aux_dict[metric] = deepcopy(formulation.state()[0].data)
            elif metric == "eq_multipliers":
                aux_dict[metric] = deepcopy(formulation.state()[1].data)

        aux_dict.update(partial_dict)
        self.save_metrics = list(set(self.save_metrics + list(partial_dict.keys())))

        self.logger[step_id] = aux_dict

    def unpack_stored_metrics(self):

        unpacked_metrics = {}
        unpacked_metrics["iters"] = list(self.logger.keys())

        for metric in self.save_metrics:
            unpacked_metrics[metric] = [_[metric] for (__, _) in self.logger.items()]

        return unpacked_metrics
