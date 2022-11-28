from collections import OrderedDict
from copy import deepcopy
from typing import List

from cooper.problem import CMPState


class StateLogger:
    """
    Utility for storing optimization metrics (e.g. loss, multipliers) through
    training.

    Args:
        save_metrics: List of metric names to be stored. Currently supported
            values are: ``loss``, ``ineq_defect``, ``eq_defect``,
            ``ineq_multipliers``, ``eq_multipliers``.
    """

    def __init__(self, save_metrics: List[str]):
        self.logger: OrderedDict = OrderedDict()
        self.save_metrics = save_metrics

    def store_metrics(
        self,
        cmp_state: CMPState,
        step_id: int,
        partial_dict: dict = None,
    ):
        """
        Store a new screenshot of the metrics.

        Args:
            cmp_state: State of the CMP to be stored.
            step_id: Identifier for the optimization step.
            partial_dict: Auxiliary dictionary with other metrics to be logged,
                but which are not part of the "canonical" options available in
                ``save_metrics``. Defaults to None.
        """
        aux_dict = {}

        for metric in self.save_metrics:
            if metric == "loss":
                aux_dict[metric] = cmp_state.loss.item()
            elif metric == "ineq_defect":
                aux_dict[metric] = deepcopy(cmp_state.ineq_defect.data)
            elif metric == "eq_defect":
                aux_dict[metric] = deepcopy(cmp_state.eq_defect.data)

        if partial_dict is not None:
            aux_dict.update(partial_dict)
            self.save_metrics = list(set(self.save_metrics + list(partial_dict.keys())))

        self.logger[step_id] = aux_dict

    def unpack_stored_metrics(self) -> dict:
        """
        Returns a dictionary containing the stored values separated by metric.
        """
        unpacked_metrics = {}
        unpacked_metrics["iters"] = list(self.logger.keys())

        for metric in self.save_metrics:
            unpacked_metrics[metric] = [_[metric] for (__, _) in self.logger.items()]

        return unpacked_metrics
