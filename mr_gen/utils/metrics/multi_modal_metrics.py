from typing import Any, List, Optional, Tuple, Dict, Union
from torch import Tensor
from torchmetrics import Metric, MeanSquaredError, MetricCollection


class SeparateMeanSquaredError(MeanSquaredError):
    def __init__(
        self,
        target_range: Tuple[int, int] = (0, -1),
        squared: bool = True,
        num_outputs: int = 1,
        **kwargs: Any
    ) -> None:
        super().__init__(squared, num_outputs, **kwargs)

        self.target_range = target_range

    def update(self, preds: Tensor, target: Tensor) -> None:
        start = self.target_range[0]
        end = self.target_range[1] if self.target_range[1] != -1 else preds.shape[-1]
        p_shape = list(preds.shape)
        t_shape = list(target.shape)

        preds = preds.reshape(-1, preds.shape[-1])[:, start:end]
        target = target.reshape(-1, target.shape[-1])[:, start:end]

        p_shape[-1] = end - start
        t_shape[-1] = end - start

        preds = preds.reshape(*p_shape).contiguous()
        target = target.reshape(*t_shape).contiguous()

        return super().update(preds, target)


class MultiTargetMetrics(MetricCollection):
    def __init__(
        self,
        target_range: Dict[str, Tuple[int, int]],
        *additional_metrics: Metric,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        compute_groups: Union[bool, List[List[str]]] = True
    ) -> None:
        if "all" not in target_range.keys():
            target_range["all"] = (0, -1)

        self.target_range = target_range
        metrics: Dict[str, Metric] = {
            key: SeparateMeanSquaredError(self.target_range[key])
            for key in self.target_range
        }
        super().__init__(
            metrics,
            *additional_metrics,
            prefix=prefix,
            postfix=postfix,
            compute_groups=compute_groups
        )
