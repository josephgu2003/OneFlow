import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import configurable
from ..registry import FUNCTIONAL_REGISTRY


@FUNCTIONAL_REGISTRY.register()
class VQLoss(nn.Module):
    """
    Multi-scale loss for optical flow estimation.
    Used in **DICL** (https://papers.nips.cc/paper/2020/hash/add5aebfcb33a2206b6497d53bc4f309-Abstract.html)

    Parameters
    ----------
    norm : str, default: "l1"
        The norm to use for the loss. Can be either "l2", "l1" or "robust"
    q : float, default: 0.4
        This parameter is used in robust loss for fine tuning. q < 1 gives less penalty to outliers
    eps : float, default: 0.01
        This parameter is a small constant used in robust loss to stabilize fine tuning.
    weights : list
        The weights to use for each scale
    average : str, default: "mean"
        The mode to set the average of the EPE map.
        If "mean", the mean of the EPE map is returned.
        If "sum", the EPE map is summed and divided by the batch size.
    resize_flow : str, default: "upsample"
        The mode to resize flow.
        If "upsample", predicted flow will be upsampled to the size of the ground truth.
        If "downsample", ground truth flow will be downsampled to the size of the predicted flow.
    extra_mask : torch.Tensor
        A mask to apply to the loss. Useful for removing the loss on the background
    use_valid_range : bool
        Whether to use the valid range of flow values for the loss
    valid_range : list
        The valid range of flow values for each scale
    """

    @configurable
    def __init__(
        self,
        norm="l1",
        q=0.4,
        eps=1e-2,
        weights=(1, 0.5, 0.25),
        average="mean",
        resize_flow="upsample",
        extra_mask=None,
        use_valid_range=True,
        valid_range=None,
        **kwargs
    ):
        super(VQLoss, self).__init__()

        assert norm.lower() in (
            "l1",
            "l2",
            "robust",
        ), "Norm must be one of L1, L2, Robust"
        assert resize_flow.lower() in (
            "upsample",
            "downsample",
        ), "Resize flow must be one of upsample or downsample"
        assert average.lower() in ("mean", "sum"), "Average must be one of mean or sum"

        self.norm = norm.lower()
        self.q = q
        self.eps = eps
        self.weights = weights
        self.extra_mask = extra_mask
        self.use_valid_range = use_valid_range
        self.valid_range = valid_range
        self.average = average.lower()
        self.resize_flow = resize_flow.lower()

    @classmethod
    def from_config(cls, cfg):
        return {
            "norm": cfg.NORM,
            "weights": cfg.WEIGHTS,
            "average": cfg.AVERAGE,
            "resize_flow": cfg.RESIZE_FLOW,
            "extra_mask": cfg.EXTRA_MASK,
            "use_valid_range": cfg.USE_VALID_RANGE,
            "valid_range": cfg.VALID_RANGE,
        }

    def forward(self, latents, gt_latents, my_mask, **kwargs):
        return (self.loss(latents[0], gt_latents[0], my_mask, **kwargs) + self.loss(latents[1], gt_latents[1], my_mask, **kwargs)) * 0.5

    def loss(self, latents, gt_latents, my_mask, **kwargs):
        flow_preds = latents
        flow_gt = gt_latents.reshape(flow_preds.shape[0], flow_preds.shape[2], flow_preds.shape[3])
        nan_mask = (~torch.isnan(flow_gt)).float()
        flow_gt[torch.isnan(flow_gt)] = 0.0
        target = flow_gt 
        
        if self.use_valid_range and self.valid_range is not None:
            
            with torch.no_grad():
                mask = (target[:, 0, :, :].abs() <= self.valid_range[i][1]) & (
                    target[:, 1, :, :].abs() <= self.valid_range[i][0]
                )
        else:
            with torch.no_grad():
                mask = torch.ones(target[:, :, :].shape).type_as(target)
                
        loss_value = F.cross_entropy(flow_preds, flow_gt, reduction='none')
        #loss_value = loss_value * mask.float() * my_mask.float()
        loss_value = loss_value * mask.float() # TODO: account for masks

        if self.extra_mask is not None:
            val = self.extra_mask > 0
            loss_value = loss_value[val]

        return torch.mean(loss_value)
