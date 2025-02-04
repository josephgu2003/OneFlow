from matplotlib import pyplot as plt
from ezflow.data import DataloaderCreator
from ezflow.engine.tiled_model import TiledModel
from ezflow.models import build_model
from ezflow.utils import AverageMeter, InputPadder
import torch 
import numpy as np
import time

import numpy as np
import torch
from torch.nn import DataParallel
from torch.profiler import profile, record_function
from tqdm import tqdm

from diffusers.models import AutoencoderKL
import torchshow

from ezflow.utils.invert_flow import colorize_mag, decolorize_mag, decompose_flow, pad_flow, unpad_flow

log_folder = 'vq_new'

def endpointerror(pred, flow_gt, valid=None, multi_magnitude=False, **kwargs):
    """
    Endpoint error

    Parameters
    ----------
    pred : torch.Tensor
        Predicted flow
    flow_gt : torch.Tensor
        flow_gt flow
    valid : torch.Tensor
        Valid flow vectors

    Returns
    -------
    torch.Tensor
        Endpoint error
    """
    if isinstance(pred, tuple) or isinstance(pred, list):
        pred = pred[-1]

    epe = torch.norm(pred - flow_gt, p=2, dim=1)
    
    mags = torch.norm(flow_gt, p=2, dim=1, keepdim=True)
    epe_keepdim = torch.norm(pred-flow_gt, p=2, dim=1, keepdim=True)
    mag0 = mags < 10
    mag10 = (mags > 10) * (mags <= 50)
    mag50 = (mags > 50)
    
    f1 = None

    if valid is not None:
        mag = torch.sum(flow_gt**2, dim=1).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid.reshape(-1) >= 0.5

        f1 = ((epe > 3.0) & ((epe / mag) > 0.05)).float()

        epe = epe[val]
        f1 = f1[val].cpu().numpy()

    if not multi_magnitude:
        if f1 is not None:
            return epe.mean().item(), f1

        return epe.mean().item(), [epe_keepdim.cpu(), mag0.cpu(), mag10.cpu(), mag50.cpu()]

    assert False 
    epe = epe.view(-1)
    multi_magnitude_epe = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    if f1 is not None:
        return multi_magnitude_epe, f1

    return multi_magnitude_epe


def vq_error(model: AutoencoderKL, dataloader, device, metric_fn, flow_scale=1.0, pad_divisor=1):
    """
    Uses a model to perform inference on a dataloader and captures inference time and evaluation metric

    Parameters
    ----------
    model : torch.nn.Module
        Model to be used for prediction / inference
    dataloader : torch.utils.data.DataLoader
        Dataloader to be used for prediction / inference
    device : torch.device
        Device (CUDA / CPU) to be used for prediction / inference
    metric_fn : function
        Function to be used to calculate the evaluation metric
    flow_scale : float, optional
        Scale factor to be applied to the predicted flow
    pad_divisor : int, optional
        The divisor to make the image dimensions evenly divisible by using padding, by default 1

    Returns
    -------
    metric_meter : AverageMeter
        AverageMeter object containing the evaluation metric information
    avg_inference_time : float
        Average inference time

    """

    metric_meter = AverageMeter()
    times = []

    inp, target = next(iter(dataloader))
    batch_size = target["flow_gt"].shape[0]
    f1_list = []
    padder = InputPadder(inp[0].shape, divisor=pad_divisor)

    mags = []
    
    mag_metadata = []
    
    with torch.no_grad():

        for i, (inp, target) in tqdm(enumerate(dataloader)):

            img1, img2 = inp
            img1, img2 = img1.to(device), img2.to(device)
            for key, val in target.items():
                target[key] = val.to(device)

            img1, img2 = padder.pad(img1, img2)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.time()
            
        #    flow_gt = pad_flow(target['flow_gt']) /40 
         #   output = unpad_flow(model(flow_gt).sample) * 40
            dir, mag = decompose_flow(target['flow_gt'])
            mag0 = mag
            dir0 = dir
            mags.append(mag.flatten().cpu().numpy())
            mag = mag / 40
          #  mag = colorize_mag(mag)
            dir = model(dir).sample
            mag = model(mag).sample * 40
        #    mag = decolorize_mag(model(mag).sample)
            output = dir * mag
            output = output[:, :2]
         
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

            pred = padder.unpad(output)
            pred = pred * flow_scale

            metric, mag_d = metric_fn(pred, **target)
            mag_metadata.append(mag_d)
            if "valid" in target:
                metric, f1 = metric
                f1_list.append(f1)
            #print(torch.mean(torch.abs(target['flow_gt'])))
            #print(torch.mean(torch.abs(pred)))
            metric_meter.update(metric, n=batch_size)

            if i % 8 == 0:
                torchshow.save(dir[0])
                torchshow.save(mag[0])
                torchshow.save(img1[0])
                torchshow.save(img2[0])
                torchshow.save(target['flow_gt'][0])
                torchshow.save(output[0])

    avg_inference_time = sum(times) / len(times)
    avg_inference_time /= batch_size  # Average inference time per sample

    print("=" * 100)
    if avg_inference_time != 0:
        print(
            f"Average inference time: {avg_inference_time}, FPS: {1/avg_inference_time}"
        )
    if f1_list:
        f1_list = np.concatenate(f1_list)
        f1_all = 100 * np.mean(f1_list)
        print(f"F1-all: {f1_all}")

    print(f"Average evaluation metric = {metric_meter.avg}")
    mags = np.stack(mags).flatten()
    y, x = np.histogram(mags, bins=np.arange(200))
    fig, ax = plt.subplots()
    ax.plot(x[:-1], y)
    fig.savefig('hist.png')
    
    epe_keepdim = torch.stack([x[0] for x in mag_metadata])
    mag0 = torch.stack([x[1] for x in mag_metadata])
    mag10 = torch.stack([x[2] for x in mag_metadata])
    mag50 = torch.stack([x[3] for x in mag_metadata])
    
    print("Mag0: ", (torch.sum(epe_keepdim * mag0) / torch.sum(epe_keepdim)).item())
    print("Mag10: ", (torch.sum(epe_keepdim * mag10) / torch.sum(epe_keepdim)).item())
    print("Mag50: ", (torch.sum(epe_keepdim * mag50) / torch.sum(epe_keepdim)).item())
    return metric_meter, avg_inference_time


if __name__ == "__main__":
    model = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")
    model.eval() # !! keep in eval
     
    dataloader_creator = DataloaderCreator(
        batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )
    dataloader_creator.add_Kitti(
        root_dir="/work/vig/Datasets/KITTI2015/",
        split="training",
        crop=True,
        crop_type="center",
        crop_size=[256, 256],
        norm_params={
            "use": True,
            "mean": (127.5, 127.5, 127.5),
            "std": (127.5, 127.5, 127.5),
        },
    )

    kitti_data_loader = dataloader_creator.get_dataloader()

    # code here
    device= 'cuda:0' 
    model = model.to(device)
    model.eval()
   # vq_error(model, kitti_data_loader, device,metric_fn=endpointerror)

    dataloader_creator = DataloaderCreator(
        batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )
    dataloader_creator.add_MPISintel(
        root_dir="/work/vig/Datasets/MPI_Sintel/",
        split="training",
        crop=True,
        crop_type="center",
        crop_size=[256, 256],
        dstype='clean',
        norm_params={
            "use": True,
            "mean": (127.5, 127.5, 127.5),
            "std": (127.5, 127.5, 127.5),
        },
    )

    sintel = dataloader_creator.get_dataloader()
    vq_error(model, sintel, device,metric_fn=endpointerror)
    dataloader_creator = DataloaderCreator(
        batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )
    dataloader_creator.add_MPISintel(
        root_dir="/work/vig/Datasets/MPI_Sintel/",
        split="training",
        crop=True,
        crop_type="center",
        crop_size=[256, 256],
        dstype='final',
        norm_params={
            "use": True,
            "mean": (127.5, 127.5, 127.5),
            "std": (127.5, 127.5, 127.5),
        },
    )

    sintel = dataloader_creator.get_dataloader()

    vq_error(model, sintel, device,metric_fn=endpointerror)

    print("Evaluation Complete!!")
