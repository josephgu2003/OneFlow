from ezflow.data import DataloaderCreator
from ezflow.engine.tiled_model import TiledModel
from ezflow.models import build_model
from ezflow.utils import AverageMeter, InputPadder, endpointerror
import torch 
import numpy as np
import time

import numpy as np
import torch
from torch.nn import DataParallel
from torch.profiler import profile, record_function
from tqdm import tqdm

import torchshow

log_folder = 'vq_new'

def vq_error(model, dataloader, device, metric_fn, flow_scale=1.0, pad_divisor=1):
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

            logits, magits = model.model.encode_flow(target["flow_gt"])
            logits = logits.reshape(img1.shape[0], 16, 16)
            magits = magits.reshape(img1.shape[0], 16, 16)
            bhwc = [logits.shape[0], logits.shape[1], logits.shape[2], model.model.embed_dim]
            output = model.model.decode_flow(logits, magits, bhwc, hw=img1.shape[-2:])

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

            pred = padder.unpad(output)
            pred = pred * flow_scale

            metric = metric_fn(pred, **target)
            if "valid" in target:
                metric, f1 = metric
                f1_list.append(f1)
            #print(torch.mean(torch.abs(target['flow_gt'])))
            #print(torch.mean(torch.abs(pred)))
            metric_meter.update(metric, n=batch_size)

            if i % 8 == 0:
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
    return metric_meter, avg_inference_time


if __name__ == "__main__":
    model = build_model(
        "VQFlow",
        default=True,
        weights_path=f"./log/{log_folder}/vqflow_step_last.pth",
    )
    model = TiledModel(model)
    
    dataloader_creator = DataloaderCreator(
        batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )
    dataloader_creator.add_Kitti(
        root_dir="/work/vig/Datasets/KITTI2015/",
        split="training",
        crop=True,
        crop_type="center",
        crop_size=[370, 1224],
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
    vq_error(model, kitti_data_loader, device,metric_fn=endpointerror)

    dataloader_creator = DataloaderCreator(
        batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )
    dataloader_creator.add_MPISintel(
        root_dir="/work/vig/Datasets/MPI_Sintel/",
        split="training",
        crop=True,
        crop_type="center",
        crop_size=[370, 1224],
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
        crop_size=[370, 1224],
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
