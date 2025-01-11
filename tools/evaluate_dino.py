from ezflow.data import DataloaderCreator
from ezflow.engine import eval_model
from ezflow.engine.tiled_model import TiledModel
from ezflow.models import build_model

log_folder = 'ft3d'
if __name__ == "__main__":
    model = build_model(
        "Dino",
        default=True,
        weights_path=f"./log/{log_folder}/dino_step_last.pth",
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
    eval_model(
        model,
        kitti_data_loader,
        device="0",
        pad_divisor=1,
        flow_scale=1.0,
    )
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
    eval_model(
        model,
        sintel,
        device="0",
        pad_divisor=1,
        flow_scale=1.0,
    )
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
    eval_model(
        model,
        sintel,
        device="0",
        pad_divisor=1,
        flow_scale=1.0,
    )


    print("Evaluation Complete!!")
