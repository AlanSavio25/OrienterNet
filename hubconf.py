from maploc.inference import OrienterNetv2


def orienternetv2(**kwargs):
    return OrienterNetv2(
        load_from_hub=True,
        experiment_or_path="./experiment_demo/prerelease/prerelease.ckpt",
        **kwargs
    )


dependencies = ["torch", "numpy"]
