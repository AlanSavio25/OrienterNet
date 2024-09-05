from maploc.demo import Demo


def orienternet(**kwargs):
    return Demo(load_from_hub=True, experiment_or_path="./experiment_demo/prerelease/prerelease.ckpt", **kwargs)


dependencies = ['torch', 'numpy']