from maploc.demo import Demo


def orienternet(**kwargs):
    return Demo(load_from_hub=True, **kwargs)


dependencies = ['torch', 'numpy']