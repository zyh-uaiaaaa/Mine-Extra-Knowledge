
def get_model(name):

    if name == "swin_t":
        from .swin import SwinTransformer
        return SwinTransformer(num_classes=512)

    else:
        raise ValueError()
