from src.preprocessing import train_transforms, eval_transforms


def test_train_transform_output_size():
    # Tensor shape is C,H,W after ToTensor.
    from PIL import Image
    image = Image.new("RGB", (640, 320))
    out = train_transforms()(image)
    assert tuple(out.shape) == (3, 224, 224)


def test_eval_transform_output_size():
    from PIL import Image
    image = Image.new("RGB", (640, 320))
    out = eval_transforms()(image)
    assert tuple(out.shape) == (3, 224, 224)
