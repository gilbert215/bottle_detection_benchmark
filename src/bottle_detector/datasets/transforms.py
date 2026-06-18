"""Image transforms / augmentation pipelines used across all detectors."""
import torchvision.transforms as T


def get_train_transforms() -> T.Compose:
    """Light augmentation used for the baseline (10-epoch) SSD run."""
    return T.Compose(
        [
            T.ToTensor(),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    )


def get_advanced_train_transforms() -> T.Compose:
    """Heavier augmentation, used for longer fine-tuning runs."""
    return T.Compose(
        [
            T.ToTensor(),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomResizedCrop(size=(300, 300), scale=(0.8, 1.0)),
        ]
    )


def get_eval_transforms() -> T.Compose:
    """No augmentation — used for validation/test/inference."""
    return T.Compose([T.ToTensor()])
