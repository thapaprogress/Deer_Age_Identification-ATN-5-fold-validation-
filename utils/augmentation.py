"""
Augmentation Pipeline for Deer Age Recognition
"""

from torchvision import transforms

def get_training_transforms(image_size=224):
    """
    Get training data transformations with augmentation
    
    Args:
        image_size: Target image size
        
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_validation_transforms(image_size=224):
    """
    Get validation/testing data transformations (no augmentation)
    
    Args:
        image_size: Target image size
        
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
