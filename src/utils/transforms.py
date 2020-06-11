from torchvision import transforms


def get_imagenet_transforms(image_size, resize_size):
    train_transforms = transforms.Compose([
        # these are borrowed from
        # https://github.com/zhirongw/lemniscate.pytorch/blob/master/main.py
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(resize_size),  # FIXME: hardcoded for 224 image size
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, test_transforms
