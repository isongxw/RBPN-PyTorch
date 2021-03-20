from torchvision.transforms import Compose, ToTensor
from dataset import DatasetFromFolderTest, DatasetFromFolder


def transform():
    return Compose([
        ToTensor(),
    ])


def get_training_set(LR_dir, HR_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame):
    return DatasetFromFolder(LR_dir, HR_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame,
                             transform=transform())


def get_eval_set(data_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame):
    return DatasetFromFolder(data_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame,
                             transform=transform())


def get_test_set(LR_dir, HR_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame):
    return DatasetFromFolderTest(LR_dir, HR_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=transform())
