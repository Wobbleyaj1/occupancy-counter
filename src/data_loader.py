import os
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define data transformations
data_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Resize((256, 256)),  # Resize the image
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize
    ]
)


class CaltechDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load annotations (adjust depending on your annotation format)
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_info = self.annotations[idx]
        img_path = os.path.join(self.image_dir, img_info["file_name"])
        image = cv2.imread(img_path)

        # Load bounding boxes (adjust as needed)
        boxes = img_info["annotations"]

        if self.transform:
            image = self.transform(image)

        return image, boxes


# Testing code (this part can be at the bottom of the file)
if __name__ == "__main__":
    # Set up paths and parameters for testing
    image_dir = "data/caltech/images/"  # Path to the images
    annotation_file = (
        "data/caltech/annotations/annotations.json"  # Adjust this path as needed
    )

    # Check if the image directory exists
    if not os.path.exists(image_dir):
        print(f"Image directory does not exist: {image_dir}")

    # Check if the annotation file exists
    if not os.path.exists(annotation_file):
        print(f"Annotation file does not exist: {annotation_file}")

    # Create the dataset and data loader
    caltech_dataset = CaltechDataset(
        image_dir=image_dir, annotation_file=annotation_file, transform=data_transforms
    )

    caltech_loader = DataLoader(caltech_dataset, batch_size=4, shuffle=True)

    # Test the data loader
    for images, boxes in caltech_loader:
        for img, box in zip(images, boxes):
            plt.imshow(
                img.permute(1, 2, 0)
            )  # Convert tensor to image format (C, H, W) -> (H, W, C)
            plt.axis("off")

            # Display bounding boxes
            for b in box:
                plt.gca().add_patch(
                    plt.Rectangle(
                        (b[0], b[1]),
                        b[2] - b[0],
                        b[3] - b[1],
                        fill=False,
                        color="red",
                        linewidth=2,
                    )
                )

            plt.show()  # Show image with bounding boxes
        break  # Remove this break if you want to iterate over more batches
