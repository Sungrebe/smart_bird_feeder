# import torch
# import torch.nn.functional as F
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image, UnidentifiedImageError

# resnet50 = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")

# model = torch.nn.Sequential(*list(resnet50.children())[:-1])
# model.eval()

# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     )
# ])

# def process_image(img_path):
#     try:
#         img = Image.open(img_path).convert('RGB')
#         input_tensor = transform(img).unsqueeze(0)

#         with torch.no_grad():
#             embedding = model(input_tensor)
        
#         return embedding.view(embedding.size(0), -1)
#     except UnidentifiedImageError:
#         return None

# img_1 = process_image('house_finch_001_augmented.jpg')
# img_2 = process_image('house_finch_002_augmented.jpg')
# img_3 = process_image('cardinal.png')

# print(f"img1: {img_1.cpu().numpy().flatten()}")
# print(f"img2: {img_2.cpu().numpy().flatten()}")
# print(f"img3: {img_3.cpu().numpy().flatten()}")

# print(len(img_1.cpu().numpy().flatten()))
# print(len(img_2.cpu().numpy().flatten()))
# print(len(img_3.cpu().numpy().flatten()))

# print(f"Cosine similarity: {F.cosine_similarity(img_1, img_2)}")
# print(f"Cosine similarity: {F.cosine_similarity(img_1, img_3)}")

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import timm
import time
import os
from PIL import ImageFile, Image
from tqdm import tqdm
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True

train_folder = "/projects/illinois/eng/ece/kindrtnk/smart_bird_feeder/feeder_birds_dataset/photos/train"
test_folder = "/projects/illinois/eng/ece/kindrtnk/smart_bird_feeder/feeder_birds_dataset/photos/test"
val_folder = "/projects/illinois/eng/ece/kindrtnk/smart_bird_feeder/feeder_birds_dataset/photos/val"

# Get a list of actual class folders
class_folders = [f for f in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, f))]
num_classes = len(class_folders)  # Set num_classes based on actual class folders

model_no_fc_ready_to_fine_tune = timm.create_model("resnet50d.ra2_in1k", pretrained=True, num_classes=num_classes)
resnet_model = model_no_fc_ready_to_fine_tune

data_config = resolve_data_config({}, model=resnet_model)
transform = create_transform(**data_config)

# Custom dataset class to handle image loading errors
class ImageFolderWithErrorHandler(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super(ImageFolderWithErrorHandler, self).__getitem__(index)
        except Exception as e:
            print(f"Error loading image at index {index}: {e}")
            # Return a default image or skip the image
            # For example, you can return a black image:
            # return torch.zeros(3, 224, 224), 0  # Assuming image size is 224x224
            # Or skip the image and return None:
            #return None
            # Instead of returning None, return a black image with a label of -1
            return torch.zeros(3, 224, 224), -1

train_dataset = ImageFolderWithErrorHandler(root=train_folder, transform=transform)
test_dataset = ImageFolderWithErrorHandler(root=test_folder, transform=transform)
val_dataset = ImageFolderWithErrorHandler(root=val_folder, transform=transform)

# Filter out samples with label -1 from the dataset
train_dataset.samples = [s for s in train_dataset.samples if s[1] != -1]
test_dataset.samples = [s for s in test_dataset.samples if s[1] != -1]
val_dataset.samples = [s for s in val_dataset.samples if s[1] != -1] # Added filtering for val_dataset
# Recalculate the targets after filtering
train_dataset.targets = [s[1] for s in train_dataset.samples]
test_dataset.targets = [s[1] for s in test_dataset.samples]
val_dataset.targets = [s[1] for s in val_dataset.samples] # Added target recalculation for val_dataset


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

def fine_tune():
  if torch.backends.mps.is_available():
      device = "mps"
  else:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print(f"Using device: {device}")

  resnet_model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.AdamW(resnet_model.parameters(), lr=0.001)

  num_epochs = 6
  training_start = time.time()
  for epoch in range(num_epochs):
      epoch_start = time.time()
      resnet_model.train()
      running_loss = 0.0
      correct = 0
      total = 0
      for inputs, labels in tqdm(train_loader):
        valid_indices = labels != -1
        inputs = inputs[valid_indices]
        labels = labels[valid_indices]

        if len(labels) == 0:
          continue

        inputs, labels = inputs.to(device), labels.to(device)
        # Add the following lines to synchronize streams:
        #torch.cuda.synchronize()  # Synchronize before forward pass # Removed synchronization calls
        optimizer.zero_grad()
        outputs = resnet_model(inputs)
        #torch.cuda.synchronize()  # Synchronize before loss calculation
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        #torch.cuda.synchronize()  # Synchronize before backward pass
        loss.backward()
        #torch.cuda.synchronize()  # Synchronize before optimizer step
        optimizer.step()
        running_loss += loss.item()

      training_loss = running_loss / len(train_loader)
      print(f"\nEpoch [{epoch+1}/{num_epochs}], Training Loss: {round(training_loss, 5)}")

      train_accuracy = 100 * (correct / total)
      print(f"Training Accuracy: {round(train_accuracy, 5)}%")

      # Evaluate on test and val sets
      resnet_model.eval()
      running_loss = 0.0
      correct = 0
      total = 0
      correct_val = 0
      total_val = 0

      with torch.no_grad():
          print("Evaluating on test set...")
          for inputs, labels in tqdm(test_loader):
              valid_indices = labels != -1
              inputs = inputs[valid_indices]
              labels = labels[valid_indices]

              if len(labels) == 0:
                continue

              inputs, labels = inputs.to(device), labels.to(device)
              outputs = resnet_model(inputs)
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
              loss = criterion(outputs, labels)
              running_loss += loss.item()
          print("Evaluating on val set...")
          for inputs, labels in tqdm(val_loader):
              valid_indices = labels != -1
              inputs = inputs[valid_indices]
              labels = labels[valid_indices]

              if len(labels) == 0:
                continue

              inputs, labels = inputs.to(device), labels.to(device)
              outputs = resnet_model(inputs)
              _, predicted = torch.max(outputs.data, 1)
              total_val += labels.size(0)
              correct_val += (predicted == labels).sum().item()
              loss = criterion(outputs, labels)
              running_loss += loss.item()

      test_loss = running_loss / len(test_loader)
      print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {round(test_loss, 5)}")

      test_accuracy = 100 * correct / total
      print(f"Test Accuracy: {round(test_accuracy, 5)}%")

      val_accuracy = 100 * correct_val / total_val
      print(f"Validation Accuracy: {round(val_accuracy, 5)}%")

      epoch_end = time.time()
      elapsed_time = epoch_end - epoch_start

      print(f"Epoch ({epoch+1}/{num_epochs}) time: {elapsed_time}")
      elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

  training_end = time.time()
  elapsed_time = training_end - training_start

  elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
  print(f"Total training time: {elapsed_time}")

def save_model():
   #resnet_model.reset_classifier(0)
   torch.save(resnet_model.state_dict(), "finetuned_model.pth")

def generate_embedding(image_path, model):
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=resnet_model.default_cfg['mean'], std=resnet_model.default_cfg['std'])
  ])

  image = Image.open(image_path)
  input_tensor = transform(image.convert('RGB')).unsqueeze(0)

  #input_tensor = input_tensor.to(device)

  with torch.no_grad():
    embedding = model(input_tensor)

  return embedding.cpu().numpy().flatten()

def cosine_similarity_(embedding1, embedding2):
  embedding1 = np.array(embedding1).reshape(1, -1)
  embedding2 = np.array(embedding2).reshape(1, -1)

  similarity = cosine_similarity(embedding1, embedding2)

  return similarity[0][0]

resnet_model.reset_classifier(0)
resnet_model.load_state_dict(torch.load("finetuned_model.pth", map_location=torch.device('cpu')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model.to(device)

emb1 = generate_embedding('/projects/illinois/eng/ece/kindrtnk/smart_bird_feeder/feeder_birds_dataset/photos/american_goldfinch/american_goldfinch_085.jpg', resnet_model)
emb2 = generate_embedding('/projects/illinois/eng/ece/kindrtnk/smart_bird_feeder/feeder_birds_dataset/photos/american_goldfinch/american_goldfinch_114.jpg', resnet_model)

print(cosine_similarity_(emb1, emb2))