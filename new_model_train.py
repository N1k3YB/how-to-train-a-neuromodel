import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, files_path, transform=None):
        self.files_path = files_path
        self.image_files = [os.path.join(files_path, file) for file in os.listdir(files_path)]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(image_file)
        if self.transform:
            image = self.transform(image)
        return image

files_path = r'C:\Users\Administrator\Desktop\code\zalupa\dataset'
target_size = (256, 256)
transform = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])
dataset = CustomDataset(files_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.training = True  # Флаг для обучения/генерации
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Sigmoid()
        )

    def forward(self, z):
        for layer in self.model:
            if isinstance(layer, nn.BatchNorm1d):
                layer.training = self.training
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
    

latent_dim = 100
generator = Generator(latent_dim, img_shape=(3, 256, 256))
optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
criterion = nn.BCELoss()

# Обучение модели
num_epochs = 10
for epoch in range(num_epochs):
    generator.training = True  # Включение режима обучения
    for images in dataloader:
        noise = torch.randn(images.size(0), latent_dim)
        generated_images = generator(noise)
        loss = criterion(generated_images, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Отображение прогресса
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Сохранение модели после обучения
torch.save(generator.state_dict(), 'generator.pth')

# Визуализация последнего сгенерированного изображения
generator.training = False  # Отключение режима обучения
with torch.no_grad():
    sample_noise = torch.randn(1, latent_dim)
    sample_image = generator(sample_noise).detach().cpu().squeeze()
    plt.figure()
    plt.imshow(sample_image.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

#switch_to_gpu
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#for epoch in range(num_epochs):
    #for images in dataloader:
        # Передача тензоров на GPU
        #images = images.to(device)
        # Генерация шума на GPU
        #noise = torch.randn(images.size(0), latent_dim, device=device)
