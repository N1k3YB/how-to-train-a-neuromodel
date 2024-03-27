import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, QFrame, QLabel, QPushButton, QLineEdit, QFileDialog, QScrollArea, QProgressBar
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QSizePolicy


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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Добавление картинок для обучения и генерация")
        self.setGeometry(100, 100, 800, 600)

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout()
        self.centralWidget.setLayout(self.layout)

        # Создание вкладок
        self.tabWidget = QTabWidget()
        self.layout.addWidget(self.tabWidget)

        # Вкладка для загрузки картинок
        self.loadImageFrame = QFrame()
        self.loadImageLayout = QVBoxLayout()
        self.loadImageFrame.setLayout(self.loadImageLayout)
        self.imageLabel = QLabel()
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setWidgetResizable(True)
        self.loadImageLayout.addWidget(self.scrollArea)
        self.loadButton = QPushButton("Загрузить изображения")
        self.loadButton.clicked.connect(self.load_image)
        self.loadImageLayout.addWidget(self.loadButton)
        self.trainButton = QPushButton("Обучить модель")
        self.trainButton.clicked.connect(self.train_model)
        self.loadImageLayout.addWidget(self.trainButton)
        self.progressBar = QProgressBar()
        self.loadImageLayout.addWidget(self.progressBar)
        self.tabWidget.addTab(self.loadImageFrame, "Загрузка картинок")

        # Вкладка для генерации картинок
        self.generateImageFrame = QFrame()
        self.generateImageLayout = QVBoxLayout()
        self.generateImageFrame.setLayout(self.generateImageLayout)
        self.generateImageLabel = QLabel()
        self.generateImageLabel.setAlignment(Qt.AlignCenter)
        self.generateImageLayout.addWidget(self.generateImageLabel)
        self.promptEdit = QLineEdit()
        self.generateImageLayout.addWidget(self.promptEdit)
        self.generateButton = QPushButton("Генерировать картинку")
        self.generateButton.clicked.connect(self.generate_image)
        self.generateImageLayout.addWidget(self.generateButton)
        self.tabWidget.addTab(self.generateImageFrame, "Генерация картинок")

        self.image_paths = []
        self.generator = None
        self.dataloader = None

    def load_image(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Image files (*.jpg *.png)")
        if file_dialog.exec_():
            self.image_paths = file_dialog.selectedFiles()
            if self.image_paths:
                pixmap = QPixmap(self.image_paths[0])
                scaled_pixmap = pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio)
                self.imageLabel.setPixmap(scaled_pixmap)

    def train_model(self):
        if self.image_paths:
            files_path = os.path.dirname(self.image_paths[0])
            target_size = (256, 256)
            transform = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])
            dataset = CustomDataset(files_path, transform=transform)
            self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            latent_dim = 100
            img_shape = (3, 256, 256)
            self.generator = Generator(latent_dim, img_shape)
            optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
            criterion = nn.BCELoss()

            num_epochs = 100
            self.progressBar.setMaximum(num_epochs)
            for epoch in range(num_epochs):
                self.progressBar.setValue(epoch + 1)
                self.generator.training = True  # Включение режима обучения
                for images in self.dataloader:
                    noise = torch.randn(images.size(0), latent_dim)
                    generated_images = self.generator(noise)
                    loss = criterion(generated_images, images)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            torch.save(self.generator.state_dict(), 'generator.pth')
            print("Модель обучена и сохранена")

            # Отображение последней сгенерированной картинки
            self.generator.training = False  # Отключение режима обучения
            with torch.no_grad():
                sample_noise = torch.randn(1, latent_dim)
                sample_image = self.generator(sample_noise).detach().cpu().squeeze()

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(sample_image.permute(1, 2, 0))
                ax.axis('off')
                canvas = FigureCanvas(fig)
                canvas.setSizePolicy(
                    QSizePolicy.Expanding,
                    QSizePolicy.Expanding
                )
                canvas.updateGeometry()
                self.scrollArea.takeWidget()
                self.scrollArea.setWidget(canvas)

    def generate_image(self):
        if self.generator is None:
            # Загрузка сохраненной модели
            latent_dim = 100
            img_shape = (3, 256, 256)
            self.generator = Generator(latent_dim, img_shape)
            self.generator.load_state_dict(torch.load('generator.pth'))

        self.generator.training = False  # Отключение режима обучения
        with torch.no_grad():
            sample_noise = torch.randn(1, self.generator.latent_dim)
            sample_image = self.generator(sample_noise).detach().cpu().squeeze()

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(sample_image.permute(1, 2, 0))
            ax.axis('off')
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Expanding
            )
            canvas.updateGeometry()
            self.generateImageLayout.addWidget(canvas, alignment=Qt.AlignCenter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        pixmap = self.imageLabel.pixmap()
        if pixmap is not None and not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio)
            self.imageLabel.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())