# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

Step 1: Import the required libraries and load the Fashion-MNIST dataset using torchvision.

Step 2: Apply preprocessing techniques such as converting images to tensors and normalizing the data, and create DataLoader for batch processing.

Step 3: Define the Convolutional Neural Network (CNN) architecture with convolution layers, activation functions, pooling layers, and fully connected layers.

Step 4: Train the CNN model using CrossEntropy loss function and Adam optimizer by passing the training data through multiple epochs.

Step 5: Evaluate the trained model on the test dataset to calculate accuracy, confusion matrix, classification report, and perform prediction on a sample image.
## PROGRAM

### Name:YAZHINI R R
### Register Number:212224100063
```
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
```
```
model = CNNClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
```
```
def train_model(model, train_loader, num_epochs=3):

    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print('Name:        ')
        print('Register Number:       ')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

## OUTPUT
### Training Loss per Epoch
<img width="491" height="301" alt="image" src="https://github.com/user-attachments/assets/1e6f8300-722e-4586-87b4-c03bde9a8574" />


### Confusion Matrix
<img width="1025" height="904" alt="image" src="https://github.com/user-attachments/assets/f857e2aa-2a22-4d00-a0c1-31e3d1f5b85e" />

### Classification Report
<img width="584" height="470" alt="image" src="https://github.com/user-attachments/assets/2c0eeee4-0f79-4bda-988e-147765b16b23" />

### New Sample Data Prediction
<img width="725" height="634" alt="image" src="https://github.com/user-attachments/assets/c825c38b-2e65-43b6-a029-785275dd4f65" />


## RESULT
Thus, we successfully developed a Convolutional Deep Neural Network (CNN) for image classification using the Fashion MNIST dataset.
