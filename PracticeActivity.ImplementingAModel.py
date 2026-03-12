#Practice activity: Implementing a model

# Introduction
# In this activity, you will apply the concepts learned in the previous lessons by implementing an ML model from scratch. This hands-on exercise will guide you through the process of building, training, and evaluating a model using your choice of ML frameworks—TensorFlow, PyTorch, or Scikit-learn. The goal is to reinforce your understanding of model implementation. By the end of this activity, you will gain practical experience in deploying a working model.

# By the end of this activity, you will:

# Build an ML model using a specified framework.

# Train the model on a provided dataset.

# Evaluate the model’s performance.

# Save and potentially reload the model for future use.

# Dataset
# You will use the CIFAR-10 dataset, a well-known dataset consisting of 60,000 32x32 color images in 10 different classes. The dataset is already split into 50,000 training images and 10,000 test images. You can download the dataset directly through the TensorFlow or PyTorch libraries, or if using Scikit-learn, you may 
# download it separately
# .

# Step-by-step instructions
# Step 1: Set up your environment
# Choose your framework 
# Decide whether you’ll be using TensorFlow, PyTorch, or Scikit-learn for this lab.

# If you haven’t installed the framework yet, do so now using the following commands:

# TensorFlow example
pip install tensorflow

#PyTorch example
pip install torch torchvision

#Scikit-learn example
pip install scikit-learn



# Set up your development environment
# Open your preferred coding environment (e.g., Jupyter Notebook, VSCode, or any Python IDE).

# Step 2: Import libraries and load the dataset
# Import the required libraries
# Import the necessary libraries for your chosen framework.

# TensorFlow example
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#PyTorch example
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#Scikit-learn example
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Load the CIFAR-10 dataset
#TensorFlow example
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

#PyTorch example
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

#Scikit-learn example
# Note: CIFAR-10 is not available in Scikit-learn by default, but you can use another dataset such as MNIST or download CIFAR-10 separately.
# Example using MNIST instead
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 3: Build your model
# Define your model architecture
# TensorFlow example
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

#PyTorch example
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

#Scikit-learn example
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# Step 4: Train your model
# Set up the training loop
# TensorFlow example
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

#PyTorch example
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

#Scikit-learn example
clf.fit(X_train, y_train)


# Step 5: Evaluate your model
# Evaluate the model’s performance
# TensorFlow example
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

#PyTorch example
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

#Scikit-learn example
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'\nTest accuracy: {accuracy}')


# Step 6: Save and load the model (optional)
# Save the model for future use
# TensorFlow example
model.save('my_cifar10_model.h5')

#PyTorch example
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

#Scikit-learn example
import joblib
joblib.dump(clf, 'random_forest_model.pkl')

#Load the model later (optional)
#TensorFlow example
loaded_model = tf.keras.models.load_model('my_cifar10_model.h5')

#PyTorch example
net = Net()
net.load_state_dict(torch.load(PATH))

#Scikit-learn example
clf = joblib.load('random_forest_model.pkl')

# Deliverable
# By the end of this activity, you should have a completed Jupyter Notebook or Python script that includes all the steps you’ve taken to implement, train, and evaluate the model.

# Optional: Write a brief reflection on any challenges you faced and how you overcame them.

# Conclusion
# This activity provides a practical opportunity to implement an ML model from start to finish. By following these instructions, you’ll gain hands-on experience with your chosen framework and reinforce your understanding of model development and deployment.