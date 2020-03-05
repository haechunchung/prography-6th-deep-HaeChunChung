# prography-6th-deep-HaeChunChung

****

## 문제

![문제](https://user-images.githubusercontent.com/46519866/75966954-71b3a180-5f0e-11ea-82ad-6050b2b05c8b.png)

****

## VGG16 순서도

![VGG16순서도](https://user-images.githubusercontent.com/46519866/75961526-86d80280-5f05-11ea-9585-4dc747d954d1.png)

****

## train.py
```Python
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

epochs = 5
batch_size = 1000

mnist_train = torchvision.datasets.MNIST(root="MNIST_data/", train=True, transform=torchvision.transforms.ToTensor(),
                                         download=True)
class Preprocess(Dataset):
    def __init__(self):
        self.x_data = []
        self.y_data = []
        for i, (data, label) in enumerate(mnist_train):
            self.x_data.append((torch.cat((data, data, data), 0)))
            self.y_data.append(label)

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.x_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

dataset = Preprocess()

train_set = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

class VGG_16(nn.Module):
    def __init__(self):
        super(VGG_16, self).__init__()

        self.seq1 = nn.Sequential(
            # 3*28*28 -> 64*14*14
            nn.Conv2d(3, 64, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )

        self.residual = nn.Sequential(
            # 64*14*14 -> 32*4*4
            nn.Conv2d(64, 32, 1, stride=1, padding=0),nn.LeakyReLU(0.2),
            nn.AvgPool2d(5, stride=3, padding=0)
        )

        self.seq2 = nn.Sequential(
            # 64*14*14 -> 128*7*7
            nn.Conv2d(64, 128, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            # 128*7*7 -> 256*4*4
            # stride를 2로하여 사이즈를 7에서 4로 조절하고, pooling을 제거하였다.
            nn.Conv2d(128, 256, 3, stride=2, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),nn.LeakyReLU(0.2),

            # 256*4*4 -> 512*2*2
            nn.Conv2d(256, 512, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            # 512*2*2 -> 512*1*1
            nn.Conv2d(512, 512, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.classifier = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.seq1(x)
        residual = self.residual(x)
        x = self.seq2(x)

        x = x.view(-1, 512)
        residual = residual.view(-1, 512)
        x += residual # Skip Connection

        x = self.fc1(residual)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.classifier(x)
        return x

model = VGG_16().to(DEVICE)
criterion = nn.CrossEntropyLoss() # 내부적으로 Softmax를 포함하고 있다.
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, optimizer, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


for epoch in range(1, epochs + 1):
    train(model, train_set, optimizer, epoch)

PATH = '/Users/haechun/Desktop/weight.pth'
torch.save(model.state_dict(), PATH)
```

****

## 학습 결과

* 학습시 epochs=5, batch_size=1000, learning_rate=0.001로 설정하였다.

![학습](https://user-images.githubusercontent.com/46519866/75965742-927af780-5f0c-11ea-9a32-3be6372b1c2d.png)

****

## test.py

```Python
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

batch_size = 1000

mnist_test = torchvision.datasets.MNIST(root="MNIST_data/", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)

# MNIST 데이터 1*24*24를 3*24*24로 변환
class Preprocess(Dataset):
    def __init__(self):
        self.x_data = []
        self.y_data = []
        for i, (data, label) in enumerate(mnist_test):
            self.x_data.append((torch.cat((data, data, data), 0)))
            self.y_data.append(label)

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.x_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

dataset = Preprocess()

test_set = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

class VGG_16(nn.Module):
    def __init__(self):
        super(VGG_16, self).__init__()

        self.seq1 = nn.Sequential(
            # 3*28*28 -> 64*14*14
            nn.Conv2d(3, 64, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )

        self.residual = nn.Sequential(
            # 64*14*14 -> 32*4*4
            nn.Conv2d(64, 32, 1, stride=1, padding=0),nn.LeakyReLU(0.2),
            nn.AvgPool2d(5, stride=3, padding=0)
        )

        self.seq2 = nn.Sequential(
            # 64*14*14 -> 128*7*7
            nn.Conv2d(64, 128, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            # 128*7*7 -> 256*4*4
            # stride를 2로하여 사이즈를 7에서 4로 조절하고, pooling을 제거하였다.
            nn.Conv2d(128, 256, 3, stride=2, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),nn.LeakyReLU(0.2),

            # 256*4*4 -> 512*2*2
            nn.Conv2d(256, 512, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            # 512*2*2 -> 512*1*1
            nn.Conv2d(512, 512, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.classifier = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.seq1(x)
        residual = self.residual(x)
        x = self.seq2(x)

        x = x.view(-1, 512)
        residual = residual.view(-1, 512)
        x += residual

        x = self.fc1(residual)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.classifier(x)
        return x

model = VGG_16().to(DEVICE)
criterion = nn.CrossEntropyLoss() # 내부적으로 Softmax를 포함하고 있다.
optimizer = optim.Adam(model.parameters(), lr=0.00002)

def inference(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 배치 오차를 합산
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            # 가장 높은 값을 가진 인덱스가 예측값
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


PATH = '/Users/haechun/Desktop/weight.pth'
model.load_state_dict(torch.load(PATH))

test_loss, test_accuracy = inference(model, test_set)

print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_accuracy))
```

****

## 코드 설명
1. MNIST 데이터를 dataset형태로 받아온다.
2. Preprocess 클래스에서 Dataset을 상속받아 오버라이드하여 MNIST 데이터를 전처리해준다.(1x24x24 -> 3x24x24)
3. 위의 순서도에 따라 VGG16을 진행한다.
<br> 3-1) Conv 3-1에서 W, H의 사이즈를 조절하기 위해, 첫 번째 Convolution을 3x3 Conv(stride=2, padding=1)로 설정하였다.
<br> 3-2) Skip Connection 구현 부분
<br> - Conv 2-1의 입력인 64x14x14를 1x1 Conv(stride=1, padding=0)을 통과시켜 32x14x14로 변환.
<br> - 32x14x14를 5x5 AvgPooling(stride=3, padding=0)을 통과시켜 32x4x4로 변환.
4. 나머지는 기존 VGG16과 동일

****

## 테스트 결과

![결과](https://user-images.githubusercontent.com/46519866/75965762-9870d880-5f0c-11ea-9e36-f368d6d8d49c.png)
