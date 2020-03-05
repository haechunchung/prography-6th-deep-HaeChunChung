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
