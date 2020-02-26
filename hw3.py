import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import os.path
import sys

input_size = 32 * 32 * 3
hidden_size = 512
num_classes = 10
num_epochs = 18
batch_size = 512
learning_rate = 0.001
device = 'cuda:0'
writer = SummaryWriter()

class CifarDataset(torch.utils.data.Dataset):
	def __init__(self, root_dir):
		super().__init__()
		classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]
		classes.sort()
		class_index = {classes[i]: i for i in range(len(classes))}

		images = []
		dir = os.path.expanduser(root_dir)
		for target in sorted(class_index.keys()):
			d = os.path.join(dir, target)
			for root, _, fnames in sorted(os.walk(d)):
				for fname in sorted(fnames):
					path = os.path.join(root, fname)
					item = (path, class_index[target])
					images.append(item)
		self.samples = images

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		path, target = self.samples[index]
		with open(path, 'rb') as f:
			img = plt.imread(f)
			return torch.from_numpy(img), target


class MultiLayerPerceptronModel(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super().__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.tanh = nn.Tanh()
		self.fc2 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		out = self.fc1(x.reshape(-1, 32 * 32 * 3).cuda())
		out = self.tanh(out)
		out = self.fc2(out)
		return out

def training(model, dataset_loader, criterion, optimizer):
	model.train()
	for epoch in range(0,num_epochs):
		for i, (images, labels) in enumerate(dataset_loader):
			images = images.to(device)
			labels = labels.to(device)

			outputs = model(images)
			loss = criterion(outputs, labels)
			writer.add_scalar('Loss/train-'+str(batch_size)+'-'+str(hidden_size), loss, (epoch*50000)+i)

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		print('Epoch ',epoch,' finished')

def evaluate(model, dataset_loader):
	model.eval()
	with torch.no_grad():
		correct = 0
		total = 0
		for i, (images, labels) in enumerate(dataset_loader):
			images = images.to(device)
			labels = labels.to(device)
			output = model(images)
			_, predicted = torch.max(output.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


train_dataset = CifarDataset('cifar10_train')
test_dataset = CifarDataset('cifar10_test')
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
model = MultiLayerPerceptronModel(input_size, hidden_size, num_classes).to(device)
training(model, train_dataloader,nn.CrossEntropyLoss(),torch.optim.Adam(model.parameters(), lr=learning_rate))
evaluate(model, test_dataloader)
	
