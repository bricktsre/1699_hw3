import torch
import skimage
import numpy as np
import torch.nn as nn
import torchvision as vision
from torch.utils.tensorboard import SummaryWriter
from skimage import io
from skimage.transform import resize
import os
import os.path
import sys

input_size = 32 * 32 * 3
hidden_size = 512
num_classes = 10
num_epochs = 10
batch_size = 1024
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
			img = io.imread(f)
			img = resize(img,(224,224))
			mean = np.array([[[0.406, 0.485, 0.456]]])
			std = np.array([[[0.255, 0.229, 0.224]]])
			img = (img - mean)/std
			return torch.from_numpy(img).permute(2,0,1), target


class MultiLayerPerceptronModel(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super().__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.tanh = nn.Tanh()
		self.fc2 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		out = self.fc1(x.cuda())
		out = self.tanh(out)
		out = self.fc2(out)
		return out

def training(model, dataset_loader, criterion, optimizer):
	model.train()
	running_loss = 0
	running_corrects = 0
	for epoch in range(0,num_epochs):
		for i, (images, labels) in enumerate(dataset_loader):
			images = images.float().to(device)
			labels = labels.to(device)

			outputs = model(images)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
	
			writer.add_scalar('Loss/train mobilenet', loss.item(), epoch*50000 + i)
			_, preds = torch.max(outputs, 1)
			writer.add_scalar('Loss/acurr mobilenet', torch.sum(preds == labels)/float(images.size(0)), epoch*50000 + i)
		#print('Epoch ',epoch,' finished')

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
#for i in range(0,1):
#	for j in range(0,1):
		#print('batch_size: ',batch_size,'hidden_size: ',hidden_size)
#		model = MultiLayerPerceptronModel(input_size, hidden_size, num_classes).to(device)
#		training(model, train_dataloader,nn.CrossEntropyLoss(),torch.optim.RMSprop(model.parameters(), lr=learning_rate))
#		evaluate(model, test_dataloader)
		#hidden_size *= 2
	#hidden_size = 512
model = vision.models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(1280, num_classes)
model.to(device)
training(model, train_dataloader,nn.CrossEntropyLoss(),torch.optim.RMSprop(model.parameters(), lr=learning_rate))
evaluate(model, test_dataloader)
