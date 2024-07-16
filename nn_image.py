import torch
from torch import nn
from nn_model import *
import torch.optim as optim
import os
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
device = torch.device("cuda")
# Model instantiation
model = UNet(3, 3).to(device)
# model = LuminanceAdjustmentNet().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loss_list = []
test_loss_list = []
best_loss = 1e5
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    avg_loss = 0
    for batch_idx, (data, target, luminance_changes) in enumerate(train_loader):
        data, target, luminance_changes = data.to(device), target.to(device), luminance_changes.to(device)
        luminance_changes = luminance_changes.unsqueeze(-1)  # Ensure correct shape
        optimizer.zero_grad()
        outputs = model(data, luminance_changes)
        loss = criterion(outputs, target)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:

            print(f'Batch {batch_idx+1}, Loss: {loss.item()}')
    avg_loss /= len(train_loader.dataset)
    train_loss_list.append(avg_loss)
    print('Epoch {}, Train Loss: {:.6f}'.format(epoch, avg_loss))

def test(model, test_loader, criterion, device, epoch):
    model.eval()
    avg_loss = 0
    global best_loss
    for batch_idx, (data, target, luminance_changes) in enumerate(test_loader):
        data, target, luminance_changes = data.to(device), target.to(device), luminance_changes.to(device)
        luminance_changes = luminance_changes.unsqueeze(-1)  # Ensure correct shape
        outputs = model(data, luminance_changes)
        save_fig(outputs, "./result/", batch_idx)
        loss = criterion(outputs, target)
        avg_loss += loss.item()
        loss.backward()
    avg_loss /= len(test_loader.dataset)
    test_loss_list.append(avg_loss)
    print('Epoch {}, Test Loss: {:.6f}'.format(epoch, avg_loss))
    if avg_loss > best_loss:
        torch.save(model.state_dict(),"./model.pth")
        best_loss = avg_loss
        print("======Saving model======")

def read_luminance_from_file(file_path):
    with open(file_path, 'r') as file:
        luminance_values = []
        for i, line in enumerate(file.readlines()):
            if i == 0:
                initial_luminance = float(line)
            else:
                luminance_values.append(np.log10(float(line.split()[0]) / initial_luminance))
    return luminance_values
def split_folder(folder, path):
    img_list = []
    label_list = []
    lum_list = []

    for each in folder:
        ori = sorted(os.listdir(path + '/' + each))
        lum_list.extend(read_luminance_from_file(path + '/' + each + '/' + ori[-1]))
        ori_img = ori[0]
        ori = ori[1:-1]
        # Parts of the list
        first_part = ori[:1]  # 1
        middle_part = ori[1:11]  # 10-19
        second_part = ori[11:12] + ori[13:]  #2-9
        last_part = ori[12:13] #20
        # Re-arranging the list
        ori = first_part + second_part + middle_part + last_part
        for item in ori:
            img_list.append(path + '/' + each + '/' + ori_img)
            label_list.append(path + '/' + each + '/' + item)
    return img_list, label_list, lum_list


def n_fold():
    ori_path = "../../data/new"
    folders = sorted(os.listdir(ori_path))
    random.seed(5)
    random.shuffle(folders)
    test_num = len(folders)//5
    test_folders, train_folders = folders[:test_num], folders[test_num:]
    train_img_list, train_label_list, train_lum_list = split_folder(train_folders, ori_path)
    test_img_list, test_label_list, test_lum_list = split_folder(test_folders, ori_path)

    return train_img_list, train_label_list, train_lum_list, test_img_list, test_label_list, test_lum_list


class UnetDataset(Dataset):
    """
    You need to inherit nn.Module and
    overwrite __getitem__ and __len__ methods.
    """

    def __init__(self, img_list=None, label_list=None, lum_list=None):
        self.img_list = img_list
        self.label_list = label_list
        self.lum_list = lum_list

    def __getitem__(self, index):
        image = Image.open(self.img_list[index])
        image = image.resize((256, 256))
        image = transforms.ToTensor()(image)

        label = Image.open(self.label_list[index])
        label = label.resize((256, 256))
        label = transforms.ToTensor()(label)

        lum = [self.lum_list[index]]
        lum = torch.Tensor(lum)
        lum = lum.float()

        return image, label, lum

    def __len__(self):
        return len(self.img_list)

def build_loader(train_img_list, train_label_list, train_lum_list, test_img_list, test_label_list, test_lum_list):
    train_dataset = UnetDataset(img_list=train_img_list, label_list=train_label_list, lum_list=train_lum_list)
    test_dataset = UnetDataset(img_list=test_img_list, label_list=test_label_list, lum_list=test_lum_list)

    kwargs = {'num_workers': 8, 'pin_memory': False}
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=8, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=8, shuffle=False, **kwargs)

    return train_loader, test_loader

def inter(model, image, lum_ratio): 
    pass 

def save_fig(segOut, path, index):
    segOut = segOut.detach().cpu()
    for i in range(segOut.shape[0]):
        image = transforms.ToPILImage()(segOut[i])
        image.save(path+"{}_{}.png".format(index, i))


def main():
    train_img_list, train_label_list, train_lum_list, test_img_list, test_label_list, test_lum_list = n_fold()
    train_loader, test_loader = build_loader(train_img_list, train_label_list, train_lum_list, test_img_list, test_label_list, test_lum_list)
    epochs = 25
    print("finish data loading")
    for i in range(epochs):
        train(model, train_loader, criterion, optimizer, device, i)
        test(model, test_loader, criterion, device, i)
    print(str(best_loss))
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(test_loss_list, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    if not os.path.exists("./result"):
        os.mkdir("./result")
    plt.savefig("./result/model.png")


if __name__ == "__main__":
    main()
