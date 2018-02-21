from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import Net
from misc import progress_bar

from matplotlib import pyplot as plt


class solver(object):
    def __init__(self, config, training_loader, testing_loader):
        self.model = None
        self.lr = config.lr
        self.mom = config.mom
        self.n_epochs = config.n_epochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.GPU = torch.cuda.is_available()
        self.seed = config.seed
        self.training_loader = training_loader
        self.testing_loader = testing_loader

    def build_model(self):
        def plot(tensor, num_cols=8):
            num_kernels = tensor.shape[0]
            num_rows = 1 + num_kernels // num_cols
            fig = plt.figure(figsize=(num_cols, num_rows))
            for i in range(tensor.shape[0]):
                ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
                ax1.imshow(tensor[i][0])
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig('initial.png')
            plt.show()

        self.model = Net(n_channels=1)
        self.model.weight_init()

        for parameter in self.model.parameters():
            para = parameter.data.numpy()
            break
        plot(para)

        self.criterion = nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.GPU:
            torch.cuda.manual_seed(self.seed)
            self.model.cuda()
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.mom)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay
        print(self.model)

    def save(self):
        model_out_path = "FSRCNN_model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            if self.GPU:
                data, target = Variable(data).cuda(), Variable(target).cuda()
            else:
                data, target = Variable(data), Variable(target)

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            train_loss += loss.data[0]
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def test(self):
        self.model.eval()
        avg_psnr = 0
        for batch_num, (data, target) in enumerate(self.testing_loader):
            if self.GPU:
                data, target = Variable(data).cuda(), Variable(target).cuda()
            else:
                data, target = Variable(data), Variable(target)

            prediction = self.model(data)
            mse = self.criterion(prediction, target)
            # print(mse.data[0])
            psnr = 10 * log10(1 / mse.data[0])
            avg_psnr += psnr
            progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))

    def validate(self):
        self.build_model()
        for epoch in range(1, self.n_epochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step(epoch)
            if epoch == self.n_epochs:
                self.save()
