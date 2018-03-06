from math import log10, ceil
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import Net
from misc import progress_bar
from matplotlib import pyplot as plt
from logger import Logger
from matplotlib.colors import Normalize
from dataset.dataset import load_img
from torchvision.transforms import ToTensor
from scipy.misc import imsave
from torchviz import make_dot
from loss import HuberLoss, CharbonnierLoss

class solver(object):
    def __init__(self, config, train_loader, set5_h5_loader, set14_h5_loader, set5_img_loader, set14_img_loader):
        self.model = None
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.mom = config.mom
        self.logs = config.logs
        self.n_epochs = config.n_epochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.GPU = torch.cuda.is_available()
        self.seed = config.seed
        self.train_loader = train_loader
        self.set5_h5_loader = set5_h5_loader
        self.set14_h5_loader = set14_h5_loader
        self.set5_img_loader = set5_img_loader
        self.set14_img_loader = set14_img_loader
        self.logger = Logger(self.logs + '/')
        self.info = {'loss':0, 'PSNR for Set5':0, 'PSNR for Set14':0}
        # 'PSNR for Set5 patch':0, 'PSNR for Set14 patch':0
        self.final_para = []
        self.initial_para = []
        self.graph = True
        self.to_tensor = ToTensor()

        if not os.path.isdir(self.logs):
            os.makedirs(self.logs)

    def build_model(self):
        self.model = Net(n_channels=1)
        self.model.weight_init()
        # self.model = torch.load('./logs/no7/x2/FSRCNN_model100.pth')

        self.criterion = nn.MSELoss()
        # self.criterion = HuberLoss(delta=0.9) # Huber loss
        # self.criterion = CharbonnierLoss(delta=0.0001) # Charbonnier Loss
        torch.manual_seed(self.seed)

        if self.GPU:
            torch.cuda.manual_seed(self.seed)
            self.model.cuda()
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = optim.SGD([{'params': self.model.first_part[0].weight}, # ffeature extraction
                                    {'params': self.model.first_part[0].bias, 'lr': 0.1 * self.lr},
                                    {'params': self.model.mid_part[0][0].weight}, # shrinking
                                    {'params': self.model.mid_part[0][0].bias, 'lr': 0.1 * self.lr},
                                    {'params': self.model.mid_part[1][0].weight}, # mapping
                                    {'params': self.model.mid_part[1][0].bias, 'lr': 0.1 * self.lr},
                                    {'params': self.model.mid_part[2][0].weight},
                                    {'params': self.model.mid_part[2][0].bias, 'lr': 0.1 * self.lr},
                                    {'params': self.model.mid_part[3][0].weight},
                                    {'params': self.model.mid_part[3][0].bias, 'lr': 0.1 * self.lr},
                                    {'params': self.model.mid_part[4][0].weight},
                                    {'params': self.model.mid_part[4][0].bias, 'lr': 0.1 * self.lr},
                                    {'params': self.model.mid_part[5][0].weight},  # expanding
                                    {'params': self.model.mid_part[5][0].bias, 'lr': 0.1 * self.lr},
                                    {'params': self.model.last_part.weight, 'lr': 0.1 * self.lr}, # deconvolution
                                    {'params': self.model.last_part.bias, 'lr': 0.1 * self.lr}],
                                    lr=self.lr, momentum=self.mom)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.mom)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)
        #  lr decay
        print(self.model)

    def save(self, epoch):
        model_out_path = self.logs + '/FSRCNN_model' + str(epoch) + '.pth'
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.train_loader):
            if self.GPU:
                data, target = Variable(data).cuda(), Variable(target).cuda()
            else:
                data, target = Variable(data), Variable(target)

            # if self.graph: # plot the network
            #     graph = make_dot(self.model(data))
            #     graph.view()
            #     self.graph = False

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            train_loss += loss.data[0]
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.train_loader), 'Loss: %.5f' % (train_loss / (batch_num + 1)))

        self.info['loss']= train_loss / len(self.train_loader)

    def test_set5_patch(self):
        self.model.eval()
        avg_psnr = 0
        for batch_num, (data, target) in enumerate(self.set5_h5_loader):
            if self.GPU:
                data, target = Variable(data).cuda(), Variable(target).cuda()
            else:
                data, target = Variable(data), Variable(target)

            prediction = self.model(data)
            mse = self.criterion(prediction, target)
            psnr = 10 * log10(1 / mse.data[0])
            avg_psnr += psnr
            progress_bar(batch_num, len(self.set5_h5_loader), 'PSNR: %.4fdB' % (avg_psnr / (batch_num + 1)))

        self.info['PSNR for Set5 patch'] = avg_psnr / len(self.set5_h5_loader)

    def test_set14_patch(self):
        self.model.eval()
        avg_psnr = 0
        for batch_num, (data, target) in enumerate(self.set14_h5_loader):
            if self.GPU:
                data, target = Variable(data).cuda(), Variable(target).cuda()
            else:
                data, target = Variable(data), Variable(target)

            prediction = self.model(data)
            mse = self.criterion(prediction, target)
            psnr = 10 * log10(1 / mse.data[0])
            avg_psnr += psnr
            progress_bar(batch_num, len(self.set14_h5_loader), 'PSNR: %.4fdB' % (avg_psnr / (batch_num + 1)))

        self.info['PSNR for Set14 patch'] = avg_psnr / len(self.set14_h5_loader)

    def test_set5_img(self):
        self.model.eval()
        avg_psnr = 0
        for batch_num, (data, target) in enumerate(self.set5_img_loader):
            target = target.numpy()
            target = target[:, :, 6:target.shape[2] - 6, 6:target.shape[3] - 6]
            # target = Variable(torch.from_numpy(target))
            if self.GPU:
                data, target = Variable(data).cuda(), Variable(torch.from_numpy(target)).cuda()
            else:
                data, target = Variable(data), Variable(torch.from_numpy(target))

            prediction = self.model(data)
            prediction = prediction.data.cpu().numpy()
            # prediction = prediction[:, :, 6:prediction.shape[2] - 6, 6:prediction.shape[3] - 6]
            if self.GPU:
                prediction = Variable(torch.from_numpy(prediction)).cuda()
            else:
                prediction = Variable(torch.from_numpy(prediction))
            mse = self.criterion(prediction, target)
            psnr = 10 * log10(1 / mse.data[0])
            avg_psnr += psnr
            progress_bar(batch_num, len(self.set5_img_loader), 'PSNR: %.4fdB' % (avg_psnr / (batch_num + 1)))

        self.info['PSNR for Set5'] = avg_psnr / len(self.set5_img_loader)

    def test_set14_img(self):
        self.model.eval()
        avg_psnr = 0
        for batch_num, (data, target) in enumerate(self.set14_img_loader):
            target = target.numpy()
            target = target[:, :, 6:target.shape[2] - 6, 6:target.shape[3] - 6]
            # target = Variable(torch.from_numpy(target))
            if self.GPU:
                data, target = Variable(data).cuda(), Variable(torch.from_numpy(target)).cuda()
            else:
                data, target = Variable(data), Variable(torch.from_numpy(target))

            prediction = self.model(data)
            prediction = prediction.data.cpu().numpy()
            # prediction = prediction[:, :, 6:prediction.shape[2] - 6, 6:prediction.shape[3] - 6]
            if self.GPU:
                prediction = Variable(torch.from_numpy(prediction)).cuda()
            else:
                prediction = Variable(torch.from_numpy(prediction))
            mse = self.criterion(prediction, target)
            psnr = 10 * log10(1 / mse.data[0])
            avg_psnr += psnr
            progress_bar(batch_num, len(self.set14_img_loader), 'PSNR: %.4fdB' % (avg_psnr / (batch_num + 1)))

        self.info['PSNR for Set14'] = avg_psnr / len(self.set14_img_loader)

    def predict(self, epoch):
        self.model.eval()
        butterfly = load_img('./butterfly120.bmp')
        butterfly = torch.unsqueeze(self.to_tensor(butterfly), 0)
        if self.GPU:
            data = Variable(butterfly).cuda()
        else:
            data = Variable(butterfly)
        prediction = self.model(data).data.cpu().numpy()[0][0]
        self.logger.image_summary('prediction', prediction, epoch)
        # imsave(self.logs + '/prediction_' + str(epoch) + '.bmp', prediction)

    def plot_fig(self,  tensor, filename, num_cols=8):
        num_kernels = tensor.shape[0]
        num_rows = ceil(num_kernels / num_cols)
        fig = plt.figure(figsize=(num_cols, num_rows))
        for i in range(tensor.shape[0]):
            ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
            ax1.imshow(tensor[i][0], norm=Normalize())
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(self.logs + '/' + filename + '.png')
        # plt.show()

    def get_para(self):
        para = []
        for parameter in self.model.parameters():
            para.append(parameter.data.cpu().numpy())
        return para

    def validate(self):
        self.build_model()

        self.initial_para = self.get_para()
        for epoch in range(1, self.n_epochs + 1):
            if epoch == 1: # log initial para
                self.logger.histo_summary('initial fisrt layer para', self.initial_para[0], epoch)
                self.logger.histo_summary('initial last layer para', self.initial_para[-2], epoch)
                self.plot_fig(self.initial_para[0], 'first_layer_initial')
                self.plot_fig(self.initial_para[-2], 'last_layer_initial')
            elif (epoch % 5 == 0) or (epoch == self.n_epochs): # log para
                self.logger.histo_summary('fisrt layer para', self.final_para[0] - self.initial_para[0], epoch)
                self.logger.histo_summary('last layer para', self.final_para[-2] - self.initial_para[-2], epoch)
            print("\n===> Epoch {} starts:".format(epoch))

            self.train()

            if (epoch % 2 ==0) and (self.train_loader.batch_size < self.batch_size):
                self.train_loader.batch_size *= 2
                self.train_loader.batch_sampler.batch_size *= 2

            # print('Testing Set5 patch:')
            # self.test_set5_patch()
            # print('Testing Set14 patch:')
            # self.test_set14_patch()
            print('Testing Set5:')
            self.test_set5_img()
            print('Testing Set14:')
            self.test_set14_img()
            # self.scheduler.step(epoch)
            self.final_para = self.get_para()

            for tag, value in self.info.items():
                self.logger.scalar_summary(tag, value, epoch)

            self.predict(epoch)
            if (epoch % 50 == 0) or (epoch == self.n_epochs) or (epoch == 1):
                if epoch != 1:
                    self.save(epoch)
                # plot the para
                self.plot_fig(self.final_para[0] - self.initial_para[0], '/first_diff_' + str(epoch))
                self.plot_fig(self.final_para[-2] - self.initial_para[-2], '/last_diff_' + str(epoch))
                self.plot_fig(self.final_para[0], '/first_' + str(epoch))
                self.plot_fig(self.final_para[-2], '/last_' + str(epoch))

