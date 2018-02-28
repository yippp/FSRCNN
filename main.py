from dataset.data import get_h5_set, get_img_set
from torch.utils.data import DataLoader
import argparse
from solver import solver

parser = argparse.ArgumentParser(description='FSRCNN')
# hyper-parameters
parser.add_argument('--batch_size', type=int, default=128, help='trainingbatch size. Default=128')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for. Default=100')
parser.add_argument('--lr', type=float, default=1e-3,  help='Learning Rate. Default=0.001')
parser.add_argument('--mom', type=float, default=0.9, help='Momentum. Default=0.9')
parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=1')
parser.add_argument('--train_set', type=str, default='train/91-aug.h5', help='name of train set h5 file.')
parser.add_argument('--logs', type=str, default='./logs/no4/huber0.9',
                    help='folder to save the log file. Default=./logs/')

args = parser.parse_args()

def main():
    print('===> Loading datasets')
    train_set = get_h5_set(args.train_set)
    set5_h5 = get_h5_set('test/Set5.h5')
    set14_h5 = get_h5_set('test/Set14.h5')
    set5_img = get_img_set('test/Set5')
    set14_img = get_img_set('test/Set14')
    training_data_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
    Set5_h5_loader = DataLoader(dataset=set5_h5, batch_size=1, shuffle=False)
    Set14_h5_loader = DataLoader(dataset=set14_h5, batch_size=1, shuffle=False)
    Set5_img_loader = DataLoader(dataset=set5_img, batch_size=1, shuffle=False)
    Set14_img_loader = DataLoader(dataset=set14_img, batch_size=1, shuffle=False)


    model = solver(args, training_data_loader, Set5_h5_loader, Set14_h5_loader, Set5_img_loader, Set14_img_loader)
    model.validate()

if __name__ == '__main__':
    main()