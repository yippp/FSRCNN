from dataset.data import get_training_set, get_test_set
from torch.utils.data import DataLoader
import argparse
from solver import solver

parser = argparse.ArgumentParser(description='FSRCNN')
# hyper-parameters
parser.add_argument('--batch_size', type=int, default=128, help='training batch size. Default=128')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for. Default=100')
parser.add_argument('--lr', type=float, default=1e-3,  help='Learning Rate. Default=0.001')
parser.add_argument('--mom', type=float, default=0.9, help='Momentum. Default=0.9')
parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=1')
parser.add_argument('--train_set', type=str, default='191-images-residual', help='name of train set folder.')
parser.add_argument('--test_set5', type=str, default='Set5', help='name of test Set5 folder')
parser.add_argument('--test_set14', type=str, default='Set14', help='name of test Set14 folder')
parser.add_argument('--logs', type=str, default='./logs/no123/56-12-4-sqrt-d191-res',
                    help='folder to save the log file. Default=./logs/')

args = parser.parse_args()

def main():

    print('===> Loading datasets')
    train_set = get_training_set(args.train_set)
    test_set5 = get_test_set(args.test_set5)
    test_set14 = get_test_set(args.test_set14)
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    testing5_data_loader = DataLoader(dataset=test_set5, batch_size=1, shuffle=False)
    testing14_data_loader = DataLoader(dataset=test_set14, batch_size=1, shuffle=False)

    model = solver(args, training_data_loader, testing5_data_loader, testing14_data_loader)
    model.validate()

if __name__ == '__main__':
    main()