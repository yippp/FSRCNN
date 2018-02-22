from dataset.data import get_training_set, get_test_set
from torch.utils.data import DataLoader
import argparse
from solver import solver

parser = argparse.ArgumentParser(description='FSRCNN')
# hyper-parameters
parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=10e-3, help='Learning Rate. Default=0.001')
parser.add_argument('--mom', type=float, default=0.9, help='Momentum. Default=0.9')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--train_set', type=str, default='91-images', help='name of train set folder')
parser.add_argument('--test_set', type=str, default='5cut', help='name of test set folder')

args = parser.parse_args()

def main():

    print('===> Loading datasets')
    train_set = get_training_set(args.train_set)
    test_set = get_test_set(args.test_set)
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

    model = solver(args, training_data_loader, testing_data_loader)
    model.validate()

if __name__ == '__main__':
    main()