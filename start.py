import os
import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.getcwd())
    parser.add_argument('--type', type=str, default='convolutional')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--ckpt_dir', type=str, default=os.path.join(os.getcwd(), 'checkpoint'))
    parser.add_argument('--ckpt_name', type=str, default='checkpoint')
    parser.add_argument('--seed', type=int, default=41)
    args = parser.parse_args()
    solver = Solver(args)
    solver.fit()
    acc = solver.test()
    print('Test accuracy: ' + str(acc))
    
if __name__ == '__main__':
    main()