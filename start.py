import os
import argparse
import time
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.getcwd())
    parser.add_argument('--data-scaled', type=bool, default=True)
    parser.add_argument('--type', type=str, default='convolutional')
    parser.add_argument('--lr', type=float, default=0.1) # usare scheduler per cambiare lr
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--print_every', type=int, default=3)
    parser.add_argument('--ckpt_dir', type=str, default=os.path.join(os.getcwd(), 'checkpoint'))
    parser.add_argument('--ckpt_name', type=str, default='checkpoint')
    parser.add_argument('--model-dir', type=str, default=os.path.join(os.getcwd(), 'model', 'model.pth'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    t1 = time.time()
    solver = Solver(args)
    solver.fit()
    solver.export()
    t2 = time.time()
    print('Training Time: ' + str(t2 - t1) + ' seconds')
    acc = solver.test()
    print('Test accuracy: ' + str(acc))
    
if __name__ == '__main__':
    main()