import os
import argparse
import time
from solver import Solver

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.getcwd())
    parser.add_argument('--data-scaled', type=bool, default=True)
    parser.add_argument('--type', type=str, default='convolutional')
    parser.add_argument('--lr', type=float, default=3)
    parser.add_argument('--gamma', type=float, default=0.35)
    parser.add_argument('--decay', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--ckpt_dir', type=str, default=os.path.join(os.getcwd(), 'checkpoint'))
    parser.add_argument('--ckpt_name', type=str, default='checkpoint')
    parser.add_argument('--model-dir', type=str, default=os.path.join(os.getcwd(), 'model', 'model46.pth'))
    parser.add_argument('--seed', type=int, default=103)
    args = parser.parse_args()
    
    t1 = time.time()
    solver = Solver(args)
    solver.fit()
    t2 = time.time()
    print('Training Time: ' + str(t2 - t1) + ' seconds')
    
    # Save the model
    solver.export()
    
    # Print training data
    solver.print_plot()

    acc = solver.test()