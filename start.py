import os
import argparse
import time
from solver import Solver

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.getcwd())
    parser.add_argument('--ckpt_dir', type=str, default=os.path.join(os.getcwd(), 'checkpoint'))
    parser.add_argument('--ckpt_name', type=str, default='checkpoint')
    parser.add_argument('--model_dir', type=str, default=os.path.join(os.getcwd(), 'model'))
    parser.add_argument('--model_name', type=str, default='1_output.pth')
    parser.add_argument('--training_data_dir', type=str, default=os.path.join(os.getcwd(), 'training_data'))
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--data-scaled', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--decay', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--type', type=str, default='one_class') # one_class, two_classes
    
    # one_class, one_class_mini, two_classes, two_classes_mini
    # parser.add_argument('--type', type=str, default='one_class') 
    parser.add_argument('--type', type=str, default='one_class_mini')
    # parser.add_argument('--type', type=str, default='two_classes') 
    # parser.add_argument('--type', type=str, default='two_classes_mini') 
    args = parser.parse_args()
    
    solver = Solver(args)
    
    t1 = time.time()
    solver.fit()
    print('Training Time: ' + str(time.time() - t1) + ' seconds')
    
    # Save the model
    solver.export()
    
    # Print training data
    solver.print_plot()

    solver.test()