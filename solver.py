import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from net import Net
from dataset import Dataset

class Solver():
    def __init__(self, args):
        # prepare a datasets
        self.train_data = Dataset(args.scale, train=True,
                                  data_root=args.data_root,
                                  size=args.image_size)
        self.test_data  = Dataset(args.scale, train=False,
                                  data_root=args.data_root,
                                  size=args.image_size)
        #TODO initialize loss function, optimizer, network, data loader
        
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        
    def fit(self):
        #TODO training cycle
        

    def evaluate(self, global_step):
        args = self.args
        loader = DataLoader(self.test_data,
                            batch_size=args.batch_size,
                            num_workers=1,
                            shuffle=False, drop_last=False)

        self.net.eval()
        mean_psnr = 0

        with torch.no_grad():
            for step, inputs in enumerate(loader):
                image_hr = inputs[0].to(self.device)
                image_lr = inputs[1].to(self.device)
                
                image_sr = self.net(image_lr)

                for hr, sr in zip(image_hr, image_sr):
                    hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).data.numpy()
                    sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).data.numpy()
                    
                    mean_psnr += psnr(hr, sr) / len(self.test_data)
        
        return mean_psnr

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)


def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)

    mse = np.mean((im1-im2)**2)
    return 20 * np.log10(1.0 / np.sqrt(mse))