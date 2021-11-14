from __future__ import print_function
import argparse
import os
import pprint
from math import log10

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data import get_training_set, get_test_set

# Training settings
opt={
    "dataset":'',
    "batch_size":1,
    "test_batch_size":1,
    "direction":'b2a',
    "input_nc":3,
    "output_nc":3,
    "ngf":64,
    "ndf":64,
    "epoch_count":1,
    "niter":100,
    "niter_decay":100,
    "lr":0.0002,
    "lr_policy":'lambda',
    "lr_decay_iters":50,
    "beta1":0.5,
    "cuda":"store_true",
    "threads":4,
    "seed":123,
    "lamb":10
}

prettify = pprint.PrettyPrinter(depth=1)
prettify.pprint(opt)

class pix2pix:

    def __init__(self, wb_project, wb_entity, wb_hp_config):
        self.wb_project = wb_project
        self.wb_entity = wb_entity
        self.wb_hp_config = wb_hp_config
            
    def run(self):
        with wandb.init(project=self.wb_project, entity=self.wb_entity,config=self.wb_hp_config):
            #Use wandb.config result object.property to access hyperparameters so logging hps == execution hps
            opt = wandb.config
            #Set device, seed, benchmark mode 
            if opt.cuda and not torch.cuda.is_available():
                raise Exception("No GPU found, please run without --cuda")
            
            cudnn.benchmark = True

            torch.manual_seed(opt.seed)
            if opt.cuda:
                torch.cuda.manual_seed(opt.seed)
                
            #Download, store, transform data, train and test
            self.prep_data(opt)
            
            #Define model topology
            self.build(opt)
            
            #Train
            self.train(opt)
                    
            
    def prep_data(self,opt):
        print('===> Loading datasets')
        root_path = "dataset/"
        train_set = get_training_set(root_path + opt.dataset, opt.direction)
        test_set = get_test_set(root_path + opt.dataset, opt.direction)
        self.training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
        self.testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)


    def build(self,opt):
        print('===> Building models')
        self.device = torch.device("cuda:0" if opt.cuda else "cpu")
        print("device:",self.device)
        self.net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=self.device)
        self.net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=self.device)

        self.criterionGAN = GANLoss().to(self.device)
        self.criterionL1 = nn.L1Loss().to(self.device)
        self.criterionMSE = nn.MSELoss().to(self.device)

        # setup optimizer
        self.optimizer_g = optim.Adam(self.net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_d = optim.Adam(self.net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.net_g_scheduler = get_scheduler(self.optimizer_g, opt)
        self.net_d_scheduler = get_scheduler(self.optimizer_d, opt)


    def train(self,opt):
        
        wandb.watch(models = (self.net_g,self.net_d),
                    criterion = self.criterionGAN,
                    log = "gradients",
                    log_freq =10,
                    log_graph =True)
        
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            # train
            batch_iter = 0
            for iteration, batch in enumerate(self.training_data_loader, 1):
                # forward
                real_a, real_b = batch[0].to(self.device), batch[1].to(self.device)
                fake_b =self.net_g(real_a)

                ######################
                # (1) Update D network
                ######################

                self.optimizer_d.zero_grad()

                # train with fake
                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake = self.net_d.forward(fake_ab.detach())
                loss_d_fake = self.criterionGAN(pred_fake, False)

                # train with real
                real_ab = torch.cat((real_a, real_b), 1)
                pred_real = self.net_d.forward(real_ab)
                loss_d_real = self.criterionGAN(pred_real, True)

                # Combined D loss
                loss_d = (loss_d_fake + loss_d_real) * 0.5

                loss_d.backward()

                self.optimizer_d.step()

                ######################
                # (2) Update G network
                ######################

                self.optimizer_g.zero_grad()

                # First, G(A) should fake the discriminator
                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake = self.net_d.forward(fake_ab)
                loss_g_gan = self.criterionGAN(pred_fake, True)

                # Second, G(A) = B
                loss_g_l1 = self.criterionL1(fake_b, real_b) * opt.lamb

                loss_g = loss_g_gan + loss_g_l1

                loss_g.backward()

                self.optimizer_g.step()
                
                if batch_iter % 10 == 0:
                    print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                        epoch, iteration, len(self.training_data_loader), loss_d.item(), loss_g.item()))
                    batch_iter += 1
                    wandb.log({"Generator_Loss":loss_g,"Discrimator_Loss":loss_d})
            
            update_learning_rate(self.net_g_scheduler, self.optimizer_g)
            update_learning_rate(self.net_d_scheduler, self.optimizer_d)

            # test
            avg_psnr = 0
            for batch in self.testing_data_loader:
                input, target = batch[0].to(self.device), batch[1].to(self.device)
                prediction = self.net_g(input)
                mse = self.criterionMSE(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
            print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_data_loader)))

            #checkpoint
            if epoch % 1 == 0:
                if not os.path.exists("checkpoint"):
                    os.mkdir("checkpoint")
                if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
                    os.mkdir(os.path.join("checkpoint", opt.dataset))
                net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
                net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
                torch.save(self.net_g.state_dict(), net_g_model_out_path)
                torch.save(self.net_d.state_dict(), net_d_model_out_path)
                print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

