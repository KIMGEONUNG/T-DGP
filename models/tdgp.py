import os
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from PIL import Image
from skimage import color

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from torch.autograd import Variable

import models
import utils
from models.downsampler import Downsampler


class T_DGP(object):

    def __init__(self, config):
        self.rank, self.world_size = 0, 1
        if config['dist']:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.config = config
        self.mode = config['dgp_mode']
        self.update_G = config['update_G']
        self.update_embed = config['update_embed']
        self.iterations = config['iterations']
        self.ftr_num = config['ftr_num']
        self.ft_num = config['ft_num']
        self.lr_ratio = config['lr_ratio']
        self.G_lrs = config['G_lrs']
        self.z_lrs = config['z_lrs']
        self.use_in = config['use_in']
        self.select_num = config['select_num']
        self.factor = 2 if self.mode == 'hybrid' else 4  # Downsample factor

        # T_DGP
        self.case_id = config['case_id']

        # create model
        self.G = models.Generator(**config).cuda()
        self.D = models.Discriminator(
            **config).cuda() if config['ftr_type'] == 'Discriminator' else None
        self.G.optim = torch.optim.Adam(
            [{'params': self.G.get_params(i, self.update_embed)}
                for i in range(len(self.G.blocks) + 1)],
            lr=config['G_lr'],
            betas=(config['G_B1'], config['G_B2']),
            weight_decay=0,
            eps=1e-8)

        # load weights
        if config['random_G']:
            self.random_G()
        else:
            utils.load_weights(
                self.G if not (config['use_ema']) else None,
                self.D,
                config['weights_root'],
                name_suffix=config['load_weights'],
                G_ema=self.G if config['use_ema'] else None,
                strict=False)

        self.G.eval()
        if self.D is not None:
            self.D.eval()
        self.G_weight = deepcopy(self.G.state_dict())

        # prepare latent variable and optimizer
        self._prepare_latent()
        # prepare learning rate scheduler
        self.G_scheduler = utils.LRScheduler(self.G.optim, config['warm_up'])
        self.z_scheduler = utils.LRScheduler(self.z_optim, config['warm_up'])

        # loss functions
        self.mse = torch.nn.MSELoss()
        if config['ftr_type'] == 'Discriminator':
            self.ftr_net = self.D
            self.criterion = utils.DiscriminatorLoss(ftr_num=config['ftr_num'][0])
        else:
            vgg = torchvision.models.vgg16(pretrained=True).cuda().eval()
            self.ftr_net = models.subsequence(vgg.features, last_layer='20')
            self.criterion = utils.PerceptLoss()

        # Downsampler for producing low-resolution image
        self.downsampler = Downsampler(
            n_planes=3,
            factor=self.factor,
            kernel_type='lanczos2',
            phase=0.5,
            preserve_size=True).type(torch.cuda.FloatTensor)

    def _prepare_latent(self):
        self.z = torch.zeros((1, self.G.dim_z)).normal_().cuda()
        self.z = Variable(self.z, requires_grad=True)
        self.z_optim = torch.optim.Adam(
            [{'params': self.z, 'lr': self.z_lrs[0]}],
            betas=(self.config['G_B1'], self.config['G_B2']),
            weight_decay=0,
            eps=1e-8
        )
        self.y = torch.zeros(1).long().cuda()

        self.z_b = torch.zeros((1, self.G.dim_z)).normal_().cuda()
        self.z_b = Variable(self.z_b, requires_grad=True)

    def _prepare_latent_f(self):
        self.target_layer = 3

        if self.case_id == 4:
            f_target = self.G.forward_to_f(self.z, self.G.shared(self.y),
                    target_layer=self.target_layer)

            self.feature_mid = f_target

            self.feature_mid = Variable(self.feature_mid, requires_grad=True)
            self.feature_mid_optim = torch.optim.Adam(
                [{'params': self.feature_mid, 'lr': self.z_lrs[0]}],
                betas=(self.config['G_B1'], self.config['G_B2']),
                weight_decay=0,
                eps=1e-8
            )

        if self.case_id == 5:
            f_target = self.G.forward_to_f(self.z, self.G.shared(self.y),
                    target_layer=self.target_layer)

            theta_inv = self._get_inverse_theta(self.theta)
            f_target_t = self._transform_with_theta(f_target, theta_inv)

            f_border = self.G.forward_to_f(self.z_b, self.G.shared(self.y),
                    target_layer=self.target_layer)

            self.feature_mid = self._merge(f_target_t, f_border)

            self.feature_mid = Variable(self.feature_mid, requires_grad=True)
            self.feature_mid_optim = torch.optim.Adam(
                [{'params': self.feature_mid, 'lr': self.z_lrs[0]}],
                betas=(self.config['G_B1'], self.config['G_B2']),
                weight_decay=0,
                eps=1e-8
            )

    def reset_G(self):
        self.G.load_state_dict(self.G_weight, strict=False)
        self.G.reset_in_init()
        if self.config['random_G']:
            self.G.train()
        else:
            self.G.eval()

    def random_G(self):
        self.G.init_weights()

    def _transform_with_theta(self, x, theta):
        from torch.nn.functional import grid_sample, affine_grid
        grid = affine_grid(theta, x.size())
        x_hat = grid_sample(x, grid)
        return x_hat

    def _view_img(self, x):
        from torchvision.transforms import ToPILImage
        if len(x.size()) == 4:
            x = x.squeeze(0)

        x = x.add(x.min()).div(x.max() - x.min())
        img = ToPILImage()(x)
        img.show()

    def SET_TARGET(self, target, category, theta, img_path):
        self.target_origin = target
        self.target_origin_align = self._transform_with_theta(target, theta)

        self.theta = theta
        # apply degradation transform to the original image
        self.target = self.pre_process(target, True)
        self.target_align = self.pre_process(self.target_origin_align, True)

        self.y.fill_(category.item())
        self.img_name = img_path[img_path.rfind('/') + 1:img_path.rfind('.')]

    def _get_inverse_theta(self, theta):
        a = torch.tensor([0., 0., 1.]).to(theta.device)
        a = a.expand(theta.size()[0], 3)
        a = a.view(theta.size()[0], 1, 3)
        a = torch.cat([theta, a], dim=1)
        a = a.inverse()
        a = a[:, :2, :]

        return a

    def _merge(self, target, border):
        mask = target == torch.zeros_like(target)
        border_hat = mask * border
        merged = border_hat + target

        return merged

    def run(self, save_interval=None):

        if self.case_id in [4, 5]:
            self._prepare_latent_f()

        save_imgs = self.target.clone()
        save_imgs2 = save_imgs.cpu().clone()
        loss_dict = {}
        curr_step = 0
        count = 0
        for stage, iteration in enumerate(self.iterations):
            # setup the number of features to use in discriminator
            self.criterion.set_ftr_num(self.ftr_num[stage])

            for i in range(iteration):
                curr_step += 1
                # setup learning rate
                self.G_scheduler.update(curr_step, self.G_lrs[stage],
                                        self.ft_num[stage], self.lr_ratio[stage])
                self.z_scheduler.update(curr_step, self.z_lrs[stage])

                self.z_optim.zero_grad()
                if self.update_G:
                    self.G.optim.zero_grad()

                #################################
                if self.case_id in [1]:
                    x = self.G(self.z, self.G.shared(self.y), use_in=self.use_in[stage])

                if self.case_id in [2]:
                    target_layer = 3
                    f_target = self.G.forward_to_f(self.z, self.G.shared(self.y), 
                            target_layer=target_layer, use_in=self.use_in[stage])

                    theta_inv = self._get_inverse_theta(self.theta)
                    f_target_t = self._transform_with_theta(f_target, theta_inv)

                    x = self.G.forward_from_f(self.z, f_target_t, self.G.shared(self.y), 
                            target_layer=target_layer, use_in=self.use_in[stage])

                if self.case_id in [3]:
                    target_layer = 3
                    f_target = self.G.forward_to_f(self.z, self.G.shared(self.y), 
                            target_layer=target_layer, use_in=self.use_in[stage])

                    theta_inv = self._get_inverse_theta(self.theta)
                    f_target_t = self._transform_with_theta(f_target, theta_inv)

                    f_border = self.G.forward_to_f(self.z_b, self.G.shared(self.y), 
                            target_layer=target_layer, use_in=self.use_in[stage])

                    f = self._merge(f_target_t, f_border)

                    x = self.G.forward_from_f(self.z, f, self.G.shared(self.y), 
                            target_layer=target_layer, use_in=self.use_in[stage])

                if self.case_id in [4, 5]:
                    target_layer = 3
                    self.feature_mid_optim.zero_grad()
                    x = self.G.forward_from_f(self.z, self.feature_mid, self.G.shared(self.y), 
                            target_layer=target_layer, use_in=self.use_in[stage])
                #################################

                # apply degradation transform
                x_map = self.pre_process(x, False)

                # calculate losses in the degradation space
                ftr_loss = self.criterion(self.ftr_net, x_map, self.target)
                mse_loss = self.mse(x_map, self.target)
                # nll corresponds to a negative log-likelihood loss
                nll = self.z**2 / 2
                nll = nll.mean()
                l1_loss = F.l1_loss(x_map, self.target)
                loss = ftr_loss * self.config['w_D_loss'][stage] + \
                    mse_loss * self.config['w_mse'][stage] + \
                    nll * self.config['w_nll']
                loss.backward()

                if self.case_id in [4, 5]:
                    self.feature_mid_optim.step()

                self.z_optim.step()
                if self.update_G:
                    self.G.optim.step()

                # These losses are calculated in the [-1,1] image scale
                # We record the rescaled MSE and L1 loss, corresponding to [0,1] image scale
                loss_dict = {
                    'ftr_loss': ftr_loss,
                    'nll': nll,
                    'mse_loss': mse_loss / 4,
                    'l1_loss': l1_loss / 2
                }

                # calculate losses in the non-degradation space
                if self.mode in ['reconstruct', 'colorization', 'SR', 'inpainting']:
                    # x2 is to get the post-processed result in colorization
                    metrics, x2 = self.get_metrics(x)
                    loss_dict = {**loss_dict, **metrics}

                if i == 0 or (i + 1) % self.config['print_interval'] == 0:
                    if self.rank == 0:
                        print(', '.join(
                            ['Stage: [{0}/{1}]'.format(stage + 1, len(self.iterations))] +
                            ['Iter: [{0}/{1}]'.format(i + 1, iteration)] +
                            ['%s : %+4.4f' % (key, loss_dict[key]) for key in loss_dict]
                        ))
                    # save image sheet of the reconstruction process
                    save_imgs = torch.cat((save_imgs, x), dim=0)
                    torchvision.utils.save_image(
                        save_imgs.float(),
                        '%s/images_sheet/%s_%s.jpg' %
                        (self.config['exp_path'], self.img_name, self.mode),
                        nrow=int(save_imgs.size(0)**0.5),
                        normalize=True)
                    if self.mode == 'colorization':
                        save_imgs2 = torch.cat((save_imgs2, x2), dim=0)
                        torchvision.utils.save_image(
                            save_imgs2.float(),
                            '%s/images_sheet/%s_%s_2.jpg' %
                            (self.config['exp_path'], self.img_name, self.mode),
                            nrow=int(save_imgs.size(0)**0.5),
                            normalize=True)

                if save_interval is not None:
                    if i == 0 or (i + 1) % save_interval[stage] == 0:
                        count += 1
                        save_path = '%s/images/%s' % (self.config['exp_path'],
                                                      self.img_name)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        img_path = os.path.join(
                            save_path, '%s_%03d.jpg' % (self.img_name, count))
                        utils.save_img(x[0], img_path)

                # stop the reconstruction if the loss reaches a threshold
                if mse_loss.item() < self.config['stop_mse'] or ftr_loss.item(
                ) < self.config['stop_ftr']:
                    break

        # save images
        utils.save_img(
            self.target[0], '%s/images/%s_%s_target.png' %
            (self.config['exp_path'], self.img_name, self.mode))
        utils.save_img(
            self.target_origin[0],
            '%s/images/%s_%s_target_origin.png' %
            (self.config['exp_path'], self.img_name, self.mode))
        utils.save_img(
            x[0], '%s/images/%s_%s.png' %
            (self.config['exp_path'], self.img_name, self.mode))
        if self.mode == 'colorization':
            utils.save_img(
                x2[0], '%s/images/%s_%s2.png' %
                (self.config['exp_path'], self.img_name, self.mode))
        if self.config['save_G']:
            torch.save(
                self.G.state_dict(), '%s/G_%s_%s.pth' %
                (self.config['exp_path'], self.img_name, self.mode))
            torch.save(
                self.z, '%s/z_%s_%s.pth' %
                (self.config['exp_path'], self.img_name, self.mode))
        return loss_dict

    def SELECT_Z(self, select_y=False):
        if self.case_id in [3, 5]:
            self.SELECT_Z_BORDER(select_y)
        self.SELECT_Z_TARGET(select_y)

    def SELECT_Z_TARGET(self, select_y=False):
        with torch.no_grad():
            if self.select_num == 0:
                self.z.zero_()
                return
            elif self.select_num == 1:
                self.z.normal_()
                return
            z_all, y_all, loss_all = [], [], []
            if self.rank == 0:
                print('Selecting z from {} samples'.format(self.select_num))
            # only use last 3 discriminator features to compare
            self.criterion.set_ftr_num(3)
            for i in range(self.select_num):
                self.z.normal_(mean=0, std=self.config['sample_std'])
                z_all.append(self.z.cpu())
                if select_y:
                    self.y.random_(0, self.config['n_classes'])
                    y_all.append(self.y.cpu())
                x = self.G(self.z, self.G.shared(self.y))
                x = self.pre_process(x)
                ftr_loss = self.criterion(self.ftr_net, x, self.target_align)
                loss_all.append(ftr_loss.view(1).cpu())
                if self.rank == 0 and (i + 1) % 100 == 0:
                    print('Generating {}th sample'.format(i + 1))
            loss_all = torch.cat(loss_all)
            idx = torch.argmin(loss_all)
            self.z.copy_(z_all[idx])
            if select_y:
                self.y.copy_(y_all[idx])
            self.criterion.set_ftr_num(self.ftr_num[0])

    def SELECT_Z_BORDER(self, select_y=False):
        with torch.no_grad():
            if self.select_num == 0:
                self.z_b.zero_()
                return
            elif self.select_num == 1:
                self.z_b.normal_()
                return
            z_all, y_all, loss_all = [], [], []
            if self.rank == 0:
                print('Selecting z_b from {} samples'.format(self.select_num))
            # only use last 3 discriminator features to compare
            self.criterion.set_ftr_num(3)
            for i in range(self.select_num):
                self.z_b.normal_(mean=0, std=self.config['sample_std'])
                z_all.append(self.z_b.cpu())
                if select_y:
                    self.y.random_(0, self.config['n_classes'])
                    y_all.append(self.y.cpu())
                x = self.G(self.z_b, self.G.shared(self.y))
                x = self.pre_process(x)
                ftr_loss = self.criterion(self.ftr_net, x, self.target)
                loss_all.append(ftr_loss.view(1).cpu())
                if self.rank == 0 and (i + 1) % 100 == 0:
                    print('Generating {}th sample'.format(i + 1))
            loss_all = torch.cat(loss_all)
            idx = torch.argmin(loss_all)
            self.z_b.copy_(z_all[idx])
            if select_y:
                self.y.copy_(y_all[idx])
            self.criterion.set_ftr_num(self.ftr_num[0])

    def pre_process(self, image, target=True):
        n, _, h, w = image.size()
        if self.mode in ['colorization', 'hybrid']:
            # transform the image to gray-scale
            r = image[:, 0, :, :]
            g = image[:, 1, :, :]
            b = image[:, 2, :, :]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            image = gray.view(n, 1, h, w).expand(n, 3, h, w)
        return image

    def get_metrics(self, x):
        with torch.no_grad():
            l1_loss_origin = F.l1_loss(x, self.target_origin) / 2
            mse_loss_origin = self.mse(x, self.target_origin) / 4
            metrics = {
                'l1_loss_origin': l1_loss_origin,
                'mse_loss_origin': mse_loss_origin
            }
            # transfer to numpy array and scale to [0, 1]
            target_np = (self.target_origin.detach().cpu().numpy()[0] + 1) / 2
            x_np = (x.detach().cpu().numpy()[0] + 1) / 2
            target_np = np.transpose(target_np, (1, 2, 0))
            x_np = np.transpose(x_np, (1, 2, 0))
            if self.mode == 'colorization':
                # combine the 'ab' dim of x with the 'L' dim of target image
                x_lab = color.rgb2lab(x_np)
                target_lab = color.rgb2lab(target_np)
                x_lab[:, :, 0] = target_lab[:, :, 0]
                x_np = color.lab2rgb(x_lab)
                x = torch.Tensor(np.transpose(x_np, (2, 0, 1))) * 2 - 1
                x = x.unsqueeze(0)

            ssim = compare_ssim(target_np, x_np, multichannel=True)
            psnr = compare_psnr(target_np, x_np)
            metrics['psnr'] = torch.Tensor([psnr]).cuda()
            metrics['ssim'] = torch.Tensor([ssim]).cuda()
            return metrics, x
