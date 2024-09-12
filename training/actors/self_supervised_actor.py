import os
import torch.nn.functional as F
import torch
import torch.nn as nn

from training.losses.basic_losses import realEPE, real_metrics
from admin.stats import merge_dictionaries
from .base_actor import BaseActor
from training.plot.plot_GLOCALNet import plot_basenet_during_training
from training.plot. plot_sparse_keypoints import plot_sparse_keypoints
from training.plot.plot_GLUNet import plot_during_training_with_uncertainty, plot_sparse_keypoints_GLUNet
from utils_flow.util_optical_flow import flow_to_image


def resize_image(image, factor=32):
    H_32 = image.shape[-2] // factor * factor
    W_32 = image.shape[-1] // factor * factor
    return F.interpolate(image, size=(H_32, W_32), mode='bilinear')


class GLOCALNetActor(BaseActor):
    """Actor for training the GLOCALNet or BaseNet network with a self-supervised or supervised strategy."""
    def __init__(self, net, objective, batch_processing, nbr_images_to_plot=1):
        """
        Args:
            net: The network to train
            objective: The loss function
            batch_processing: A processing class which performs the necessary processing of the batched data.
            nbr_images_to_plot: number of images to plot per epoch
        """
        super().__init__(net, objective, batch_processing)
        self.nbr_images_to_plot = nbr_images_to_plot

    def __call__(self, mini_batch, training):
        """
        Args:
            mini_batch: The mini batch input data, should at least contain the fields 'source_image', 'target_image',
                        'flow_map', 'mask', 'correspondence_mask'.
                        'flow_map' is the ground-truth flow relating the target to the source. 'mask' is the mask
                        where the loss will be applied (in coordinate system of the target).
            training: bool indicating if we are in training or evaluation mode
        returns:
            loss: the training loss
            stats: dict containing detailed losses
        """
        # Run network
        mini_batch = self.batch_processing(mini_batch)
        if 'target_image_256' in mini_batch.keys():
            output_net_256, output_net = self.net(mini_batch['target_image'], mini_batch['source_image'],
                                                  mini_batch['target_image_256'], mini_batch['source_image_256'])
        else:
            output_net = self.net(im_target=mini_batch['target_image'], im_source=mini_batch['source_image'])

        loss, stats = self.objective(output_net, mini_batch['flow_map'], mask=mini_batch['mask'])

        # Log stats
        stats['Loss/total'] = loss.item()

        # Calculates validation stats
        if not training:
            for index_reso in range(len(output_net['flow_estimates'])):
                EPE, PCK_1, PCK_3, PCK_5 = real_metrics(output_net['flow_estimates'][index_reso],
                                                        mini_batch['flow_map'], mini_batch['correspondence_mask'])

                h_, w_ = output_net['flow_estimates'][index_reso].shape[-2:]
                stats['EPE_reso_{}x{}/EPE'.format(h_, w_)] = EPE.item()
                stats['PCK_1_reso_{}x{}/EPE'.format(h_, w_)] = PCK_1.item()
                stats['PCK_3_reso_{}x{}/EPE'.format(h_, w_)] = PCK_3.item()
                stats['PCK_5_reso_{}x{}/EPE'.format(h_, w_)] = PCK_5.item()

            stats['best_value'] = stats['EPE_reso_{}x{}/EPE'.format(mini_batch['flow_map'].shape[-2]//4,
                                                                    mini_batch['flow_map'].shape[-1]//4)]

        # plot images
        epoch = mini_batch['epoch']
        iter = mini_batch['iter']
        if iter < self.nbr_images_to_plot:
            training_or_validation = 'train' if training else 'val'
            base_save_dir = os.path.join(mini_batch['settings'].env.workspace_dir, mini_batch['settings'].project_path,
                                         'plot', training_or_validation)
            if not os.path.isdir(base_save_dir):
                os.makedirs(base_save_dir)

            if mini_batch['sparse'][0]:
                _ = plot_sparse_keypoints(base_save_dir, 'epoch{}_batch{}'.format(epoch, iter), mini_batch,
                                          mini_batch['flow_map'], output_net['flow_estimates'][-1])
            else:
                _ = plot_basenet_during_training(base_save_dir, 'epoch{}_batch{}'.format(epoch, iter),
                                                 mini_batch, mini_batch['flow_map'],
                                                 output_net['flow_estimates'][-1],
                                                 mask_gt=mini_batch['mask'],
                                                 uncertainty_info=
                                                 output_net['uncertainty_estimates'][-1] if
                                                 'uncertainty_estimates' in list(output_net.keys()) else None,
                                                 warping_mask=output_net['warping_mask'] if 'warping_mask' in
                                                              list(output_net.keys()) else None,
                                                 )

        return loss, stats


class GLUNetBasedActor(BaseActor):
    """Actor for training the GLU-Net based networks with a self-supervised or supervised strategy."""

    def __init__(self, net, objective, objective_256, batch_processing, best_val_epe=True, nbr_images_to_plot=1):
        """
        Args:
            net: The network to train
            objective: The loss function to apply to the H-Net
            objective_256: The loss function to apply to the L-Net
            batch_processing: A processing class which performs the necessary processing of the batched data.
            best_val_epe: use AEPE for best value, instead can use PCK
            nbr_images_to_plot: number of images to plot per epoch
        """
        super().__init__(net, objective, batch_processing)

        self.batch_processing = batch_processing
        self.objective_256 = objective_256
        self.nbr_images_to_plot = nbr_images_to_plot
        self.best_val_epe = best_val_epe

    def __call__(self, mini_batch, training):
        """
        args:
            mini_batch: The mini batch input data, should at least contain the fields 'source_image', 'target_image',
                        'flow_map',  'mask', 'source_image_256', 'target_image_256', 'flow_map_256',
                        'mask_256', 'correspondence_mask'

                        'flow_map' is the ground-truth flow relating the target to the source. 'mask' is the mask
                        where the loss will be applied (in coordinate system of the target).
                        Similar for the 256x256 tensors.
            training: bool indicating if we are in training or evaluation mode
        returns:
            loss: the training loss
            stats: dict containing detailed losses
        """
        epoch = mini_batch['epoch']
        iter = mini_batch['iter']

        # Run network
        mini_batch = self.batch_processing(mini_batch)  # also put to GPU there
        output_net_256, output_net_original = self.net(mini_batch['target_image'], mini_batch['source_image'],
                                                       mini_batch['target_image_256'], mini_batch['source_image_256'])

        loss_o, stats_o = self.objective(output_net_original, mini_batch['flow_map'], mask=mini_batch['mask'])
        loss_256, stats_256 = self.objective_256(output_net_256, mini_batch['flow_map_256'], mask=mini_batch['mask_256'])
        loss = loss_o + loss_256

        import ipdb;ipdb.set_trace()

        # Log stats
        stats = merge_dictionaries([stats_o, stats_256])
        stats['Loss_H_Net/total'] = loss_o.item()
        stats['Loss_L_Net/total'] = loss_256.item()
        stats['Loss/total'] = loss.item()

        # Calculates validation stats
        if not training:
            b, _, h_original, w_original = mini_batch['flow_map'].shape
            for index_reso_original in range(len(output_net_original['flow_estimates'])):
                EPE, PCK_1, PCK_3, PCK_5 = real_metrics(output_net_original['flow_estimates'][-(index_reso_original+1)],
                                                        mini_batch['flow_map'], mini_batch['correspondence_mask'])
                h_, w_ = output_net_original['flow_estimates'][-(index_reso_original+1)].shape[-2:]
                stats['EPE_HNet_reso_{}x{}/EPE'.format(h_, w_)] = EPE.item()
                stats['PCK_1_HNet_reso_{}x{}/EPE'.format(h_, w_)] = PCK_1.item()
                stats['PCK_3_HNet_reso_{}x{}/EPE'.format(h_, w_)] = PCK_3.item()
                stats['PCK_5_HNet_reso_{}x{}/EPE'.format(h_, w_)] = PCK_5.item()

            for index_reso_256 in range(len(output_net_256['flow_estimates'])):
                EPE = realEPE(output_net_256['flow_estimates'][-(index_reso_256+1)], mini_batch['flow_map'],
                              mini_batch['correspondence_mask'],
                              ratio_x=float(w_original) / 256.0,
                              ratio_y=float(h_original) / 256.0)
                h_, w_ = output_net_256['flow_estimates'][-(index_reso_256+1)].shape[-2:]
                stats['EPE_LNet_reso_{}x{}/EPE'.format(h_, w_)] = EPE.item()

            if self.best_val_epe:
                stats['best_value'] = stats['EPE_HNet_reso_{}x{}/EPE'.format(h_original//4, w_original//4)]
            else:
                stats['best_value'] = - stats['PCK_1_HNet_reso_{}x{}/EPE'.format(h_original//4, w_original//4)]

        # plot images
        if iter < self.nbr_images_to_plot:
            training_or_validation = 'train' if training else 'val'
            base_save_dir = os.path.join(mini_batch['settings'].env.workspace_dir,
                                         mini_batch['settings'].project_path,
                                         'plot', training_or_validation)
            if not os.path.isdir(base_save_dir):
                os.makedirs(base_save_dir)

            if mini_batch['sparse'][0]:
                _ = plot_sparse_keypoints_GLUNet(base_save_dir, epoch, iter,
                                                 mini_batch['source_image'], mini_batch['target_image'],
                                                 mini_batch['source_image_256'], mini_batch['target_image_256'],
                                                 mini_batch['flow_map'], mini_batch['flow_map_256'],
                                                 output_net_original['flow_estimates'][-1],
                                                 output_net_256['flow_estimates'][-1],
                                                 normalization=True,
                                                 uncertainty_info_original=output_net_original['uncertainty_estimates'][
                                                     -1] if 'uncertainty_estimates' in list(output_net_original.keys()) else None,
                                                 uncertainty_info_256=output_net_256['uncertainty_estimates'][-1]
                                                 if 'uncertainty_estimates' in list(output_net_original.keys()) else None)
            else:
                _ = plot_during_training_with_uncertainty(base_save_dir, epoch, iter,
                                                          mini_batch['source_image'], mini_batch['target_image'],
                                                          mini_batch['source_image_256'],
                                                          mini_batch['target_image_256'],
                                                          mini_batch['flow_map'], mini_batch['flow_map_256'],
                                                          output_net=output_net_original['flow_estimates'][-1],
                                                          output_net_256=output_net_256['flow_estimates'][-1],
                                                          mask=mini_batch['mask'], mask_256=mini_batch['mask_256'],
                                                          uncertainty_info_original=
                                                          output_net_original['uncertainty_estimates'][-1] if
                                                          'uncertainty_estimates' in list(output_net_original.keys()) else None,
                                                          uncertainty_info_256=output_net_256['uncertainty_estimates'][-1]
                                                          if 'uncertainty_estimates' in list(output_net_original.keys()) else None)

        return loss, stats


class CrocoBasedActor(BaseActor):
    """Actor for training the GLU-Net based networks with a self-supervised or supervised strategy."""

    def __init__(self, net, objective, objective_256, batch_processing, best_val_epe=True, nbr_images_to_plot=1, cost_agg=False):
        """
        Args:
            net: The network to train
            objective: The loss function to apply to the H-Net
            objective_256: The loss function to apply to the L-Net
            batch_processing: A processing class which performs the necessary processing of the batched data.
            best_val_epe: use AEPE for best value, instead can use PCK
            nbr_images_to_plot: number of images to plot per epoch
        """
        super().__init__(net, objective, batch_processing)

        self.batch_processing = batch_processing
        self.objective_256 = objective_256
        self.nbr_images_to_plot = nbr_images_to_plot
        self.best_val_epe = best_val_epe
        self.cost_agg = cost_agg
        

    def __call__(self, mini_batch, training):
        """
        args:
            mini_batch: The mini batch input data, should at least contain the fields 'source_image', 'target_image',
                        'flow_map',  'mask', 'source_image_256', 'target_image_256', 'flow_map_256',
                        'mask_256', 'correspondence_mask'

                        'flow_map' is the ground-truth flow relating the target to the source. 'mask' is the mask
                        where the loss will be applied (in coordinate system of the target).
                        Similar for the 256x256 tensors.
            training: bool indicating if we are in training or evaluation mode
        returns:
            loss: the training loss
            stats: dict containing detailed losses
        """
        epoch = mini_batch['epoch']
        iter = mini_batch['iter']

        # Run network
        mini_batch = self.batch_processing(mini_batch)  # also put to GPU there
        
        mini_batch['target_image'] = resize_image(mini_batch['target_image'])
        mini_batch['source_image'] = resize_image(mini_batch['source_image'])
        if self.net.cost_agg == 'CRAFT' and self.net.reciprocity:
            output_net_original, output_net_rev = self.net(mini_batch['target_image'], mini_batch['source_image'])
            output_net_rev = F.interpolate(output_net_rev, size=(mini_batch['flow_map'].shape[-2], mini_batch['flow_map'].shape[-1]), mode='bilinear')
            mask = mini_batch['mask']
            
        elif self.net.occlusion_mask:
            output_net_original, flow_target, flow_source = self.net(mini_batch['target_image'], mini_batch['source_image'])
            mask = mini_batch['mask']
            
            # mask = ((flow_target-flow_source).abs().mean(dim=1) < 5).detach()
            
            # mask_fw, mask_bw = self._forward_backward_occ_check(flow_target, flow_source)
            # if mini_batch['mask'] is None:
            #     mask = mask_fw.detach().squeeze()
            # else:
                # mask = mini_batch['mask'] * mask_fw.detach()

        else:
            output_net_original = self.net(mini_batch['target_image'], mini_batch['source_image'])
            mask = mini_batch['mask']
        
        # H_384, W_320 = 224,224
        # import torchvision
        # resize = torchvision.transforms.Resize((H_384, W_320))
        # import torch
        # from PIL import Image
        # import numpy as np
        # import cv2
        # import matplotlib.pyplot as plt
        # import matplotlib as mpl
        # import matplotlib.cm as cm
        # target_img,source_img = mini_batch['target_image'],mini_batch['source_image']
        # height, width = torch.randint(0, H_384//16, (1,)).item(), torch.randint(0, W_320//16, (1,)).item()
        # in1k_mean, in1k_std = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(source_img.device), torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(source_img.device)
        # flow_img = F.interpolate(output_net_original, size=(mini_batch['flow_map'].shape[-2], mini_batch['flow_map'].shape[-1]), mode='bilinear')
        # flow_img = flow_to_image(flow_img.squeeze().permute(1,2,0).detach().cpu().numpy())
        # flow_img = Image.fromarray(flow_img)
        # flow_img.save('./output_tmp/pred_flow.png')
        
        # flow_img = flow_to_image(mini_batch['flow_map'].squeeze().permute(1,2,0).detach().cpu().numpy())
        # flow_img = Image.fromarray(flow_img)
        # flow_img.save('./output_tmp/gt_flow.png')
        
        # img_tmp = target_img * in1k_std + in1k_mean
        
        # img_tmp[...,height*16:(height+1)*16,width*16:(width+1)*16] = 1
        # img_tmp = img_tmp.squeeze().clone().permute(1,2,0).cpu().numpy()
        # img_input = Image.fromarray((img_tmp*255).astype(np.uint8))
        # img_input.save('./output_tmp/target_input.png')
        
        # attn_maps = self.net.attn_map
        
        # for i, attn_map in enumerate(attn_maps):
        #     grid_x = self.net.grid_x
        #     grid_y = self.net.grid_y
        #     grid_x_margin = ((grid_x+1)*(14-1) / 2.).squeeze()
        #     grid_y_margin = ((grid_y+1)*(14-1) / 2.).squeeze()
        #     attn_map = attn_map.reshape(H_384//16,W_320//16,-1)
        #     attn_map = attn_map[height][width].reshape(H_384//16,W_320//16)       # 24, 32
            
        #     img_tmp = source_img* in1k_std + in1k_mean
        #     img_tmp = img_tmp.squeeze(dim=0).clone().permute(1,2,0).cpu().detach().numpy()
        #     attn_map = torch.zeros_like(attn_map)
        #     attn_map[grid_y_margin[height,width].long(),grid_x_margin[height,width].long()] =1

        #     attn_map = resize(attn_map.unsqueeze(0).unsqueeze(0)).squeeze(0).permute(1,2,0).cpu().detach().numpy()


        #     vmax, vmin = np.percentile(attn_map, 100), attn_map.min()
        #     normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        #     mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')
        #     colormapped_im = (mapper.to_rgba(attn_map[:,:,0])[:, :, :3] * 255).astype(np.uint8)
        #     attn_map = cv2.addWeighted((img_tmp*255).astype(np.uint8), 0.6, colormapped_im, 0.4, 0)
        #     attn_map = Image.fromarray(attn_map)
        #     # attn_map.save(fname+'_attn_map_'+str(j)+'_'+str(k)+'.png')
        #     attn_map.save(f'./output_tmp/{i}_attn_map.png')
        # import ipdb;ipdb.set_trace()
        # output_net_original = F.interpolate(output_net_original, size=(mini_batch['flow_map'].shape[-2], mini_batch['flow_map'].shape[-1]), mode='bilinear')
        # output_net_256, output_net_original = self.net(mini_batch['target_image'], mini_batch['source_image'],
                                                #    mini_batch['target_image_256'], mini_batch['source_image_256'])
        if self.net.occlusion_mask:
            loss_o_target, stats_o = self.objective(flow_target, mini_batch['flow_map'], mask=mask)
            loss_o_source, stats_o = self.objective(flow_source, mini_batch['flow_map'], mask=mask)
            loss_o = loss_o_target + loss_o_source
            output_net_original = flow_target
            # loss_o, stats_o = self.objective(output_net_original, mini_batch['flow_map'], mask=mask)
            
        else:
            
            loss_o, stats_o = self.objective(output_net_original, mini_batch['flow_map'], mask=mask)

            if self.net.cost_agg == 'CRAFT' and self.net.reciprocity:
                loss_rev_o, stats_rev_o = self.objective(output_net_rev, mini_batch['flow_map'], mask=mask)
                loss_o = loss_o + loss_rev_o
        
        
        # loss_256, stats_256 = self.objective_256(output_net_256, mini_batch['flow_map_256'], mask=mini_batch['mask_256'])
        # loss = loss_o + loss_256
        loss = loss_o

        # Log stats
        stats = stats_o
        stats['Loss_H_Net/total'] = loss_o.item()
        stats['Loss/total'] = loss.item()
        if self.net.occlusion_mask:
            stats['Loss_H_Net_target/total'] = loss_o_target.item()
            stats['Loss_H_Net_source/total'] = loss_o_source.item()

        # Calculates validation stats
        if not training:
            b, _, h_original, w_original = mini_batch['flow_map'].shape
            # for index_reso_original in range(len(output_net_original)):
            if isinstance(output_net_original, dict):
                EPE, PCK_1, PCK_3, PCK_5 = real_metrics(output_net_original['flow_estimates'][0],
                                                        mini_batch['flow_map'], mini_batch['correspondence_mask'])
                h_, w_ = output_net_original['flow_estimates'][0].shape[-2:]
            elif isinstance(output_net_original, list):
                EPE, PCK_1, PCK_3, PCK_5 = real_metrics(output_net_original[0],
                                                        mini_batch['flow_map'], mini_batch['correspondence_mask'])
                h_, w_ = output_net_original[0].shape[-2:]#[-(index_reso_original+1)].shape[-2:]
            else:
                EPE, PCK_1, PCK_3, PCK_5 = real_metrics(output_net_original,
                                                        mini_batch['flow_map'], mini_batch['correspondence_mask'])
                h_, w_ = output_net_original.shape[-2:]#[-(index_reso_original+1)].shape[-2:]
            stats['EPE_HNet_reso_{}x{}/EPE'.format(h_, w_)] = EPE.item()
            stats['PCK_1_HNet_reso_{}x{}/EPE'.format(h_, w_)] = PCK_1.item()
            stats['PCK_3_HNet_reso_{}x{}/EPE'.format(h_, w_)] = PCK_3.item()
            stats['PCK_5_HNet_reso_{}x{}/EPE'.format(h_, w_)] = PCK_5.item()


            if self.best_val_epe:
                stats['best_value'] = stats['EPE_HNet_reso_{}x{}/EPE'.format(h_, w_)]
            else:
                stats['best_value'] = - stats['PCK_1_HNet_reso_{}x{}/EPE'.format(h_, w_)]

        # plot images
        if iter < self.nbr_images_to_plot:
            training_or_validation = 'train' if training else 'val'
            base_save_dir = os.path.join(mini_batch['settings'].env.workspace_dir,
                                         mini_batch['settings'].project_path,
                                         'plot', training_or_validation)
            if not os.path.isdir(base_save_dir):
                os.makedirs(base_save_dir)

            if mini_batch['sparse'][0]:
                output_net_256 = output_net_original
                _ = plot_sparse_keypoints_GLUNet(base_save_dir, epoch, iter,
                                                 mini_batch['source_image'], mini_batch['target_image'],
                                                 mini_batch['source_image_256'], mini_batch['target_image'],
                                                 mini_batch['flow_map'], mini_batch['flow_map'],
                                                 output_net_original,
                                                 output_net_256,
                                                 normalization=True,
                                                 uncertainty_info_original=output_net_original['uncertainty_estimates'][
                                                     -1] if 'uncertainty_estimates' in list(output_net_original.keys()) else None,
                                                 uncertainty_info_256=output_net_256['uncertainty_estimates'][-1]
                                                 if 'uncertainty_estimates' in list(output_net_original.keys()) else None)
                
                ## if output_net_original type is dict
                if isinstance(output_net_original, dict):
                    _ = plot_during_training_with_uncertainty(base_save_dir, epoch, iter,
                                            mini_batch['source_image'], mini_batch['target_image'],
                                            mini_batch['source_image'],
                                            mini_batch['target_image'],
                                            mini_batch['flow_map'], mini_batch['flow_map'],
                                            output_net=output_net_original['flow_estimates'][-1],
                                            output_net_256=output_net_original['flow_estimates'][-1],
                                            mask=mini_batch['mask'], mask_256=mini_batch['mask'],
                                            uncertainty_info_original= None,
                                            uncertainty_info_256= None)
                else:
                    _ = plot_during_training_with_uncertainty(base_save_dir, epoch, iter,
                                            mini_batch['source_image'], mini_batch['target_image'],
                                            mini_batch['source_image'],
                                            mini_batch['target_image'],
                                            mini_batch['flow_map'], mini_batch['flow_map'],
                                            output_net=output_net_original,
                                            output_net_256=output_net_original,
                                            mask=mini_batch['mask'], mask_256=mini_batch['mask'],
                                            uncertainty_info_original= None,
                                            uncertainty_info_256= None)

        return loss, stats
    
    def _forward_backward_occ_check(self, flow_fw, flow_bw, scale=1):
        """
        In this function, the parameter alpha needs to be improved
        """

        def length_sq_v0(x):
            return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)

        def length_sq(x):
            temp = torch.sum(x ** 2, dim=1, keepdim=True)
            temp = torch.pow(temp, 0.5)
            return temp

        sum_func = length_sq_v0
        mag_sq = sum_func(flow_fw) + sum_func(flow_bw)
        flow_bw_warped = self.torch_warp(flow_bw, flow_fw)  # torch_warp(img,flow)
        flow_fw_warped = self.torch_warp(flow_fw, flow_bw)
        flow_diff_fw = flow_fw + flow_bw_warped
        flow_diff_bw = flow_bw + flow_fw_warped
        occ_thresh = 1 * mag_sq + 0.5 / scale
        occ_fw = sum_func(flow_diff_fw) < occ_thresh  # 0 means the occlusion region where the photo loss we should ignore
        occ_bw = sum_func(flow_diff_bw) < occ_thresh
        return occ_fw, occ_bw
    
    def torch_warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """

        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
        return output


#### visualize tools

        

        
        # H_384, W_320 = 224,224
        # import torchvision
        # resize = torchvision.transforms.Resize((H_384, W_320))
        # import torch
        # from PIL import Image
        # import numpy as np
        # import cv2
        # import matplotlib.pyplot as plt
        # import matplotlib as mpl
        # import matplotlib.cm as cm
        # target_img,source_img = mini_batch['target_image'],mini_batch['source_image']
        # height, width = torch.randint(0, H_384//16, (1,)).item(), torch.randint(0, W_320//16, (1,)).item()
        # in1k_mean, in1k_std = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(source_img.device), torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(source_img.device)
        
        # flow_img = flow_to_image(output_net_original.squeeze().permute(1,2,0).detach().cpu().numpy())
        # flow_img = Image.fromarray(flow_img)
        # flow_img.save('./output_tmp/pred_flow.png')
        
        # flow_img = flow_to_image(mini_batch['flow_map'].squeeze().permute(1,2,0).detach().cpu().numpy())
        # flow_img = Image.fromarray(flow_img)
        # flow_img.save('./output_tmp/gt_flow.png')
        
        # img_tmp = target_img * in1k_std + in1k_mean
        
        # img_tmp[...,height*16:(height+1)*16,width*16:(width+1)*16] = 1
        # img_tmp = img_tmp.squeeze().clone().permute(1,2,0).cpu().numpy()
        # img_input = Image.fromarray((img_tmp*255).astype(np.uint8))
        # img_input.save('./output_tmp/target_input.png')
        
        # attn_maps = self.net.attn_map
        
        # for i, attn_map in enumerate(attn_maps):
        #     grid_x = self.net.cats.grid_x
        #     grid_y = self.net.cats.grid_y
        #     grid_x_margin = ((grid_x+1)*(14-1) / 2.).squeeze()
        #     grid_y_margin = ((grid_y+1)*(14-1) / 2.).squeeze()
        #     attn_map = attn_map.reshape(H_384//16,W_320//16,-1)
        #     attn_map = attn_map[height][width].reshape(H_384//16,W_320//16)       # 24, 32
            
        #     img_tmp = source_img* in1k_std + in1k_mean
        #     img_tmp = img_tmp.squeeze(dim=0).clone().permute(1,2,0).cpu().detach().numpy()
        #     attn_map = torch.zeros_like(attn_map)
        #     attn_map[grid_y_margin[height,width].long(),grid_x_margin[height,width].long()] =1

        #     attn_map = resize(attn_map.unsqueeze(0).unsqueeze(0)).squeeze(0).permute(1,2,0).cpu().detach().numpy()


        #     vmax, vmin = np.percentile(attn_map, 100), attn_map.min()
        #     normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        #     mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')
        #     colormapped_im = (mapper.to_rgba(attn_map[:,:,0])[:, :, :3] * 255).astype(np.uint8)
        #     attn_map = cv2.addWeighted((img_tmp*255).astype(np.uint8), 0.6, colormapped_im, 0.4, 0)
        #     attn_map = Image.fromarray(attn_map)
        #     # attn_map.save(fname+'_attn_map_'+str(j)+'_'+str(k)+'.png')
        #     attn_map.save(f'./output_tmp/{i}_attn_map.png')
        # import ipdb;ipdb.set_trace()
        # output_net_original = F.interpolate(output_net_original, size=(mini_batch['flow_map'].shape[-2], mini_batch['flow_map'].shape[-1]), mode='bilinear')
        # output_net_256, output_net_original = self.net(mini_batch['target_image'], mini_batch['source_image'],
        #                                                mini_batch['target_image_256'], mini_batch['source_image_256'])