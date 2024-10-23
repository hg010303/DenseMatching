from termcolor import colored
import torch.optim as optim
import os
import torch
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler


from utils_data.image_transforms import ArrayToTensor
from training.actors.batch_processing import GLUNetBatchPreprocessing
from training.losses.neg_log_likelihood import NLLMixtureLaplace
from training.losses.multiscale_loss import MultiScaleMixtureDensity
from training.trainers.matching_trainer import MatchingTrainer
from utils_data.loaders import Loader
from admin.multigpu import MultiGPU
from training.actors.self_supervised_actor import GLUNetBasedActor, CrocoBasedActor
# from models.PDCNet.PDCNet import PDCNet_vgg16
# from datasets.load_pre_made_datasets.load_pre_made_dataset import PreMadeDataset
# from datasets.object_augmented_dataset import MSCOCO, AugmentedImagePairsDatasetMultipleObjects
from datasets.object_augmented_dataset.synthetic_object_augmentation_for_pairs_multiple_ob import RandomAffine
from datasets.MegaDepth.megadepth import MegaDepthDataset
# from datasets.mixture_of_datasets import MixDatasets
from utils_data.sampler import RandomSampler
from admin.loading import partial_load
from models.croco.croco_downstream import croco_args_from_ckpt
from models.croco.croco import CroCoNet

def run(settings, args=None):
    settings.description = 'Default train settings for PDC-Net+ stage 2 (final)'
    settings.data_mode = 'local'
    settings.batch_size = 10
    settings.n_threads = 8
    settings.multi_gpu = True
    settings.print_interval = 500
    settings.lr = 0.00005
    settings.scheduler_steps = [70, 100, 120]
    settings.n_epochs = 150

    # training parameters
    settings.dataset_callback_fn = 'sample_new_items'  # use to resample image pair at each epoch
    settings.initial_pretrained_model = os.path.join(settings.env.workspace_dir,
                                                     'train_settings/PDCNet/train_PDCNet_plus_stage1/',
                                                     'PDCNetModel_model_best.pth.tar')

    # dataset parameters
    # independently moving objects
    settings.nbr_objects = 4
    settings.min_area_objects = 1300
    settings.compute_object_reprojection_mask = True

    # add independently moving objects + compute the reprojection mask
    source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    target_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    flow_transform = transforms.Compose([ArrayToTensor()])  # just put channels first and put it to float
    co_transform = None

    # 2nd training dataset: MegaDepth data
    source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])

    megadepth_cfg = {'scene_info_path': os.path.join(settings.env.megadepth_training, 'scene_info'),
                     'train_num_per_scene': 300, 'val_num_per_scene': 25,
                     'output_image_size': [520, 520], 'pad_to_same_shape': True,
                     'output_flow_size': [[520, 520], [256, 256]]}
    training_dataset_megadepth = MegaDepthDataset(root=settings.env.megadepth_training, cfg=megadepth_cfg,
                                                  source_image_transform=source_img_transforms,
                                                  target_image_transform=source_img_transforms,
                                                  flow_transform=flow_transform, co_transform=co_transform,
                                                  split='train', store_scene_info_in_memory=False)
    # put store_scene_info_in_memory to True if more than 55GB of cpu memory is available. Sampling will be faster

    train_dataset = training_dataset_megadepth

    # validation data
    megadepth_cfg['exchange_images_with_proba'] = 0.
    val_dataset = MegaDepthDataset(root=settings.env.megadepth_training,
                                   cfg=megadepth_cfg, split='val',
                                   source_image_transform=source_img_transforms,
                                   target_image_transform=source_img_transforms,
                                   flow_transform=flow_transform, co_transform=co_transform,
                                   store_scene_info_in_memory=False)

    # 2. Define dataloaders
    train_loader = Loader('train', train_dataset, batch_size=settings.batch_size,
                          sampler=RandomSampler(train_dataset, num_samples=30000),
                          drop_last=False, training=True, num_workers=settings.n_threads)

    val_loader = Loader('val', val_dataset, batch_size=settings.batch_size, shuffle=False,
                        epoch_interval=1.0, training=False, num_workers=settings.n_threads)

    # 3. Define model
    ckpt = torch.load(settings.env.croco_pretrained_path,'cpu')
    croco_args = croco_args_from_ckpt(ckpt)
    croco_args['img_size'] = ((224//32)*32,(224//32)*32)
    croco_args['args'] = args

    model = CroCoNet(**croco_args)
    model.load_state_dict(ckpt['model'], strict=False)
    
    if args.cost_agg == 'cats':
        for key,value in model.named_parameters():
            if 'cats' not in key:
                value.requires_grad = False
    elif args.cost_agg == 'CRAFT':
        for key,value in model.named_parameters():
            if 'craft' not in key:
                value.requires_grad = False
            
    # but better results are obtained with using simple bilinear interpolation instead of deconvolutions.
    print(colored('==> ', 'blue') + 'model created.')

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model)

    # 4. batch pre-processing module: put all the inputs to cuda as well as in the right format, defines mask
    # used during training, ect
    # since we have sparse ground-truth, we cannot simple downsample the ground-truth flow.
    batch_processing = GLUNetBatchPreprocessing(settings, apply_mask=True, apply_mask_zero_borders=False,
                                                sparse_ground_truth=True, mapping=False)
    # sparse_gt means you can't just downsample the gt flow field at resolution 256x256.
    # it needs to be done before, on the ground truth sparse matches (done in the dataloader)

    # 5. Define Loss module
    objective = NLLMixtureLaplace()
    weights_level_loss = [0.08, 0.08, 0.02, 0.02]
    loss_module_256 = MultiScaleMixtureDensity(level_weights=weights_level_loss[:2], loss_function=objective,
                                               downsample_gt_flow=False)
    loss_module = MultiScaleMixtureDensity(level_weights=[0.32], loss_function=objective,
                                           downsample_gt_flow=False)
    CrocoActor = CrocoBasedActor(model, objective=loss_module, objective_256=loss_module_256,
                                    batch_processing=batch_processing)

    # 6. Define Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=settings.lr, weight_decay=0.0004)

    # 7. Define Scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=settings.scheduler_steps, gamma=0.5)

    # 8. Define trainer
    trainer = MatchingTrainer(CrocoActor, [train_loader, val_loader], optimizer, settings, lr_scheduler=scheduler,
                              make_initial_validation=True)

    trainer.train(settings.n_epochs, load_latest=True, fail_safe=True)




