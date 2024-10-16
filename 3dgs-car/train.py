# import sys
# import SimpleITK
import numpy as np
import yaml
#from skimage.metrics import structural_similarity as ssim
import os
import json
from sklearn.metrics import mean_squared_error as mse
import torch
# from torch.nn import functional as F

import nibabel as nib
import time
import torch.backends.cudnn as cudnn
from ct_geometry_projector import ConeBeam3DProjector

cudnn.benchmark = True
# 定义一个函数来选择投影
def select_projections(projs, num_projections):
    total_projections = projs.size(1)
    step = total_projections // num_projections
    indices = [i * step for i in range(num_projections)]
    return projs[:,indices, :, :]
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true = y_true > 0
    y_pred = y_pred > 0
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

def psnr(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    return 20 * torch.log10(torch.max(y_true) / torch.sqrt(mse))

def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                            torch.linspace(0, 1, steps=h), \
                                            torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid

def psnr_mask(y_true, y_pred):
    gt_mask = y_true > 0
    y_true = y_true[gt_mask]
    y_pred = y_pred[gt_mask]
    mse = torch.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    max_pixel = 1
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def psnr_mask_projs(y_true, y_pred):
    gt_mask = y_true > 0
    y_true = y_true[gt_mask]
    y_pred = y_pred[gt_mask]
    mse = torch.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    max_pixel = 16
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.dice = []
        self.psnr = []
        self.mse = []

    def update(self, eval_res):
        self.dice.append(eval_res[0])
        self.psnr.append(eval_res[1])
        self.mse.append(eval_res[2])
    def average(self):
        return [np.mean(self.dice), np.mean(self.psnr), np.mean(self.mse)]
    # 将所有的评估指标和平均结果保存到csv文件中



def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)
import copy

from gaussian_model_anisotropic import GaussianModelAnisotropic
def evaluate_gaussian_fbp(dataset,num_proj, save_dir, opt, args):
    algo = 'gaussian_fbp'
    # projs_num = [2,4,8,16]
    num_proj = num_proj
    image_size = [128] * 3
    proj_size = [128] * 3
    ct_projector_train = ConeBeam3DProjector(image_size, proj_size, num_proj)
    ct_projector_new = ConeBeam3DProjector(image_size, proj_size, num_proj, start_angle=5)

    test_list = CCTA_test_list
    dataset_path = CCTADataset_path
    for filename in test_list:
        print("Start to evaluate " + str(filename) + " with " + str(num_proj) + " views")
        best_psnr = 0
        patient = 0
        best_iter=0
        # prepare gaussian model
        opt.density_lr = args.density_lr
        opt.sigma_lr = args.sigma_lr
        # opt.densify_from_iter= 100
        gaussians = GaussianModelAnisotropic()
        # volume data CAS
        data_path = os.path.join(dataset_path, str(filename) + '_volume.pt')
        gt_volume = torch.load(data_path)['volume'][0,:,:,:,:].cuda()
        input_projs = ct_projector_train.forward_project(gt_volume)   #[1, num_proj, x, y]
        # fbp initial gaussian model
        fbp_recon = ct_projector_train.backward_project(input_projs)
        gaussians.create_from_fbp(fbp_recon, air_threshold=0.05, ini_density=0.04, ini_sigma=0.01, spatial_lr_scale=1, num_samples=args.num_init_gaussian)
        gaussians.training_setup(opt)

        mse_loss = torch.nn.functional.mse_loss
        # scaler = GradScaler()
        max_iter = args.max_iter
        starttime = time.time()
        for iteration in range(max_iter):
            # Forward pass
            gaussians.update_learning_rate(iteration)
            # gaussians.optimizer.zero_grad()
            # with autocast():
            # with torch.no_grad():
            grid = create_grid_3d(*image_size)
            grid = grid.cuda()
            # train_data[0] grid: [batchsize, z, x, y, 3]
            grid = grid.unsqueeze(0).repeat(input_projs.shape[0], 1, 1, 1, 1)
            train_output = gaussians.grid_sample(grid, expand=[5, 15, 15])
            del grid
            torch.cuda.empty_cache()
            ##loss
            train_projs = ct_projector_train.forward_project(train_output.transpose(1,4).squeeze(1))

            l1 = mse_loss(train_projs, input_projs)
            # sparsity_loss = 0.000001 * torch.sum(torch.abs(train_output))

            loss = l1
            loss.backward()
            train_psnr = -10 * torch.log10(l1).item()

            # Densification
            if iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 1.5)

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            # if iteration == 0 or (iteration + 1) % 1 == 0:
            if iteration == 0 or (iteration + 1) % 100 == 0:
                # gaussians.eval()
                # with torch.no_grad():
                    # test_psnr = -10 * torch.log10(l1).item()
                if train_psnr > best_psnr:
                    best_psnr = train_psnr
                    patient = 0
                    saved_model = copy.deepcopy(gaussians.state_dict())
                    best_iter = iteration
                    print("best_psnr: ", best_psnr)
                else:
                    patient += 1
                    if patient > 6:
                        print("Early stopping at iteration: ", best_iter, "best_psnr: ", best_psnr)
                        break
        endtime = time.time()
        print("num_views: ", num_proj)
        print("************Training time: s", endtime - starttime)
        gaussians.load_state_dict(saved_model)
        #test
        # gaussians.eval()
        with torch.no_grad():
            grid = create_grid_3d(*image_size)
            grid = grid.cuda()
            # train_data[0] grid: [batchsize, z, x, y, 3]
            grid = grid.unsqueeze(0).repeat(input_projs.shape[0], 1, 1, 1, 1)
            train_output = gaussians.grid_sample(grid, expand=[15, 15, 15])
            #清除缓存释放gpu
            del grid
            torch.cuda.empty_cache()
        # evaluate voxel result
        fbp_recon = train_output.transpose(1,4).squeeze(1).detach()
        # save fbp_recon
        fbp_recon_saved = fbp_recon.squeeze(0).detach().cpu().numpy()
        fbp_recon_saved = nib.Nifti1Image(fbp_recon_saved, np.eye(4))
        nib.save(fbp_recon_saved, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '.nii.gz'))
        #generate new projs
        new_projs_tr = ct_projector_new.forward_project(fbp_recon)
        new_projs_gt = ct_projector_new.forward_project(gt_volume)

        # save new_projs to torch pt file
        torch.save(new_projs_tr, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '-new_projs_train.pt'))
        torch.save(new_projs_gt, os.path.join(save_dir, str(filename) + "-views-" + str(num_proj) + '-new_projs_label.pt'))


if __name__ == "__main__":
    # yaml_path = './configs/default_config.yaml'
    import time
    cpu_num = 10
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    # 加载数据  newdata: 最新的数据集(更新了降采样方法)    new: 之前的数据集

    CCTADataset_path = r"/data/xuemingfu/PublicREPO/3dgs-car"
    CCTA_test_list = ['Normal_1.mha']

    ##TODO: gaussian_fbp
    import sys
    from arguments_init import *
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument('--mydensity_lr', type=float, default=1e-2)
    parser.add_argument('--mysigma_lr', type=float, default=1e-2)
    
    parser.add_argument('--max_iter', type=int, default=8000)
    parser.add_argument('--num_init_gaussian', type=int, default=10000)
    parser.add_argument('--num_proj', type=int, default=16)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    gaussian_fbp_dir = r"/gaussian_fbp_result"
    # dataset = 'CCTA' # 'CAS'
    # evaluate_gaussian_fbp(dataset, args.num_proj, gaussian_fbp_dir, op.extract(args), args)
    for num_proj in [2,4]:
        args.num_proj = num_proj
        dataset = 'CCTA'
        evaluate_gaussian_fbp(dataset, args.num_proj, gaussian_fbp_dir, op.extract(args), args)





