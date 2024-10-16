import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import time
import torch.nn.functional as F


class GaussianModelAnisotropic:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            #symm = strip_symmetric(actual_covariance)
            #return symm
            # actual_covariance.retain_grad()
            return actual_covariance
        #self._scaling_activation = torch.exp
        self._scaling_activation = torch.sigmoid
        self.scaling_inverse_activation = inverse_sigmoid

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.density_activation = torch.sigmoid
        self.inverse_density_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
    

    def __init__(self):
        self._xyz = torch.empty(0)
        #self._sigma = torch.empty(0) # replace covariance,只需要使用各向同性的表示
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._density = torch.empty(0)
        #self.max_radii2D = torch.empty(0)
        #self.xyz_gradient_accum = torch.empty(0)
        #self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self._xyz,
            #self._sigma,
            self._scaling,
            self._rotation,
            self._density,
            #self.max_radii2D,
            #self.xyz_gradient_accum,
            #self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (
        self._xyz, 
        #self._sigma, 
        self._scaling,
        self._rotation,
        self._density,
        #self.max_radii2D, 
        #xyz_gradient_accum, 
        #denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        #self.xyz_gradient_accum = xyz_gradient_accum
        #self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        #return self._scaling_activation(self._sigma)
        return self._scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz
    #get the number of gaussians
    @property
    def get_gaussians_num(self):
        return self._xyz.shape[0]
    @property
    def get_density(self):
        return self.density_activation(self._density)
    
    @property
    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)
    
    def create_from_fbp(self, fbp_image, air_threshold=0.05, ini_density=0.04, ini_sigma=0.01, spatial_lr_scale=1, num_samples=150000):
        self.spatial_lr_scale = spatial_lr_scale
        # convert shape of fbp_image from [bs,z,x,y] to [bs,z,x,y,1]
        fbp_image = fbp_image.unsqueeze(-1)

        # fbp recon:[bs,z,x,y,1]
        bs, D, H, W,_ = fbp_image.shape
        #
        fbp_image = fbp_image.permute(0, 4, 1, 2, 3)  # [bs,1,z,x,y]
        fbp_image = F.interpolate(fbp_image, size=(H, H, W), mode='trilinear', align_corners=False)
        # 计算每个voxel在3个方向上的梯度
        # fbp_image[fbp_image<air threshold]=0
        # 对fbp图像进行平滑去噪
        # fbp_image F.avg_pool3d(fbp_image,kernel_size=3,stride=1,padding=1)
        # [bs,1,z,x,y]
        grad_x = torch.abs(fbp_image[:, :, 1:-1, 1:-1, 1:-1] - fbp_image[:, :, 1:-1, 1:-1, 2:])
        grad_y = torch.abs(fbp_image[:, :, 1:-1, 1:-1, 1:-1] - fbp_image[:, :, 1:-1, 2:, 1:-1])
        grad_z = torch.abs(fbp_image[:, :, 1:-1, 1:-1, 1:-1] - fbp_image[:, :, 2:, 1:-1, 1:-1])
        # 在每个维度的开始和结束填充©
        grad_x_padded = F.pad(grad_x, (1, 1, 1, 1, 1, 1), "constant", 0)
        grad_y_padded = F.pad(grad_y, (1, 1, 1, 1, 1, 1), "constant", 0)
        grad_z_padded = F.pad(grad_z, (1, 1, 1, 1, 1, 1), "constant", 0)
        # 计算每个voxel梯度的范数
        # [bs,1,z,X,y]
        # print(grad_x.shape,grad_y.shape,grad_z.shape)
        grad_norm = torch.sqrt(grad_x_padded ** 2 + grad_y_padded ** 2 + grad_z_padded ** 2)
        # 按梯度大小进行排序，取出num_.samples个梯度最大的voxel的index
        grad_norm = grad_norm.reshape(-1)
        #TODO:
        # _, indices = torch.topk(grad_norm, 200000 + num_samples)
        # indices = indices[200000:]
        _, indices = torch.topk(grad_norm, num_samples)
        # 取出这些indices对应的3 D coordinates
        coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(H), torch.arange(W)), dim=-1).reshape(-1,3).cuda()
        # 实际上现在的实现应该略微有一点不对应，因为去掉了边界的voxe1
        sampled_coords = coords[indices]
        # Create a 3D grid to count the number of sampled points around each point
        grid = torch.zeros((H, H, W), dtype=torch.int32, device="cuda")
        # Increase the count in the grid at the location of each sampled point
        indices_3d = sampled_coords.long()
        grid[indices_3d[:, 0], indices_3d[:, 1], indices_3d[:, 2]] += 1
        # Apply a 3D convolution to count neighbours
        kernel_size = 5  # Define the size of the neighbourhood
        padding = kernel_size // 2
        conv_kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device="cuda", dtype=torch.float32)
        neighbours_count = F.conv3d(grid.unsqueeze(0).unsqueeze(0).float(), conv_kernel, padding=padding).squeeze()
        # Retrieve the number of neighbours for each sampled point
        num_neighbours = neighbours_count[indices_3d[:, 0], indices_3d[:, 1], indices_3d[:, 2]]
        # Adjust sigmas based on the number of neighbours
        #sigmas = ini_sigma / num_neighbours.float()
        # 设置densities和FBP图像的值成正比
        fbp_image[fbp_image < air_threshold] = 0
        densities = ini_density * fbp_image.reshape(-1)[indices]
        #print("sigma:", sigmas.max(), sigmas.mean(), sigmas.min())  #
        #print("density:", densities.max(), densities.mean(), densities.min())  #
        sampled_coords = sampled_coords.float()
        # 归一化到[0，1]
        sampled_coords = sampled_coords / torch.tensor([H, H, W], dtype=torch.float, device="cuda")
        # print(sampled coords.shape)#[num samples,3]
        #print("Number of points at initialisation: ", num_samples)
        # 对于每个sampled_coord.,它周围的sampled_coordi越多，则它的sigmai越小，否则sigma变大，density都采用一致的值
        densities = inverse_sigmoid(densities).unsqueeze(1)
        #sigmas = inverse_sigmoid(ini_sigma * torch.ones((num_samples, 3), dtype=torch.float, device="cuda"))
        # sigmas = inverse_sigmoid(sigmas).unsqueeze(1)

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(sampled_coords.cpu())).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((num_samples, 4), device="cuda")
        rots[:, 0] = 1
        self._xyz = nn.Parameter(sampled_coords.requires_grad_(True))
        self._density = nn.Parameter(densities.requires_grad_(True))
        # self._sigma = nn.Parameter(sigmas.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

    def create_from_points_cloud(self, pcd, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        #pcd  torch.tensor
        pcd = pcd.float()
        print("Number of points at initialisation: ", pcd.shape[0])

        dist2 = torch.clamp_min(distCUDA2(pcd).float(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((pcd.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        densities = 0.04*inverse_sigmoid(0.1 * torch.ones((pcd.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(pcd.requires_grad_(True))
        self._density = nn.Parameter(densities.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

    def create_from_gaussians(self, pcd, densities, scales, rots, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

        self._xyz = nn.Parameter(pcd.requires_grad_(True))
        self._density = nn.Parameter(densities.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

    def load_ply_new(self, point_array, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        #对point_array进行归一化
        #point_array = (point_array - np.min(point_array, axis=0)) / (np.max(point_array, axis=0) - np.min(point_array, axis=0))
        pcd = torch.tensor(point_array).float().cuda()   #[::2,:]
        print("Number of points at initialisation: ", pcd.shape[0])
        dist2 = torch.clamp_min(distCUDA2(pcd).float(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        densities = 0.04*np.ones(pcd.shape[0], dtype=np.float32)[:, np.newaxis]
        # scales = 0.04*np.ones((pcd.shape[0], 3))
        rots = torch.zeros((pcd.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # densities = -0.004*inverse_sigmoid(0.1 * torch.ones((pcd.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(torch.tensor(pcd, dtype=torch.float, device="cuda").requires_grad_(True))
        self._density = nn.Parameter(torch.tensor(densities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._density], 'lr': training_args.density_lr, "name": "density"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('density')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l


    def reset_density(self):
        density_new = inverse_sigmoid(torch.min(self.get_density, torch.ones_like(self.get_density)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(density_new, "density")
        self._density = optimizable_tensors["density"]


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        #self._sigma = optimizable_tensors["sigma"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        #self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        #self.denom = self.denom[valid_points_mask]
        #self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_densities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "density": new_densities,
        #"sigma": new_sigmas,
        "scaling": new_scaling,
        "rotation": new_rotation
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        #self._sigma = optimizable_tensors["sigma"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,torch.max(self.get_scaling,dim=1).values > self.percent_dense * scene_extent)
        # selected_pts_mask = (torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        print("=================================================")
        print("Spliting {} points".format(selected_pts_mask.sum()))

        if self.get_gaussians_num + selected_pts_mask.sum() < 12000:
            stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            # rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyz = samples + self.get_xyz[selected_pts_mask].repeat(N, 1)

            new_densities = self._density[selected_pts_mask].repeat(N, 1)
            # new_sigma = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
            new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)

            self.densification_postfix(new_xyz, new_densities, new_scaling, new_rotation)

            prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
            self.prune_points(prune_filter)

    def densify_and_split_update(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,torch.max(self.get_scaling,dim=1).values > self.percent_dense * scene_extent)
        # selected_pts_mask = (torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        print("=================================================")
        print("Spliting {} points".format(selected_pts_mask.sum()))

        # if self.get_gaussians_num + selected_pts_mask.sum() < 30000:
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        # rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = samples + self.get_xyz[selected_pts_mask].repeat(N, 1)

        new_densities = self._density[selected_pts_mask].repeat(N, 1)
        # new_sigma = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_densities, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        #selected_pts_mask = (torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        print("=================================================")
        print("Cloning {} points".format(selected_pts_mask.sum()))
        # if self.get_gaussians_num + selected_pts_mask.sum() < 30000:

        new_xyz = self._xyz[selected_pts_mask]

        new_density = self._density[selected_pts_mask]
        #new_sigma = self._sigma[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_density, new_scaling, new_rotation)

    def process_depth_map(self, depth_map):
        # 上采样到128x128的分辨率
        depth_map = torch.nn.functional.interpolate(depth_map, size=(128, 128), mode='bilinear', align_corners=False).squeeze()
        # 将深度值大于0.95的像素设置为0
        depth_map = torch.where(depth_map > 0.95, torch.zeros_like(depth_map), depth_map)
        return depth_map

    def process_gaussian_cloud_torch(self, gaussian_cloud, depth_map):
        # 将高斯点云的坐标转换为64x64深度图的尺度
        coords = (gaussian_cloud[:, :3] * 63).long()
        # 确保坐标在合法范围内
        coords = torch.clamp(coords, 0, 63)
        # 获取对应的深度图像像素值
        pixel_values = depth_map[coords[:, 1], coords[:, 2]]
        # 检查像素值是否为0，或者高斯点云的深度是否小于像素值
        delete_mask = (pixel_values == 0) | (gaussian_cloud[:, 2] < pixel_values / 63.0)  # 归一化回0-1范围
        # delete_list = torch.nonzero(delete_mask).squeeze(1)
        return delete_mask

    def process_gaussian_cloud_torch_Y(self, gaussian_cloud, depth_map):
        # 将高斯点云的坐标转换为64x64深度图的尺度
        coords = (gaussian_cloud[:, :3] * 63).long()
        # 确保坐标在合法范围内
        coords = torch.clamp(coords, 0, 63)
        # 获取对应的深度图像像素值，注意此处与X轴不同，Y轴的深度信息位于不同的索引位置
        pixel_values = depth_map[coords[:, 0], coords[:, 2]]  # 深度图是沿Y轴的，所以使用coords的X和Z作为索引
        # 检查像素值是否为0，或者高斯点云的深度是否小于像素值
        delete_mask = (pixel_values == 0) | (gaussian_cloud[:, 1] < pixel_values / 63.0)  # 归一化回0-1范围，此处比较Y轴的深度
        # delete_list = torch.nonzero(delete_mask).squeeze(1)
        return delete_mask

    # def process_gaussian_cloud_torch(self, gaussian_cloud, depth_map):
    #     # 将高斯点云的坐标转换为128x128深度图的尺度
    #     coords = (gaussian_cloud[:, :3] * 127).long()
    #     # 确保坐标在合法范围内
    #     coords = torch.clamp(coords, 0, 127)
    #     # 将深度图转换为二值图（深度为0的地方设为1，其他设为0）
    #     binary_map = (depth_map == 1).float()
    #     # 执行膨胀操作，使用3x3的核
    #     kernel_size = 3  # 膨胀核大小
    #     padding = kernel_size // 2
    #     dilated_map = F.max_pool2d(binary_map.unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=padding).squeeze()
    #     # 获取膨胀后二值图像对应的深度图像像素值
    #     pixel_values = dilated_map[coords[:, 1], coords[:, 2]]
    #     # 检查膨胀后的二值图中像素是否为1（表示原始或膨胀区域深度为0），或者高斯点云的深度是否小于深度图的像素值
    #     delete_mask = (pixel_values == 0) | (gaussian_cloud[:, 2] < depth_map[coords[:, 1], coords[:, 2]] / 127.0)  # 归一化回0-1范围
    #     # delete_list = torch.nonzero(delete_mask).squeeze(1)
    #     return delete_mask
    # def process_gaussian_cloud_torch_Y(self, gaussian_cloud, depth_map):
    #     # 将高斯点云的坐标转换为128x128深度图的尺度
    #     coords = (gaussian_cloud[:, :3] * 127).long()
    #     # 确保坐标在合法范围内
    #     coords = torch.clamp(coords, 0, 127)
    #     # 将深度图转换为二值图（深度为0的地方设为1，其他设为0）
    #     binary_map = (depth_map == 0).float()
    #     # 执行膨胀操作，使用3x3的核
    #     kernel_size = 3  # 膨胀核大小
    #     padding = kernel_size // 2
    #     dilated_map = F.max_pool2d(binary_map.unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=padding).squeeze()
    #     # 获取膨胀后二值图像对应的深度图像像素值
    #     pixel_values = dilated_map[coords[:, 0], coords[:, 2]]
    #     # 检查像素值是否为0，或者高斯点云的深度是否小于像素值
    #     delete_mask = (pixel_values == 0) | (gaussian_cloud[:, 1] < pixel_values / 127.0)  # 归一化回0-1范围，此处比较Y轴的深度
    #     # delete_list = torch.nonzero(delete_mask).squeeze(1)
    #     return delete_mask
    #TODO: 增加一个高斯操作，根据一组正交的深度图像，对现有的高斯模型进行裁剪和补充
    def densify_based_on_depth_torch(self, depmapX, depmapY):
        # 处理深度图像
        depmapX_processed = self.process_depth_map(depmapX)
        depmapY_processed = self.process_depth_map(depmapY)
        # 处理高斯点云
        delete_list_X = self.process_gaussian_cloud_torch(self.get_xyz, depmapX_processed)
        delete_list_Y = self.process_gaussian_cloud_torch_Y(self.get_xyz, depmapY_processed)
        # list中为true or false 因此可以直接使用逻辑或
        delete_list = delete_list_X | delete_list_Y
        # delete_list = torch.unique(torch.cat((delete_list_X, delete_list_Y)))
        print("Number of gaussians to be pruned: ", delete_list.sum())
        # 删除高斯点云
        if delete_list.size(0) > 0:  # 只有在存在要删除的高斯时才执行
            self.prune_points(delete_list)
        torch.cuda.empty_cache()

    def densify_and_prune(self, max_grad, min_density, sigma_extent, depmapX=None, depmapY=None):
        #grads = self.xyz_gradient_accum / self.denom
        #grads[grads.isnan()] = 0.0
        grads = torch.norm(self._xyz.grad, dim=-1, keepdim=True)
        # 处理深度图像
        # depmapX_processed = self.process_depth_map(depmapX)
        # depmapY_processed = self.process_depth_map(depmapY)

        self.densify_and_clone(grads, max_grad, sigma_extent)
        self.densify_and_split(grads, max_grad, sigma_extent)

        prune_mask = (self.get_density < min_density).squeeze()

        print("Pruning {} points".format(prune_mask.sum()))
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify_and_prune_update(self, max_grad, min_density, sigma_extent, depmapX=None, depmapY=None):
        #grads = self.xyz_gradient_accum / self.denom
        #grads[grads.isnan()] = 0.0
        grads = torch.norm(self._xyz.grad, dim=-1, keepdim=True)
        # 处理深度图像
        # depmapX_processed = self.process_depth_map(depmapX)
        # depmapY_processed = self.process_depth_map(depmapY)

        self.densify_and_clone(grads, max_grad, sigma_extent)
        self.densify_and_split_update(grads, max_grad, sigma_extent)

        prune_mask = (self.get_density < min_density).squeeze()

        print("Pruning {} points".format(prune_mask.sum()))
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def only_prune(self, max_grad, min_density, sigma_extent):
        # grads = self.xyz_gradient_accum / self.denom
        # grads[grads.isnan()] = 0.0
        # grads = torch.norm(self._xyz.grad, dim=-1, keepdim=True)

        prune_mask = (self.get_density < min_density).squeeze()
        print("Pruning {} points".format(prune_mask.sum()))
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    #def add_densification_stats(self, viewspace_point_tensor, update_filter):
    def add_densification_stats(self, viewspace_point_tensor):
        #self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        #self.denom[update_filter] += 1
        self.xyz_gradient_accum += torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1, keepdim=True)
        self.denom += 1
    
    def grid_sample(self, grid, expand):
        # grid: [batchsize, z, x, y, 3]
        grid_shape = grid.shape
        # density_grid = torch.empty(grid_shape, device="cuda")
        # expand dimensions for broadcasting
        grid_expanded = grid.unsqueeze(-2)  # [batchsize, z, x, y, 1, 3]
        #self_xyz_expanded = self._xyz.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 1, num_gaussians, 3]

        # compute density for each point in the grid
        density_grid = self.compute_density(self._xyz, grid_expanded, self.get_density, self.get_covariance, expand)
  
        return density_grid#.unsqueeze(-1)  # [batchsize, z, x, y, num_gaussians, 1]
    
    def compute_density(self, gaussian_centers, grid_point, density, covariance, expand=[5,15,15]):
        # grid_point: [1, z, x, y, 1, 3]
        z,x,y = grid_point.shape[1:4]
        num_gaussians = gaussian_centers.shape[-2]
        # initialize density_grid outside the loop
        density_grid = torch.zeros(1, z, x, y, 1, device='cuda')
        expanded_grid_point = grid_point.expand(num_gaussians,z,x,y,1,3)

        mean_zxy = gaussian_centers * torch.tensor([z, x, y]).cuda() # [num_gaussian, 3]
        mean_z, mean_x, mean_y = mean_zxy[:,0], mean_zxy[:,1], mean_zxy[:,2]
        # 在计算距离时,索引num_gaussians_in_batch个小patch,而不是整个大patch, 与每个gaussian_center分别计算距离
        # pytorch只支持索引统一大小的patch,因此使用固定大小
        z_indices = torch.clamp((mean_z.unsqueeze(-1)-expand[0]/2).int() + torch.arange(0, expand[0], device='cuda'), 0, z-1) #[num_gaussian_in_patch, expand[0]]
        x_indices = torch.clamp((mean_x.unsqueeze(-1)-expand[1]/2).int() + torch.arange(0, expand[1], device='cuda'), 0, x-1) #[num_gaussian_in_patch, expand[1]]
        y_indices = torch.clamp((mean_y.unsqueeze(-1)-expand[2]/2).int() + torch.arange(0, expand[2], device='cuda'), 0, y-1) #[num_gaussian_in_patch, expand[2]]

        grid_indices = torch.arange(num_gaussians, device='cuda').view(-1, 1, 1, 1)
        z_indices = z_indices.view(num_gaussians, -1, 1, 1) # [num_gaussians_in_batch, expand[0], 1, 1]
        x_indices = x_indices.view(num_gaussians, 1, -1, 1) # [num_gaussians_in_batch, 1, expand[1], 1]
        y_indices = y_indices.view(num_gaussians, 1, 1, -1) # [num_gaussians_in_batch, 1, 1, expand[2]]
        patches = expanded_grid_point[grid_indices, z_indices, x_indices, y_indices, :, :] # [num_gaussians_in_batch, expand[0], expand[1], expand[2], 1, 3]
        regularization_term = 1e-6 * torch.eye(3, device='cuda')
        regularized_covariance = covariance + regularization_term
        density_patch = (density.view(-1, 1, 1, 1, 1) * torch.exp(-0.5 * torch.matmul(torch.matmul((patches - gaussian_centers.view(num_gaussians, 1,1,1,1, 3)).unsqueeze(-2), torch.inverse(regularized_covariance.view(num_gaussians, 1,1,1,1, 3, 3))), (patches - gaussian_centers.view(num_gaussians, 1,1,1,1, 3)).unsqueeze(-1)).squeeze(-1).squeeze(-1))) # [num_gaussian_in_patch, expand[0], expand[1], expand[2], 1]
        # Prepare indices for adding the patch back to the density_grid
        indices = ((z_indices * x + x_indices) * y + y_indices).view(-1)
        # Add the density patch back to the density_grid
        density_grid = density_grid.view(-1) # [1*z*x*y*1]
        density_patch = density_patch.view(-1) # [num_gaussians_in_batch*expand[0]*expand[1]*expand[2]*1]
        
        density_grid.scatter_add_(0, indices, density_patch)
        # Reshape density_grid back to its original shape
        density_grid = density_grid.view(1, z, x, y, 1)

        return density_grid
    
    def state_dict(self):
        return {
            '_xyz': self._xyz,
            # '_sigma': self._sigma,
            '_density': self._density,
            '_scaling': self._scaling,
            '_rotation': self._rotation,
            # 添加其他需要保存的参数
        }

    def load_state_dict(self, state_dict):
        self._xyz = state_dict['_xyz']
        # self._sigma = state_dict['_sigma']
        self._density = state_dict['_density']
        self._scaling = state_dict['_scaling']
        self._rotation = state_dict['_rotation']
        # 加载其他参数



