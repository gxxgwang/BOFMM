import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.dense_motion import DenseMotionNetwork


class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None,
                 estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
        #########################
        up_blocks2 = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks2.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks2 = nn.ModuleList(up_blocks2)

        self.bottleneck2 = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck2.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final2 = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, driving_image, kp_source, kp_driving):
        # Encoding (downsampling) part
        output_dict = {}
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        output_dict['source_fea'] = out.detach()

        out2 = self.first(driving_image)
        for i in range(len(self.down_blocks)):
            out2 = self.down_blocks[i](out2)
        output_dict['driving_fea'] = out2.detach()
        # Transforming feature representation according to deformation and occlusion

        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']

            output_dict['deformation'] = deformation

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

            out = self.deform_input(out, deformation)
            output_dict["warpfea_StoD"] = out
            output_dict["deformed_StoD"] = self.deform_input(source_image, deformation)####D
            ######################################################################
            dense_motion2 = self.dense_motion_network(source_image=driving_image, kp_driving=kp_source,
                                                      kp_source=kp_driving)
            output_dict['mask2'] = dense_motion2['mask']
            output_dict['sparse_deformed2'] = dense_motion2['sparse_deformed']

            if 'occlusion_map' in dense_motion2:
                occlusion_map2 = dense_motion2['occlusion_map']
                output_dict['occlusion_map2'] = occlusion_map2
            else:
                occlusion_map2 = None

            deformation2 = dense_motion2['deformation']
            output_dict['deformation2'] = deformation2

            out_driving = self.deform_input(output_dict["warpfea_StoD"], deformation2)  # 2 warp
            if occlusion_map2 is not None:
                if out_driving.shape[2] != occlusion_map2.shape[2] or out_driving.shape[3] != occlusion_map2.shape[3]:
                    occlusion_map2 = F.interpolate(occlusion_map2, size=out_driving.shape[2:], mode='bilinear')
                out_driving = out_driving * occlusion_map2

            output_dict["warpfea_StoDtoS"] = out_driving
            output_dict["deformed_StoDtoS"] = self.deform_input(output_dict["deformed_StoD"].detach(), deformation2)###S
            ##########################################################################
            out2 = self.deform_input(out2, deformation2)  # 2 warp

            if occlusion_map2 is not None:
                if out2.shape[2] != occlusion_map2.shape[2] or out2.shape[3] != occlusion_map2.shape[3]:
                    occlusion_map2 = F.interpolate(occlusion_map2, size=out2.shape[2:], mode='bilinear')
                out2 = out2 * occlusion_map2

            output_dict["warpfea_DtoS"] = out2

            output_dict["deformed_DtoS"] = self.deform_input(driving_image, deformation2)###S
            ##############################################################
            out_source = self.deform_input(output_dict["warpfea_DtoS"], deformation)  ###D
            if occlusion_map is not None:
                if out_source.shape[2] != occlusion_map.shape[2] or out_source.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out_source.shape[2:], mode='bilinear')
                out_source = out_source * occlusion_map

            output_dict["warpfea_DtoStoD"] = out_source
            output_dict["deformed_DtoStoD"] = self.deform_input(output_dict["deformed_DtoS"].detach(), deformation)###D

        # Decoding part
        #########S->D
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)
        ###################(S->D)->S
        out_driving = self.bottleneck2(out_driving)
        for i in range(len(self.up_blocks2)):
            out_driving = self.up_blocks2[i](out_driving)
        out_driving = self.final2(out_driving)
        out_driving = F.sigmoid(out_driving)
        #####################
        ###################D->S
        out2 = self.bottleneck(out2)
        for i in range(len(self.up_blocks)):
            out2 = self.up_blocks[i](out2)
        out2 = self.final(out2)
        out2 = F.sigmoid(out2)
        #########(D->S)->D
        out_source = self.bottleneck2(out_source)
        for i in range(len(self.up_blocks2)):
            out_source = self.up_blocks2[i](out_source)
        out_source = self.final2(out_source)
        out_source = F.sigmoid(out_source)

        output_dict["prediction_sd"] = out
        output_dict["prediction_sds"] = out_driving
        output_dict["prediction_ds"] = out2
        output_dict["prediction_dsd"] = out_source

        return output_dict
