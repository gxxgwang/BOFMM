import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
# from modules.dense_motion import DenseMotionNetwork


class InpaintingNetwork(nn.Module):
    """
    Inpaint the missing regions and reconstruct the Driving image.
    """
    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks, multi_mask = True, **kwargs):
        super(InpaintingNetwork, self).__init__()

        self.num_down_blocks = num_down_blocks
        self.multi_mask = multi_mask
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        up_blocks = []
        resblock = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            decoder_in_feature = out_features * 2
            if i==num_down_blocks-1:
                decoder_in_feature = out_features
            up_blocks.append(UpBlock2d(decoder_in_feature, in_features, kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(decoder_in_feature, kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(decoder_in_feature, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks[::-1])
        self.resblock = nn.ModuleList(resblock[::-1])

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels
        ###################################################################3

        self.up_blocks2 = nn.ModuleList(up_blocks[::-1])
        self.resblock2 = nn.ModuleList(resblock[::-1])
        self.final2 = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation,align_corners=True)

    def occlude_input(self, inp, occlusion_map):
        if not self.multi_mask:
            if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear',align_corners=True)
        out = inp * occlusion_map
        return out

    def forward(self, source_image, driving_image, dense_motion, dense_motion2):
        out = self.first(source_image) 
        encoder_map = [out]
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            encoder_map.append(out)#(bs,64,256,256),(bs,128,128,128),(bs,256,64,64),(bs,512,32,32)
        
        out2 = self.first(driving_image) 
        encoder_map2 = [out2]
        for i in range(len(self.down_blocks)):
            out2 = self.down_blocks[i](out2)
            encoder_map2.append(out2)

        output_dict = {}
        output_dict['contribution_maps'] = dense_motion['contribution_maps']
        output_dict['deformed_source'] = dense_motion['deformed_source']
        occlusion_map = dense_motion['occlusion_map']
        output_dict['occlusion_map'] = occlusion_map#(bs,1,32,32),(bs,1,64,64),(bs,1,128,128),(bs,1,256,256)
        deformation = dense_motion['deformation']#(bs,64,64,2)
        output_dict['deformation'] = deformation
        
        output_dict['contribution_maps2'] = dense_motion2['contribution_maps']
        output_dict['deformed_driving'] = dense_motion2['deformed_source']
        occlusion_map2 = dense_motion2['occlusion_map']
        output_dict['occlusion_map2'] = occlusion_map2
        deformation2 = dense_motion2['deformation']
        output_dict['deformation2'] = deformation2
        
        ###############################################
        out_ij = self.deform_input(out.detach(), deformation)#(-1,512,32,32)
        out = self.deform_input(out, deformation)#(-1,64,64,2)->(-1,512,32,32)
        out_ij = self.occlude_input(out_ij, occlusion_map[0].detach())#(-1,1,32,32)->(-1,512,32,32)
        out = self.occlude_input(out, occlusion_map[0])#(-1,512,32,32)
        warped_encoder_maps = []
        warped_encoder_maps.append(out_ij)
        
        out_ij_1 = self.deform_input(out_ij, deformation2)
        out_1 = self.deform_input(out, deformation2)
        out_ij_1 = self.occlude_input(out_ij_1, occlusion_map2[0].detach())
        out_1 = self.occlude_input(out_1, occlusion_map2[0])
        warped_encoder_maps_1 = []
        warped_encoder_maps_1.append(out_ij_1)
        
        out_ij2 = self.deform_input(out2.detach(), deformation2)
        out2 = self.deform_input(out2, deformation2)
        out_ij2 = self.occlude_input(out_ij2, occlusion_map2[0].detach())
        out2 = self.occlude_input(out2, occlusion_map2[0])
        warped_encoder_maps2 = []
        warped_encoder_maps2.append(out_ij2)
        
        out_ij_2 = self.deform_input(out_ij2, deformation)
        out_2 = self.deform_input(out2, deformation)
        out_ij_2 = self.occlude_input(out_ij_2, occlusion_map[0].detach())
        out_2 = self.occlude_input(out_2, occlusion_map[0])
        warped_encoder_maps_2 = []
        warped_encoder_maps_2.append(out_ij_2)
        ###########################################s->d
        en_is = []
        for i in range(self.num_down_blocks): 
            out = self.resblock[2*i](out)#(-1,512,32,32)->(bs,512,64,64)->(bs,256,128,128)
            out = self.resblock[2*i+1](out)#(-1,512,32,32)->(bs,512,64,64)->(bs,256,128,128)
            out = self.up_blocks[i](out)#(-1,256,64,64)->(bs,128,128,128)->(bs,64,256,256)
            encode_i = encoder_map[-(i+2)]#(bs,256,64,64)->(bs,128,128,128)->(bs,64,256,256)
            encode_ij = self.deform_input(encode_i.detach(), deformation)#(bs,256,64,64)->(bs,128,128,128)->(bs,64,256,256)
            encode_i = self.deform_input(encode_i, deformation)
            occlusion_ind = 0
            if self.multi_mask:
                occlusion_ind = i+1
            encode_ij = self.occlude_input(encode_ij, occlusion_map[occlusion_ind].detach())
            #(bs,256,64,64), (bs,1,64,64)---((bs,256,64,64))
            encode_i = self.occlude_input(encode_i, occlusion_map[occlusion_ind])
            en_is.append(encode_i)
            warped_encoder_maps.append(encode_ij)
            if(i==self.num_down_blocks-1):
                break
            out = torch.cat([out, encode_i], 1)#final (bs,64,256,256)
        deformed_source = self.deform_input(source_image, deformation)
        output_dict["deformed"] = deformed_source
        output_dict["warped_encoder_maps"] = warped_encoder_maps#(bs,512,32,32),(bs,256,64,64),(bs,128,128,128),(bs,64,256,256)
        occlusion_last = occlusion_map[-1]#(bs,1,256,256)
        if not self.multi_mask:#no
            occlusion_last = F.interpolate(occlusion_last, size=out.shape[2:], mode='bilinear',align_corners=True)
        out = out * (1 - occlusion_last) + encode_i#(bs,64,256,256)
        out = self.final(out)#(bs,3,256,256)
        out = torch.sigmoid(out)
        out = out * (1 - occlusion_last) + deformed_source * occlusion_last
        output_dict["prediction_sd"] = out
        ####################################################################d->s
        en_i2s = []
        for i in range(self.num_down_blocks): 
            out2 = self.resblock[2*i](out2)
            out2 = self.resblock[2*i+1](out2)
            out2 = self.up_blocks[i](out2)
            encode_i2 = encoder_map2[-(i+2)]
            encode_ij2 = self.deform_input(encode_i2.detach(), deformation2)
            encode_i2 = self.deform_input(encode_i2, deformation2)
            occlusion_ind = 0
            if self.multi_mask:
                occlusion_ind = i+1
            encode_ij2 = self.occlude_input(encode_ij2, occlusion_map2[occlusion_ind].detach())
            encode_i2 = self.occlude_input(encode_i2, occlusion_map2[occlusion_ind])
            en_i2s.append(encode_i2)
            warped_encoder_maps2.append(encode_ij2)
            if(i==self.num_down_blocks-1):
                break
            out2 = torch.cat([out2, encode_i2], 1)
        deformed_driving = self.deform_input(driving_image, deformation2)
        output_dict["deformed2"] = deformed_driving
        output_dict["warped_encoder_maps2"] = warped_encoder_maps2
        occlusion_last2 = occlusion_map2[-1]
        if not self.multi_mask:
            occlusion_last2 = F.interpolate(occlusion_last2, size=out2.shape[2:], mode='bilinear',align_corners=True)
        out2 = out2 * (1 - occlusion_last2) + encode_i2
        out2 = self.final(out2)
        out2 = torch.sigmoid(out2)
        out2 = out2 * (1 - occlusion_last2) + deformed_driving * occlusion_last2
        output_dict["prediction_ds"] = out2
        #################################################################s->d->s
        for i in range(self.num_down_blocks): 
            out_1 = self.resblock2[2*i](out_1)
            out_1 = self.resblock2[2*i+1](out_1)
            out_1 = self.up_blocks2[i](out_1)
            # encode_i = encoder_map[-(i+2)]
            encode_ij_1 = self.deform_input(en_is[i].detach(), deformation2)
            encode_i_1 = self.deform_input(en_is[i], deformation2)
            occlusion_ind = 0
            if self.multi_mask:
                occlusion_ind = i+1
            encode_ij_1 = self.occlude_input(encode_ij_1, occlusion_map2[occlusion_ind].detach())
            encode_i_1 = self.occlude_input(encode_i_1, occlusion_map2[occlusion_ind])
            warped_encoder_maps_1.append(encode_ij_1)
            if(i==self.num_down_blocks-1):
                break
            out_1 = torch.cat([out_1, encode_i_1], 1)
        # deformed_sds = self.deform_input(output_dict["prediction_sd"].detach(), deformation2)
        # output_dict["deformed_1"] = deformed_sds
        output_dict["warped_encoder_maps_1"] = warped_encoder_maps_1
        # occlusion_last2 = occlusion_map2[-1]
        # if not self.multi_mask:
        #     occlusion_last2 = F.interpolate(occlusion_last2, size=out_1.shape[2:], mode='bilinear',align_corners=True)
        out_1 = out_1 * (1 - occlusion_last2) + encode_i_1
        out_1 = self.final2(out_1)
        out_1 = torch.sigmoid(out_1)
        out_1 = out_1 * (1 - occlusion_last2) + deformed_driving * occlusion_last2
        output_dict["prediction_sds"] = out_1
        #########################################################d->s->d
        for i in range(self.num_down_blocks): 
            out_2 = self.resblock2[2*i](out_2)
            out_2 = self.resblock2[2*i+1](out_2)
            out_2 = self.up_blocks2[i](out_2)
            # encode_i = encoder_map[-(i+2)]
            encode_ij_2 = self.deform_input(en_i2s[i].detach(), deformation)
            encode_i_2 = self.deform_input(en_i2s[i], deformation)
            occlusion_ind = 0
            if self.multi_mask:
                occlusion_ind = i+1
            encode_ij_2 = self.occlude_input(encode_ij_2, occlusion_map[occlusion_ind].detach())
            encode_i_2 = self.occlude_input(encode_i_2, occlusion_map[occlusion_ind])
            warped_encoder_maps_2.append(encode_ij_2)
            if(i==self.num_down_blocks-1):
                break
            out_2 = torch.cat([out_2, encode_i_2], 1)
        # deformed_dsd = self.deform_input(output_dict["prediction_ds"].detach(), deformation)
        # output_dict["deformed_2"] = deformed_dsd
        output_dict["warped_encoder_maps_2"] = warped_encoder_maps_2
        # occlusion_last2 = occlusion_map2[-1]
        # if not self.multi_mask:
        #     occlusion_last2 = F.interpolate(occlusion_last2, size=out_1.shape[2:], mode='bilinear',align_corners=True)
        out_2 = out_2 * (1 - occlusion_last) + encode_i_2
        out_2 = self.final2(out_2)
        out_2 = torch.sigmoid(out_2)
        out_2 = out_2 * (1 - occlusion_last) + deformed_source * occlusion_last
        output_dict["prediction_dsd"] = out_2
        return output_dict
    
    def get_encode(self, driver_image, occlusion_map):
        out = self.first(driver_image)
        encoder_map = []
        encoder_map.append(self.occlude_input(out.detach(), occlusion_map[-1].detach()))
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out.detach())
            out_mask = self.occlude_input(out.detach(), occlusion_map[2-i].detach())
            encoder_map.append(out_mask.detach())

        return encoder_map

