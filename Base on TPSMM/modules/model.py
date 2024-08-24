from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, TPS
from torchvision import models
import numpy as np

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, bg_predictor, dense_motion_network, inpainting_network, train_params, *kwargs):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.inpainting_network = inpainting_network
        self.dense_motion_network = dense_motion_network

        self.bg_predictor = None
        if bg_predictor:
            self.bg_predictor = bg_predictor
            self.bg_start = train_params['bg_start']

        self.train_params = train_params
        self.scales = train_params['scales']

        self.pyramid = ImagePyramide(self.scales, inpainting_network.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']
        self.dropout_epoch = train_params['dropout_epoch']
        self.dropout_maxp = train_params['dropout_maxp']
        self.dropout_inc_epoch = train_params['dropout_inc_epoch']
        self.dropout_startp =train_params['dropout_startp']
        
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()


    def forward(self, x, epoch):
        kp_source = self.kp_extractor(x['source'])
        kp_driving = self.kp_extractor(x['driving'])
        bg_param = None
        bg_param2 = None
        if self.bg_predictor:
            if(epoch>=self.bg_start):
                bg_param = self.bg_predictor(x['source'], x['driving'])
                bg_param2 = self.bg_predictor(x['driving'], x['source'])
          
        if(epoch>=self.dropout_epoch):
            dropout_flag = False
            dropout_p = 0
        else:
            # dropout_p will linearly increase from dropout_startp to dropout_maxp 
            dropout_flag = True
            dropout_p = min(epoch/self.dropout_inc_epoch * self.dropout_maxp + self.dropout_startp, self.dropout_maxp)
        
        dense_motion = self.dense_motion_network(source_image=x['source'], kp_driving=kp_driving,
                                                    kp_source=kp_source, bg_param = bg_param, 
                                                    dropout_flag = dropout_flag, dropout_p = dropout_p)
        dense_motion2 = self.dense_motion_network(source_image=x['driving'], kp_driving=kp_source,
                                                    kp_source=kp_driving, bg_param = bg_param2, 
                                                    dropout_flag = dropout_flag, dropout_p = dropout_p)
        generated = self.inpainting_network(x['source'], x['driving'], dense_motion, dense_motion2)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_real_S = self.pyramid(x['source'])
        pyramide_generated = self.pyramid(generated['prediction_sd'])
        pyramide_generated2 = self.pyramid(generated['prediction_ds'])
        pyramide_generated3 = self.pyramid(generated['prediction_sds'])
        pyramide_generated4 = self.pyramid(generated['prediction_dsd'])

        # reconstruction loss
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual_sd'] = value_total
            
            value_total2 = 0
            for scale in self.scales:
                x_vgg2 = self.vgg(pyramide_generated2['prediction_' + str(scale)])
                y_vgg2 = self.vgg(pyramide_real_S['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg2[i] - y_vgg2[i].detach()).mean()
                    value_total2 += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual_ds'] = value_total2
            
            value_total3 = 0
            for scale in self.scales:
                x_vgg3 = self.vgg(pyramide_generated3['prediction_' + str(scale)])
                y_vgg3 = self.vgg(pyramide_real_S['prediction_' + str(scale)])
                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg3[i] - y_vgg3[i].detach()).mean()
                    value_total3 += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual_sds'] = value_total3
            
            value_total4 = 0
            for scale in self.scales:
                x_vgg4 = self.vgg(pyramide_generated4['prediction_' + str(scale)])
                y_vgg4 = self.vgg(pyramide_real['prediction_' + str(scale)])
                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg4[i] - y_vgg4[i].detach()).mean()
                    value_total4 += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual_dsd'] = value_total4

        # equivariance loss
        if self.loss_weights['equivariance_value'] != 0:
            transform_random = TPS(mode = 'random', bs = x['driving'].shape[0], **self.train_params['transform_params'])
            transform_grid = transform_random.transform_frame(x['driving'])
            transformed_frame = F.grid_sample(x['driving'], transform_grid, padding_mode="reflection",align_corners=True)
            transformed_kp = self.kp_extractor(transformed_frame)
            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp
            warped = transform_random.warp_coordinates(transformed_kp['fg_kp'])
            kp_d = kp_driving['fg_kp']
            value = torch.abs(kp_d - warped).mean()
            loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value
            ####################################
            transform_random = TPS(mode = 'random', bs = x['source'].shape[0], **self.train_params['transform_params'])
            transform_grid = transform_random.transform_frame(x['source'])
            transformed_frame = F.grid_sample(x['source'], transform_grid, padding_mode="reflection",align_corners=True)
            transformed_kp = self.kp_extractor(transformed_frame)
            generated['transformed_frame2'] = transformed_frame
            generated['transformed_kp2'] = transformed_kp
            warped = transform_random.warp_coordinates(transformed_kp['fg_kp'])
            kp_s = kp_source['fg_kp']
            value = torch.abs(kp_s - warped).mean()
            loss_values['equivariance_value2'] = self.loss_weights['equivariance_value'] * value

        # warp loss
        if self.loss_weights['warp_loss'] != 0:
            occlusion_map = generated['occlusion_map']
            encode_map = self.inpainting_network.get_encode(x['driving'], occlusion_map)
            encode_map_1 = generated['warped_encoder_maps_2']
            decode_map = generated['warped_encoder_maps']
            value_1 = 0
            value = 0
            for i in range(len(encode_map)):
                value += torch.abs(encode_map[i].detach()-decode_map[-i-1]).mean()
            for i in range(len(encode_map_1)):
                value_1 += torch.abs(encode_map_1[i].detach()-decode_map[i]).mean()

            loss_values['warp_loss'] = self.loss_weights['warp_loss'] * (value +value_1)
            ####################################
            occlusion_map2 = generated['occlusion_map2']
            encode_map2 = self.inpainting_network.get_encode(x['source'], occlusion_map2)
            encode_map_2 = generated['warped_encoder_maps_1']
            decode_map2 = generated['warped_encoder_maps2']
            value = 0
            value_1 = 0
            for i in range(len(encode_map2)):
                value += torch.abs(encode_map2[i].detach()-decode_map2[-i-1]).mean()
            for i in range(len(encode_map_2)):
                value_1 += torch.abs(encode_map_2[i].detach()-decode_map2[i]).mean()

            loss_values['warp_loss2'] = self.loss_weights['warp_loss'] * (value + value_1)
        
        # bg loss
        if self.bg_predictor and epoch >= self.bg_start and self.loss_weights['bg'] != 0:
            # bg_param_reverse = self.bg_predictor(x['driving'], x['source'])
            value = torch.matmul(bg_param, bg_param2)
            eye = torch.eye(3).view(1, 1, 3, 3).type(value.type())
            value = torch.abs(eye - value).mean()
            loss_values['bg'] = self.loss_weights['bg'] * value
            
        value_flow = self.compute_loss_flow_smooth(generated['deformation'], x['driving']) +\
           self.compute_loss_flow_smooth(generated['deformation2'], x['source']) 
        loss_values['value_flow'] = value_flow

        return loss_values, generated
    
    def gradients(self, img):
        dy = img[:,:,1:,:] - img[:,:,:-1,:]
        dx = img[:,:,:,1:] - img[:,:,:,:-1]
        return dx, dy
    def compute_loss_flow_smooth(self, opitcal_flow, img):
        opitcal_flow = opitcal_flow.permute(0, 3, 1, 2)
        opitcal_flow = F.interpolate(opitcal_flow, size=(img.shape[2], img.shape[3]), mode='bilinear')
        flow = opitcal_flow/20.0
        img_grad_x, img_grad_y = self.gradients(img)
        w_x = torch.exp(-10.0 * torch.abs(img_grad_x).mean(1).unsqueeze(1))
        w_y = torch.exp(-10.0 * torch.abs(img_grad_y).mean(1).unsqueeze(1))

        dx, dy = self.gradients(flow)
        dx2, _ = self.gradients(dx)
        _, dy2 = self.gradients(dy)
        error = (w_x[:,:,:,1:] * torch.abs(dx2)).mean((1,2,3)) + (w_y[:,:,1:,:] * torch.abs(dy2)).mean((1,2,3))
        loss = error / 2.0
        return loss
