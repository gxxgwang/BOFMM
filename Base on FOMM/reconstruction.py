import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
# import cv2
from sync_batchnorm import DataParallelWithCallback

torch.cuda.set_device(0)
device = torch.device('cuda:0')


def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    loss_list = []
    # if torch.cuda.is_available():
    #     generator = DataParallelWithCallback(generator)
    #     kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].to(device)
            kp_source = kp_detector(x['video'][:, :, 0])

            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]
                driving = x['video'][:, :, frame_idx]

                kp_driving = kp_detector(driving)
                out = generator(driving, source, kp_source=kp_driving, kp_driving=kp_source)
                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                del out['sparse_deformed2']
                predictions.append(np.transpose(out['prediction_ds'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
                                                                                    driving=driving, out=out)
                visualizations.append(visualization)

                loss_list.append(torch.abs(out['prediction_ds'] - driving).mean().cpu().numpy())

            # predictions = np.concatenate(predictions, axis=1)
            if not os.path.exists(os.path.join(png_dir, x['name'][0])):
                os.makedirs(os.path.join(png_dir, x['name'][0]))
            png_dir1 = os.path.join(png_dir, x['name'][0])
            for i in range(x['video'].shape[2]):
                img = predictions[i]
                imageio.imsave(os.path.join(png_dir1, "{:05d}.png".format(i)), (255 * img).astype(np.uint8))

            # image_name = x['name'][0] + config['reconstruction_params']['format']
            # imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    print("Reconstruction loss: %s" % np.mean(loss_list))
