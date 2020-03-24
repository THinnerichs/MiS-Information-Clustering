from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as tf
from PIL import Image
from torch.autograd import Variable

from projected_sinkhorn import conjugate_sinkhorn, projected_sinkhorn, wasserstein_cost

from scipy.ndimage import gaussian_filter



def custom_greyscale_to_tensor(include_rgb):
    def _inner(img):
        grey_img_tensor = tf.to_tensor(tf.to_grayscale(img, num_output_channels=1))
        result = grey_img_tensor  # 1, 96, 96 in [0, 1]
        assert (result.size(0) == 1)

        if include_rgb:  # greyscale last
            img_tensor = tf.to_tensor(img)
            result = torch.cat([img_tensor, grey_img_tensor], dim=0)
            assert (result.size(0) == 4)

        return result

    return _inner


def custom_cutout(min_box=None, max_box=None):
    def _inner(img):
        w, h = img.size

        # find left, upper, right, lower
        box_sz = np.random.randint(min_box, max_box + 1)
        half_box_sz = int(np.floor(box_sz / 2.))
        x_c = np.random.randint(half_box_sz, w - half_box_sz)
        y_c = np.random.randint(half_box_sz, h - half_box_sz)
        box = (
          x_c - half_box_sz, y_c - half_box_sz, x_c + half_box_sz,
          y_c + half_box_sz)

        img.paste(0, box=box)
        return img

    return _inner


def sobel_process(imgs, include_rgb, using_IR=False):
    bn, c, h, w = imgs.size()

    if not using_IR:
        if not include_rgb:
            assert (c == 1)
            grey_imgs = imgs
        else:
            assert (c == 4)
            grey_imgs = imgs[:, 3, :, :].unsqueeze(1)
            rgb_imgs = imgs[:, :3, :, :]
    else:
        if not include_rgb:
            assert (c == 2)
            grey_imgs = imgs[:, 0, :, :].unsqueeze(1)  # underneath IR
            ir_imgs = imgs[:, 1, :, :].unsqueeze(1)
        else:
            assert (c == 5)
            rgb_imgs = imgs[:, :3, :, :]
            grey_imgs = imgs[:, 3, :, :].unsqueeze(1)
            ir_imgs = imgs[:, 4, :, :].unsqueeze(1)

    sobel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(
        torch.Tensor(sobel1).cuda().float().unsqueeze(0).unsqueeze(0))
    dx = conv1(Variable(grey_imgs)).data

    sobel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = nn.Parameter(
        torch.from_numpy(sobel2).cuda().float().unsqueeze(0).unsqueeze(0))
    dy = conv2(Variable(grey_imgs)).data

    sobel_imgs = torch.cat([dx, dy], dim=1)
    assert (sobel_imgs.shape == (bn, 2, h, w))

    if not using_IR:
        if include_rgb:
            sobel_imgs = torch.cat([rgb_imgs, sobel_imgs], dim=1)
            assert (sobel_imgs.shape == (bn, 5, h, w))
    else:
        if include_rgb:
            # stick both rgb and ir back on in right order (sobel sandwiched inside)
            sobel_imgs = torch.cat([rgb_imgs, sobel_imgs, ir_imgs], dim=1)
        else:
            # stick ir back on in right order (on top of sobel)
            sobel_imgs = torch.cat([sobel_imgs, ir_imgs], dim=1)

    return sobel_imgs


def per_img_demean(img):
    assert (len(img.size()) == 3 and img.size(0) == 3)  # 1 RGB image, tensor
    mean = img.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True) / \
           (img.size(1) * img.size(2))

    return img - mean  # expands


def sobel_make_transforms(config, random_affine=False,
                          cutout=False,
                          cutout_p=None,
                          cutout_max_box=None,
                          affine_p=None):
    tf1_list = []
    tf2_list = []
    tf3_list = []
    if config.crop_orig:
        tf1_list += [
            torchvision.transforms.RandomCrop(tuple(np.array([config.rand_crop_sz,
                                                              config.rand_crop_sz]))),
            torchvision.transforms.Resize(tuple(np.array([config.input_sz,
                                                          config.input_sz]))),
        ]
        tf3_list += [
            torchvision.transforms.CenterCrop(tuple(np.array([config.rand_crop_sz,
                                                              config.rand_crop_sz]))),
            torchvision.transforms.Resize(tuple(np.array([config.input_sz,
                                                          config.input_sz]))),
        ]

    print("(_sobel_multioutput_make_transforms) config.include_rgb: %s" %
          config.include_rgb)
    tf1_list.append(custom_greyscale_to_tensor(config.include_rgb))
    tf3_list.append(custom_greyscale_to_tensor(config.include_rgb))

    if config.fluid_warp:
        # 50-50 do rotation or not
        print("adding rotation option for imgs_tf: %d" % config.rot_val)
        tf2_list += [torchvision.transforms.RandomApply(
            [torchvision.transforms.RandomRotation(config.rot_val)], p=0.5)]

        imgs_tf_crops = []
        for crop_sz in config.rand_crop_szs_tf:
            print("adding crop size option for imgs_tf: %d" % crop_sz)
            imgs_tf_crops.append(torchvision.transforms.RandomCrop(crop_sz))
        tf2_list += [torchvision.transforms.RandomChoice(imgs_tf_crops)]
    else:
        # default
        tf2_list += [torchvision.transforms.RandomCrop(tuple(np.array([config.rand_crop_sz,
                                                                         config.rand_crop_sz])))]

    if random_affine:
        print("adding affine with p %f" % affine_p)
        tf2_list.append(torchvision.transforms.RandomApply(
              [torchvision.transforms.RandomAffine(18,
                                                   scale=(0.9, 1.1),
                                                   translate=(0.1, 0.1),
                                                   shear=10,
                                                   resample=Image.BILINEAR,
                                                   fillcolor=0)], p=affine_p)
        )

    assert (not (cutout and config.cutout))
    if cutout or config.cutout:
        assert (not config.fluid_warp)
        if config.cutout:
              cutout_p = config.cutout_p
              cutout_max_box = config.cutout_max_box

        print("adding cutout with p %f max box %f" % (cutout_p,
                                                      cutout_max_box))
        # https://github.com/uoguelph-mlrg/Cutout/blob/master/images
        # /cutout_on_cifar10.jpg
        tf2_list.append(
            torchvision.transforms.RandomApply(
              [custom_cutout(min_box=int(config.rand_crop_sz * 0.2),
                             max_box=int(config.rand_crop_sz *
                                         cutout_max_box))],
                p=cutout_p)
        )
    else:
        print("not using cutout")

    tf2_list += [
        torchvision.transforms.Resize(tuple(np.array([config.input_sz,
                                                      config.input_sz]))),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                           saturation=0.4, hue=0.125)
    ]

    tf2_list.append(custom_greyscale_to_tensor(config.include_rgb))

    if config.demean:
        print("demeaning data")
        tf1_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                         std=config.data_std))
        tf2_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                         std=config.data_std))
        tf3_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                         std=config.data_std))
    else:
        print("not demeaning data")

    if config.per_img_demean:
        print("per image demeaning data")
        tf1_list.append(per_img_demean)
        tf2_list.append(per_img_demean)
        tf3_list.append(per_img_demean)
    else:
        print("not per image demeaning data")

    tf1 = torchvision.transforms.Compose(tf1_list)
    tf2 = torchvision.transforms.Compose(tf2_list)
    tf3 = torchvision.transforms.Compose(tf3_list)

    return tf1, tf2, tf3


def greyscale_make_transforms(config):
    tf1_list = []
    tf3_list = []
    tf2_list = []

    # tf1 and 3 transforms
    if config.crop_orig:
        # tf1 crop
        if config.tf1_crop == "random":
            print("selected random crop for tf1")
            tf1_crop_fn = torchvision.transforms.RandomCrop(config.tf1_crop_sz)
    elif config.tf1_crop == "centre_half":
        print("selected centre_half crop for tf1")
        tf1_crop_fn = torchvision.transforms.RandomChoice([
          torchvision.transforms.RandomCrop(config.tf1_crop_sz),
          torchvision.transforms.CenterCrop(config.tf1_crop_sz)
        ])
    elif config.tf1_crop == "centre":
        print("selected centre crop for tf1")
        tf1_crop_fn = torchvision.transforms.CenterCrop(config.tf1_crop_sz)
    else:
        assert (False)
    tf1_list += [tf1_crop_fn]

    if config.tf3_crop_diff:
        print("tf3 crop size is different to tf1")
        tf3_list += [torchvision.transforms.CenterCrop(config.tf3_crop_sz)]
    else:
        print("tf3 crop size is same as tf1")
        tf3_list += [torchvision.transforms.CenterCrop(config.tf1_crop_sz)]

    tf1_list += [torchvision.transforms.Resize(config.input_sz),
                 torchvision.transforms.ToTensor()]
    tf3_list += [torchvision.transforms.Resize(config.input_sz),
                 torchvision.transforms.ToTensor()]

    # tf2 transforms
    if config.rot_val > 0:
        # 50-50 do rotation or not
        print("adding rotation option for imgs_tf: %d" % config.rot_val)
        if config.always_rot:
            print("always_rot")
            tf2_list += [torchvision.transforms.RandomRotation(config.rot_val)]
    else:
        print("not always_rot")
        tf2_list += [torchvision.transforms.RandomApply(
            [torchvision.transforms.RandomRotation(config.rot_val)], p=0.5)]

    if config.crop_other:
        imgs_tf_crops = []
        for tf2_crop_sz in config.tf2_crop_szs:
            if config.tf2_crop == "random":
                print("selected random crop for tf2")
                tf2_crop_fn = torchvision.transforms.RandomCrop(tf2_crop_sz)
            elif config.tf2_crop == "centre_half":
                print("selected centre_half crop for tf2")
                tf2_crop_fn = torchvision.transforms.RandomChoice([
                    torchvision.transforms.RandomCrop(tf2_crop_sz),
                    torchvision.transforms.CenterCrop(tf2_crop_sz)
                ])
            elif config.tf2_crop == "centre":
                print("selected centre crop for tf2")
                tf2_crop_fn = torchvision.transforms.CenterCrop(tf2_crop_sz)
            else:
                assert (False)

        print("adding crop size option for imgs_tf: %d" % tf2_crop_sz)
        imgs_tf_crops.append(tf2_crop_fn)

    tf2_list += [torchvision.transforms.RandomChoice(imgs_tf_crops)]

    tf2_list += [torchvision.transforms.Resize(tuple(np.array([config.input_sz,
                                                               config.input_sz])))]

    if not config.no_flip:
        print("adding flip")
        tf2_list += [torchvision.transforms.RandomHorizontalFlip()]
    else:
        print("not adding flip")

    if not config.no_jitter:
        print("adding jitter")
        tf2_list += [
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                               saturation=0.4, hue=0.125)]
    else:
        print("not adding jitter")

    tf2_list += [torchvision.transforms.ToTensor()]

    # admin transforms
    if config.demean:
        print("demeaning data")
        tf1_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                         std=config.data_std))
        tf2_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                         std=config.data_std))
        tf3_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                         std=config.data_std))
    else:
        print("not demeaning data")

    if config.per_img_demean:
        print("per image demeaning data")
        tf1_list.append(per_img_demean)
        tf2_list.append(per_img_demean)
        tf3_list.append(per_img_demean)
    else:
        print("not per image demeaning data")

    tf1 = torchvision.transforms.Compose(tf1_list)
    tf2 = torchvision.transforms.Compose(tf2_list)
    tf3 = torchvision.transforms.Compose(tf3_list)

    return tf1, tf2, tf3

def greyscale_ADef_linf_norm_transform(config):
    """
    Build perturbations with the ADef algorithm from an arbitrary loss function (e.g. categorical-cross entropy).
    Sampled from Wasserstein gaussian within a l-infinity ball around the datapoint.
    :param config:
    :return:
    """

    tf1_list = []
    tf2_list = []
    tf3_list = []

    # tf1 and 3 transforms
    if config.crop_orig:
        # tf1 crop
        if config.tf1_crop == "random":
            print("selected random crop for tf1")
            tf1_crop_fn = torchvision.transforms.RandomCrop(config.tf1_crop_sz)
        elif config.tf1_crop == "centre_half":
            print("selected centre_half crop for tf1")
            tf1_crop_fn = torchvision.transforms.RandomChoice([
                torchvision.transforms.RandomCrop(config.tf1_crop_sz),
                torchvision.transforms.CenterCrop(config.tf1_crop_sz)
            ])
        elif config.tf1_crop == "centre":
            print("selected centre crop for tf1")
            tf1_crop_fn = torchvision.transforms.CenterCrop(config.tf1_crop_sz)
        else:
            assert (False)
        tf1_list += [tf1_crop_fn]

        if config.tf3_crop_diff:
            print("tf3 crop size is different to tf1")
            tf3_list += [torchvision.transforms.CenterCrop(config.tf3_crop_sz)]
        else:
            print("tf3 crop size is same as tf1")
            tf3_list += [torchvision.transforms.CenterCrop(config.tf1_crop_sz)]

    tf1_list += [torchvision.transforms.Resize(config.input_sz),
                 torchvision.transforms.ToTensor()]
    tf3_list += [torchvision.transforms.Resize(config.input_sz),
                 torchvision.transforms.ToTensor()]

    # tf2 transforms
    if config.rot_val > 0:
        # 50-50 do rotation or not
        print("adding rotation option for imgs_tf: %d" % config.rot_val)
        if config.always_rot:
            print("always_rot")
            tf2_list += [torchvision.transforms.RandomRotation(config.rot_val)]
        else:
            print("not always_rot")
            tf2_list += [torchvision.transforms.RandomApply(
                [torchvision.transforms.RandomRotation(config.rot_val)], p=0.5)]

    if config.crop_other:
        imgs_tf_crops = []
        for tf2_crop_sz in config.tf2_crop_szs:
            if config.tf2_crop == "random":
                print("selected random crop for tf2")
                tf2_crop_fn = torchvision.transforms.RandomCrop(tf2_crop_sz)
            elif config.tf2_crop == "centre_half":
                print("selected centre_half crop for tf2")
                tf2_crop_fn = torchvision.transforms.RandomChoice([
                    torchvision.transforms.RandomCrop(tf2_crop_sz),
                    torchvision.transforms.CenterCrop(tf2_crop_sz)
                ])
            elif config.tf2_crop == "centre":
                print("selected centre crop for tf2")
                tf2_crop_fn = torchvision.transforms.CenterCrop(tf2_crop_sz)
            else:
                assert (False)

            print("adding crop size option for imgs_tf: %d" % tf2_crop_sz)
            imgs_tf_crops.append(tf2_crop_fn)

        tf2_list += [torchvision.transforms.RandomChoice(imgs_tf_crops)]

    tf2_list += [torchvision.transforms.Resize(tuple(np.array([config.input_sz,
                                                               config.input_sz])))]

    if not config.no_flip:
        print("adding flip")
        tf2_list += [torchvision.transforms.RandomHorizontalFlip()]
    else:
        print("not adding flip")

    if not config.no_jitter:
        print("adding jitter")
        tf2_list += [
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                             saturation=0.4, hue=0.125)]
    else:
        print("not adding jitter")

    tf2_list += [torchvision.transforms.ToTensor()]

    # admin transforms
    if config.demean:
        print("demeaning data")
        tf1_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                         std=config.data_std))
        tf2_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                         std=config.data_std))
        tf3_list.append(torchvision.transforms.Normalize(mean=config.data_mean,
                                                       std=config.data_std))
    else:
        print("not demeaning data")

    if config.per_img_demean:
        print("per image demeaning data")
        tf1_list.append(per_img_demean)
        tf2_list.append(per_img_demean)
        tf3_list.append(per_img_demean)
    else:
        print("not per image demeaning data")

    tf1 = torchvision.transforms.Compose(tf1_list)
    tf2 = torchvision.transforms.Compose(tf2_list)
    tf3 = torchvision.transforms.Compose(tf3_list)

    return tf1, tf2, tf3

def greyscale_sinkhorn_ball_perturbation(X,
                                         epsilon=0.01,
                                         epsilon_iters=10,
                                         epsilon_factor=1.1,
                                         p=2,
                                         kernel_size=5,
                                         maxiters=400,
                                         alpha=0.1,
                                         xmin=0,
                                         xmax=1,
                                         normalize=lambda x:x,
                                         verbose=0,
                                         regularization=1000,
                                         sinkhorn_maxiters=400,
                                         ball='wasserstein',
                                         norm='wasserstein'
                                         ):

    """
    Use iterated Sinkhorn iterations over an arbitrary loss function (e.g. categorial cross entropy with random label) over an arbitrary model.
    Call with torchvision.transforms.Lambda to use as torchvision.Transformation.

    :param X:
    :param epsilon:
    :param epsilon_iters:
    :param epsilon_factor:
    :param p:
    :param kernel_size:
    :param maxiters:
    :param alpha:
    :param xmin:
    :param xmax:
    :param normalize:
    :param verbose:
    :param regularization:
    :param sinkhorn_maxiters:
    :param ball:
    :param norm:
    :return:
    """



    batch_size = X.size(0)

    # initialize net as multiplication of image with random matrix
    np.random.seed(42)
    random_matrix = torch.from_numpy(np.random.random_sample(list(X.size())))
    downscaling_factor = np.prod(list(X.size())[1:])
    random_matrix /= downscaling_factor

    # randomly initialize y
    y = torch.rand(0, 2, (batch_size))

    def net(input_batch):
        return_list = []
        for i in range(batch_size):
            input_matrix = input_batch[i]
            input_matrix /= 255

            return_list+= [(input_matrix * random_matrix).sum()]
        return torch.Tensor(return_list)

    epsilon = X.new_ones(batch_size) * epsilon
    C = wasserstein_cost(X, p=p, kernel_size=kernel_size)
    normalization = X.view(batch_size, -1).sum(-1).view(batch_size, 1, 1, 1)
    X_ = X.clone()

    X_best = X.clone()
    result = net(normalize(X))
    print(result.size())
    err_best = err = net(normalize(X)).max(1) != y
    epsilon_best = epsilon.clone()

    t = 0
    while True:
        X_.requires_grad = True
        opt = optim.SGD([X_], lr=0.1)
        loss = nn.CrossEntropyLoss()(net(normalize(X_)), y)
        opt.zero_grad()
        loss.backward()

        with torch.no_grad():
            # take a step
            if norm == 'linfinity':
                X_[~err] += alpha * torch.sign(X_.grad[~err])
            elif norm == 'l2':
                X_[~err] += (alpha * X_.grad / (X_.grad.view(X.size(0), -1).norm(dim=1).view(X.size(0), 1, 1, 1)))[~err]
            elif norm == 'wasserstein':
                sd_normalization = X_.view(batch_size, -1).sum(-1).view(batch_size, 1, 1, 1)
                X_[~err] = (conjugate_sinkhorn(X_.clone() / sd_normalization,
                                               X_.grad, C, alpha, regularization,
                                               verbose=verbose, maxiters=sinkhorn_maxiters
                                               ) * sd_normalization)[~err]
            else:
                raise ValueError("Unknown norm")

            # project onto ball
            if ball == 'wasserstein':
                X_[~err] = (projected_sinkhorn(X.clone() / normalization,
                                               X_.detach() / normalization,
                                               C,
                                               epsilon,
                                               regularization,
                                               verbose=verbose,
                                               maxiters=sinkhorn_maxiters) * normalization)[~err]
            elif ball == 'linfinity':
                X_ = torch.min(X_, X + epsilon.view(X.size(0), 1, 1, 1))
                X_ = torch.max(X_, X - epsilon.view(X.size(0), 1, 1, 1))
            else:
                raise ValueError("Unknown ball")
            X_ = torch.clamp(X_, min=xmin, max=xmax)

            err = (net(normalize(X_)).max(1)[1] != y)
            err_rate = err.sum().item() / batch_size
            if err_rate > err_best.sum().item() / batch_size:
                X_best = X_.clone()
                err_best = err
                epsilon_best = epsilon.clone()

            if verbose and t % verbose == 0:
                print(t, loss.item(), epsilon.mean().item(), err_rate)

            t += 1
            if err_rate == 1 or t == maxiters:
                break

            if t > 0 and t % epsilon_iters == 0:
                epsilon[~err] *= epsilon_factor

    epsilon_best[~err] = float('inf')
    return X_best, err_best, epsilon_best

def greyscale_random_smoothed_deformation(X,
                                          deformation_range=4,
                                          sigma=2):

    """
    Apply a random deformation within deformation_range smoothed by a gaussian filter
    with sigma=sigma. Call with torchvision.transforms.Lambda to use as torchvision.Transformation.

    :param X:
    :param deformation_range:
    :param sigma:
    :return:
    """

    batch_size = X.size(0)
    height, width = X.size(1), X.size(2)

    x_deformation_matrix = np.random.randint(-deformation_range,
                                             deformation_range+1,
                                             size=(height, width)
                                             )


    y_deformation_matrix = np.random.randint(-deformation_range,
                                             deformation_range+1,
                                             size=(height, width)
                                             )

    x_filtered = gaussian_filter(x_deformation_matrix, sigma=sigma)
    y_filtered = gaussian_filter(y_deformation_matrix, sigma=sigma)

    return_batch = X.clone()

    for image_index in range(batch_size):
        for i in range(height):
            for j in range(width):
                # Check whether indices are still within bounds
                x_index = j + x_filtered[i,j]
                x_index = width-1 if x_index >= width else x_index
                x_index = 0 if x_index < 0 else x_index

                y_index = j + y_filtered[i, j]
                y_index = height - 1 if y_index >= height else y_index
                y_index = 0 if y_index < 0 else y_index

                return_batch[image_index, i, j] = X[image_index,
                                                    x_index,
                                                    y_index]


def ADef_linearized_wasserstein_perturbation(radius):
    # @TODO use ADef linearization for plane and take perturbations with repsect to Wasserstein ball around sample
    pass
