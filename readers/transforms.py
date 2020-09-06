import sys
sys.path.append('/home/aistudio')
import random
import numpy as np 
from PIL import Image
import math


__all__ = ['GroupImageTransform']


class GroupCrop(object):

    def __init__(self, crop_quadruple):
        self.crop_quadruple = crop_quadruple
    

    def __call__(self, img_group, is_flow=False):
        return [image.resize(self.crop_quadruple) for image in img_group], self.crop_quadruple
    

class GroupCenterCrop(object):

    def __init__(self, size):
        self.size = size if not isinstance(size, int) else (size, size)
    

    def __call__(self, img_group, is_flow=False):
        w, h = img_group[0].size
        tw, th = self.size
        x1 = (w - tw) // 2
        y1 = (h - th) // 2
        box = np.array([x1, y1, x1 + tw, y1 + th])
        return ([image.crop(box) for image in img_group],
                np.array([x1, y1, tw, th], dtype=np.float32))


class GroupColorJitter(object):

    def __init__(self, color_space_aug=False, alphastd=0.1, eigval=None, eigvec=None):
        if eigval is None:
            self.eigval = np.array([55.46, 4.794, 1.148])
        if eigvec is None:
            self.eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                                    [-0.5808, -0.0045, -0.8140],
                                    [-0.5836, -0.6948, 0.4203]])
        self.alphastd = alphastd
        self.color_space_aug = color_space_aug
    

    @staticmethod
    def brightness(img, delta):
        if random.uniform(0, 1) > 0.5:
            delta = np.array(delta).astype(np.float32)
            img = img + delta
        
        return img
    

    @staticmethod
    def contrast(img, alpha):
        if random.uniform(0, 1) > 0.5:
            alpha = np.array(alpha).astype(np.float32)
            img = img * alpha
        
        return img
    

    @staticmethod
    def saturation(img, alpha):
        if random.uniform(0, 1) > 0.5:
            gray = img * np.array([0.299, 0.587, 0.114]).astype(np.float32)
            gray = np.sum(gray, 2, keepdims=True)
            gray *= (1.0 - alpha)
            img = img * alpha
            img = img + gray
        
        return img
    

    @staticmethod
    def hue(img, alpha):
        if random.uniform(0, 1) > 0.5:
            u = np.cos(alpha * np.pi)
            w = np.sin(alpha * np.pi)
            bt = np.array([[1.0, 0.0, 0.0],
                           [0.0, u, -w],
                           [0.0, w, u]])
            tyiq = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.274, -0.321],
                             [0.211, -0.523, 0.311]])
            ityiq = np.array([[1.0, 0.956, 0.114],
                              [1.0, -0.272, -0.647],
                              [1.0, -1.107, 1.705]])
            t = np.dot(np.dot(ityiq, bt), tyiq).T
            t = np.array(t).astype(np.float32)
            img = np.dot(img, t)
        
        return img
    

    def __call__(self, img_group):
        if self.color_space_aug:
            bright_delta = np.random.uniform(-32, 32)
            contrast_alpha = np.random.uniform(0.6, 1.4)
            saturation_alpha = np.random.uniform(0.6, 1.4)
            hue_alpha = random.uniform(-18, 18)
            out = []
            for img in img_group:
                img = np.array(img)[:,:,::-1] # rgb -> bgr
                img = self.brightness(img, delta=bright_delta)
                if random.uniform(0, 1) > 0.5:
                    img = self.contrast(img, alpha=contrast_alpha)
                    img = self.saturation(img, alpha=saturation_alpha)
                    img = self.hue(img, alpha=hue_alpha)
                else:
                    img = self.saturation(img, alpha=saturation_alpha)
                    img = self.hue(img, alpha=hue_alpha)
                    img = self.contrast(img, alpha=contrast_alpha)
                out.append(img)
            img_group = out
        
        alpha = np.random.normal(0, self.alphastd, size=(3,))
        rgb = np.array(np.dot(self.eigvec * alpha, self.eigval)).astype(np.float32)
        bgr = np.expand_dims(np.expand_dims(rgb[::-1], 0), 0)
        return [img + bgr for img in img_group]



class GroupNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    

    def __call__(self, img_group):
        outs = []
        for image in img_group:
            out = np.array(image)
            out = (out / 255.0 - self.mean) / self.std
            outs.append(out)
        return outs


class RandomResizeCrop(object):

    def __init__(self, size, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
    

    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w
            
            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w
    

    def __call__(self, img_group):
        x1, y1, th, tw = self.get_params(img_group[0], self.scale, self.ratio)
        box = np.array([x1, y1, x1 + tw - 1, y1 + th - 1], dtype=np.float32)
        return ([image.crop(box).resize(self.size) for image in img_group], box)


class Group3CropSample(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
    

    def __call__(self, img_group, is_flow=False):
        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size
        assert crop_h == image_h or crop_w == image_w

        if crop_h == image_h:
            w_step = (image_w - crop_w) // 2
            offsets = list()
            offsets.append((0, 0))
            offsets.append((2 * w_step, 0))
            offsets.append((w_step, 0))
        elif crop_w == image_w:
            h_step = (image_h - crop_h) // 2
            offsets = list()
            offsets.append((0, 0))
            offsets.append((0, 2 * h_step))
            offsets.append((0, h_step))
        
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop([o_w, o_h, o_w + crop_w - 1, o_h + crop_h - 1])
                normal_group.append(crop)
                flip_crop = crop.transpose(Image.FLIP_LEFT_RIGHT)

                flip_group.append(flip_crop)
            
            oversample_group.extend(normal_group)
            # oversample_group.extend(flip_group)
        return oversample_group, None
    

class GroupOverSample(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
    

    def __call__(self, img_group, is_flow=False):

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(
            False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()

            for i, img in enumerate(img_group):
                crop = img.crop([o_w, o_h, o_w + crop_w - 1, o_h + crop_h - 1])
                normal_group.append(crop)
                flip_crop = crop.transpose(Image.FLIP_LEFT_RIGHT)

                flip_group.append(flip_group)
            
            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        
        return oversample_group, None


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 0.875, 0.75]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
    

    def __call__(self, img_group, is_flow=False):
        im_w, im_h = img_group[0].shape
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size((im_w, im_h))
        box = np.array([offset_w, offset_h, offset_w + crop_w - 1, offset_h + crop_h - 1])
        crop_image_group = [image.crop(box) for image in img_group]
        ret_img_group = [image.resize((self.input_size[0], self.input_size[1])) 
            for image in crop_image_group]
        return (ret_img_group, np.array([offset_w, offset_w, crop_w, crop_h], dtype=np.float32))
    

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size
        base_sizes = min(image_w, image_h)
        crop_sizes = [int(base_sizes * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))
        
        crop_pair = random.choice(pairs)

        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_size(image_w, image_h, crop_pair[0], crop_pair[1])
        
        return crop_pair[0], crop_pair[1], w_offset, h_offset

    
    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)
    

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupImageTransform(object):

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 crop_size=None,
                 oversample=None,
                 resize_crop=False,
                 colorjitter=False,
                 color_space_aug=False,
                 max_distort=1):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.resize_crop = resize_crop
        self.colorjitter = GroupColorJitter(color_space_aug=color_space_aug) if color_space_aug else None
        self.normalize = GroupNormalize(mean, std)

        if resize_crop:
            self.op_crop = RandomResizeCrop(crop_size)
        else:
            if crop_size is not None:
                if oversample == 'three_crop':
                    self.op_crop = Group3CropSample(crop_size)
                elif oversample == 'ten_crop':
                    self.op_crop = GroupOverSample(crop_size)
                else:
                    self.op_crop = GroupCenterCrop(crop_size)
            else:
                self.op_crop = None
    

    def __call__(self, img_group, scale, crop_history=None, flip=False,
        keep_ratio=True, div_255=False, is_flow=False):

        if self.resize_crop:
            img_group, crop_quadruple = self.op_crop(img_group)
            img_shape = img_group[0].size
            scale_factor = None
        else:
            if keep_ratio:
                w, h = img_group[0].size
                new_w, new_h = int(scale[1]), int(scale[1] * h / w)
                img_group = [image.resize((new_w, new_h)) for image in img_group]
            else:
                raise NotImplementedError
                
            if crop_history is not None:
                self.op_crop = GroupCrop(crop_history)
            if self.op_crop is not None:
                img_group, crop_quadruple = self.op_crop(img_group, is_flow=is_flow)
            else:
                crop_quadruple = None
            
            img_shape = img_group[0].size
        
        
        if flip:
            img_group = [image.transpose(Image.FLIP_LEFT_RIGHT) for image in img_group]
        
        if self.colorjitter is not None:
            img_group = self.colorjitter(img_group)
        

        img_group = self.normalize(img_group)

        img_group = [image.transpose(2, 0, 1) for image in img_group]

        img_group = np.stack(img_group, 0)

        return img_group
        