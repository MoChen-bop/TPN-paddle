import sys
sys.path.append('/home/aistudio')
import os
import os.path as path
import random
import numpy as np 
from PIL import Image
import cv2

import paddle
import paddle.fluid as fluid

from TPN.readers.transforms import GroupImageTransform


class RawFramesRecord(object):

    def __init__(self, row):
        self._data = row
    

    @property
    def path(self,):
        return self._data[0]
    

    @property
    def num_frames(self):
        return int(self._data[1])
    

    @property
    def label(self,):
        return int(self._data[2])


class RawFramesDataset():

    def __init__(self, 
                 ann_file, 
                 img_prefix, 
                 img_norm_cfg, 
                 num_segments=3,
                 new_length=1, 
                 new_step=1, 
                 random_shift=True, 
                 temporal_jitter=False,
                 modality='RGB', 
                 image_tmpl='img_{}.jpg', 
                 img_scale=256, 
                 img_scale_file=None, 
                 input_size=224, 
                 div_255=False, 
                 flip_ratio=0.5,
                 resize_keep_ratio=True, 
                 resize_crop=False, 
                 color_jitter=False, 
                 color_space_aug=False,
                 test_mode=False, 
                 oversample=None, 
                 max_distort=1,
                 input_format='NCHW',
                 num_threads=8, 
                 buf_size=1024 * 4,
                 visualize=False):
        self.img_prefix = img_prefix 

        self.video_infos = self.load_annotations(ann_file)
        random.shuffle(self.video_infos)

        self.img_norm_cfg = img_norm_cfg

        self.num_segments = num_segments

        self.old_length = new_length * new_step
        self.new_length = new_length
        self.new_step = new_step

        self.random_shift = random_shift
        
        self.temporal_jitter = temporal_jitter
        self.resize_crop = resize_crop
        self.color_jitter = color_jitter
        self.color_space_aug = color_space_aug

        if isinstance(modality, (list, tuple)):
            self.modality = modality
            num_modality = len(modality)
        else:
            self.modality = [modality]
            num_modality = 1
        
        if isinstance(image_tmpl, (list, tuple)):
            self.image_tmpls = image_tmpl
        else:
            self.image_tmpls = [image_tmpl]
        assert len(self.image_tmpls) == num_modality

        if isinstance(img_scale, int):
            img_scale = (np.Inf, img_scale)
        self.img_scale = img_scale

        if img_scale_file is not None:
            self.img_scale_dict = {line.split(' ')[0]: 
                                        (int(line.split(' ')[1]),
                                         int(line.split(' ')[2]))
                                   for line in open(img_scale_file)}
        else:
            self.img_scale_dict = None
        
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size

        self.div_255 = div_255

        self.div_255 = div_255

        self.flip_ratio = flip_ratio
        self.resize_keep_ratio = resize_keep_ratio

        self.test_mode = test_mode

        if not self.test_mode:
            self._set_group_flag()
        

        assert oversample in [None, 'three_crop', 'ten_crop']
        self.oversample = oversample
        self.img_group_transform = GroupImageTransform(
            crop_size=self.input_size,
            oversample=oversample,
            max_distort=max_distort,
            resize_crop=self.resize_crop,
            colorjitter=self.color_jitter,
            color_space_aug=self.color_space_aug,
            **self.img_norm_cfg
        )

        assert input_format in ['NCHW', 'NCTHW']
        self.input_format = input_format
        self.num_threads = num_threads
        self.buf_size = buf_size
        self.visualize = visualize


    def create_reader(self, ):

        def reader():
            random.shuffle(self.video_infos)
            for idx in range(len(self.video_infos)):
                yield idx
        
        return paddle.reader.xmap_readers(self.get_item, reader, self.num_threads, self.buf_size)


    def batch_reader(self, batch_size):
        reader = self.create_reader()
        
        def _batch_reader():
            batch_out = []
            for img_group, label in reader():
                if img_group is None:
                    continue
                
                batch_out.append((img_group, label))
                if len(batch_out) == batch_size:
                    yield batch_out
                    batch_out = []
        return _batch_reader


    def __len__(self,):
        return len(self.video_infos)


    def load_annotations(self, ann_file):
        return [RawFramesRecord(x.strip().split(' ')) for x in open(ann_file)]


    def get_ann_info(self, idx):
        return {'path': self.video_infos[idx].path,
                'num_frames': self.video_infos[idx].num_frames,
                'label': self.video_infos[idx].label}


    def _set_group_flag(self,):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1


    def _load_image(self, directory, image_tmpl, modality, idx):
        if modality in ['RGB', 'RGBDiff']:
            return [Image.open(path.join(directory, image_tmpl.format(idx)))]
        elif modality == 'Flow':
            x_imgs = Image.open(path.join(directory, image_tmpl.format('x', idx), 'L'))
            y_imgs = Image.open(path.join(directory, image_tmpl.format('y', idx)))

            return [x_imgs, y_imgs]
        else:
            raise ValueError(
                'Not implemented yet; modality should be ["RGB", "RGBDiff", "Flow"]'
            )
    

    def _sample_indices(self, record):
        average_duration = (record.num_frames - self.old_length + 1) // self.num_segments

        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(average_duration, size=self.num_segments)
        
        elif record.num_frames > max(self.num_segments, self.old_length):
            offsets = np.sort(np.random.randint(record.num_frames - self.old_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments))
        
        if self.temporal_jitter:
            skip_offsets = np.random.randint(self.new_step, size=self.old_length // self.new_step)
        else:
            skip_offsets = np.zeros(self.old_length // self.new_step, dtype=int)
        
        return offsets + 1, skip_offsets
    

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.old_length - 1:
            tick = (record.num_frames - self.old_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments, ))
        
        if self.temporal_jitter:
            skip_offsets = np.random.randint(self.new_step, size=self.old_length // self.new_step)
        else:
            skip_offsets = np.zeros(self.old_length // self.new_step, dtype=int)
        
        return offsets + 1, skip_offsets
    

    def _get_test_indices(self, record):
        if record.num_frames > self.old_length - 1:
            tick = (record.num_frames - self.old_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments))
        
        if self.temporal_jitter:
            skip_offsets = np.random.randint(self.new_step, size=self.old_length // self.new_step)
        else:
            skip_offsets = np.zeros(self.old_length // self.new_step, dtype=int)
        
        return offsets + 1, skip_offsets
    

    def _get_frames(self, record, image_tmpl, modality, indices, skip_offsets):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i, ind in enumerate(range(0, self.old_length, self.new_step)):
                if p + skip_offsets[i] <= record.num_frames:
                    seg_imgs = self._load_image(path.join(self.img_prefix, record.path),
                        image_tmpl, modality, p + skip_offsets[i])
                else:
                    seg_imgs = self._load_image(path.join(self.img_prefix, record.path), 
                        image_tmpl, modality, p)
                
                images.extend(seg_imgs)

                if p + self.new_step < record.num_frames:
                    p += self.new_step
        return images
    

    def get_item(self, idx):
        record = self.video_infos[idx]
        if self.test_mode:
            segment_indices, skip_offsets = self._get_test_indices(record)
        else:
            segment_indices, skip_offsets = self._sample_indices(
                record) if self.random_shift else self._get_val_indices(record)
        
        modality = self.modality[0]
        image_tmpl = self.image_tmpls[0]
        img_group = self._get_frames(record, image_tmpl, modality, segment_indices, skip_offsets)

        if self.visualize:
            raw_img_group = [np.array(image)[:,:,::-1] for image in img_group]
            raw_img_group = np.stack(raw_img_group, 0)
            raw_img_group = raw_img_group.reshape((self.num_segments, self.new_length, ) + raw_img_group.shape[1:])
            raw_img_group = raw_img_group.transpose((0, 4, 1, 2, 3))
            

        flip = True if np.random.rand() < self.flip_ratio else False
        if (self.img_scale_dict is not None and record.path in self.img_scale_dict):
            img_scale = self.img_scale_dict[record.path]
        else:
            img_scale = self.img_scale
        
        img_group = self.img_group_transform(
                img_group, img_scale, crop_history=None, flip=flip,
                keep_ratio=self.resize_keep_ratio, div_255=self.div_255,
                is_flow=True if modality == 'Flow' else False)
        ori_shape = (256, 340, 3)
        
        if self.input_format == 'NCTHW':
            img_group = img_group.reshape(
                (self.num_segments, self.new_length, ) + img_group.shape[1:])
            img_group = np.transpose(img_group, (0, 2, 1, 3, 4))
        
        
        if self.visualize:
            data = raw_img_group, img_group, np.array([record.label] * self.num_segments)
        else:
            data = img_group, np.array([[record.label]] * self.num_segments)

        return data



if __name__ == '__main__':
    import os
    from TPN.utils.config import cfg

    cfg.dataset.hmdb.visualize = True
    dataset = RawFramesDataset(**cfg.dataset.hmdb)

    reader = dataset.create_reader()

    for i, (raw_img_group, img_group, label) in enumerate(reader()):
        print(raw_img_group.shape)
        print(img_group.shape)
        print(label.shape)
        if i > 20:
            break
        
        save_dir = os.path.join('/home/aistudio/TPN/visualize/transform')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        show_img = img_group[0,:,0,:,:].transpose((1, 2, 0))
        show_img = (show_img * cfg.dataset.hmdb.img_norm_cfg.std + \
            cfg.dataset.hmdb.img_norm_cfg.mean) * 255.0
        show_img = show_img.astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, str(i) + '.jpg'), show_img)

        raw_img = raw_img_group[0,:,0,:,:].transpose((1, 2, 0))
        raw_img = raw_img.astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, 'raw_%d.jpg' % i), raw_img)







