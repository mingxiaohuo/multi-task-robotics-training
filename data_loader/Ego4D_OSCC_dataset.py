import os
import cv2
import sys
import json
import tqdm
import numpy as np
import pandas as pd
import random
sys.path.append('/mnt/hdd1/ego4d_proj/mingxiaohuo_ego4d/EgoVLP/')

from base.base_dataset import TextVideoDataset
from data_loader.transforms import init_transform_dict, init_video_transform_dict
#find the system variable in the system road
import torch

class ObjectStateChangeClassification(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            'train': 'fho_oscc-pnr_train.json',
            'val': 'fho_oscc-pnr_val.json',            # there is no test
            'test': 'fho_oscc-pnr_val.json'    # pnr_test_unannotated;  pnr_val; pnr_train
        }
        target_split_fp = split_files[self.split]
        #define self.split as train/val/test （）tuple []list {}dictionary with key and value
        with open(os.path.join(self.meta_dir, target_split_fp)) as f:
            anno_json = json.load(f)
        #load the corresponding annotation jason
        self.cfg_DATA_CLIPS_SAVE_PATH = '/rscratch/data/tianran/ego4d_data/v1/frames_jpeg'
        #where to save the clips
        self.cfg_DATA_NO_SC_SPLIT_PATH = '/rscratch/data/tianran/ego4d_data/v1/frames_jpeg_neg'
        self.cfg_DATA_SAMPLING_FPS = 2
        self.cfg_DATA_CLIP_LEN_SEC = 8
        self.num_frames = self.cfg_DATA_SAMPLING_FPS * self.cfg_DATA_CLIP_LEN_SEC
        #16 frames in every clip
        self.metadata = pd.DataFrame(columns=['unique_id', 'video_id','clip_uid','clip_id',
                                              'pnr_frame', 'parent_pnr_frame', 'state',
                                              'clip_start_sec', 'clip_end_sec',
                                              'parent_start_sec', 'parent_end_sec',
                                              'clip_start_frame',  'clip_end_frame',
                                              'parent_start_frame', 'parent_end_frame'])
        #create the columns label
        clip_count = 0
        positive_count = 0
        negative_count = 0

        for i, data in enumerate(tqdm.tqdm(anno_json["clips"][:1000])):
            #tqdm：visualize the proceed of loading the json clips
            try:
                state_change = 1 if data['state_change'] else 0
            except:
                state_change=  0
            new = pd.DataFrame({
                "unique_id": data['unique_id'],
                "video_id": data['video_uid'],
                "clip_uid": data['clip_uid'],
                "clip_id": data['clip_id'],
                "pnr_frame": data['clip_pnr_frame'],
                "parent_pnr_frame": data['parent_pnr_frame']  if state_change == 1 else False,
                "state": state_change,
                "clip_start_sec":  data['clip_start_sec']  if state_change == 1 else False,
                "clip_end_sec":  data['clip_end_sec']  if state_change == 1 else False,
                "parent_start_sec": data['parent_start_sec'],
                "parent_end_sec": data['parent_end_sec'],
                "clip_start_frame": data['clip_start_frame']  if state_change == 1 else False,
                "clip_end_frame": data['clip_end_frame']  if state_change == 1 else False,
                "parent_start_frame": data['parent_start_frame'],
                "parent_end_frame": data['parent_end_frame'],
            }, index=[1])#when creating a dataframe, when using a dictionary, the index is used to allocate the rows label
            if state_change == 1: positive_count += 1;
            else: negative_count += 1;
            clip_count += 1

            self.metadata = self.metadata.append(new, ignore_index=True)
            #ignore_index=True, still continue the remain number of index,0,1,2,...
        print('Number of clips for', self.split, len(self.metadata))
        print(f"{clip_count} clip_countl, {positive_count} positive count, {negative_count} negative count")

        self.transforms = init_video_transform_dict()[self.split]
        #self.split train,test,val
    def _get_video_path(self, sample):
        rel_video_fp = sample[2] + '.mp4'
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        return sample[6]

    def _load_frame(self, frame_path):
        """
        This method is used to read a frame and do some pre-processing.

        Args:
            frame_path (str): Path to the frame

        Returns:
            frames (ndarray): Image as a numpy array
        """
        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0).astype(np.float32)
        return frame

    def _sample_frames(
            self,
            unique_id,
            clip_start_frame,
            clip_end_frame,
            num_frames_required,
            pnr_frame,#object state change frame
            info
    ):
        num_frames = clip_end_frame - clip_start_frame
        if num_frames < num_frames_required:
            pass
            # print(f'Issue: {unique_id}; {num_frames}; {num_frames_required}')
        error_message = "Can\'t sample more frames than there are in the video"
        assert num_frames >= num_frames_required, error_message
        lower_lim = np.floor(num_frames / num_frames_required)
        #get the lower int, every lower_lim gets a frame
        upper_lim = np.ceil(num_frames / num_frames_required)
        #get the upper int
        lower_frames = list()
        upper_frames = list()
        lower_keyframe_candidates_list = list()
        upper_keyframe_candidates_list = list()
        for frame_count in range(clip_start_frame, clip_end_frame, 1):
            if frame_count % lower_lim == 0:
                lower_frames.append(frame_count)
                if pnr_frame:
                    lower_keyframe_candidates_list.append(
                        np.abs(frame_count - pnr_frame)
                    )
                else:
                    lower_keyframe_candidates_list.append(0.0)
            if frame_count % upper_lim == 0:
                upper_frames.append(frame_count)
                if pnr_frame:
                    upper_keyframe_candidates_list.append(
                        np.abs(frame_count - pnr_frame)
                    )   
                else:
                    upper_keyframe_candidates_list.append(0.0)
        if len(upper_frames) < num_frames_required:
            return (
                lower_frames[:num_frames_required],
                lower_keyframe_candidates_list[:num_frames_required]
            )
        return (
            upper_frames[:num_frames_required],
            upper_keyframe_candidates_list[:num_frames_required]
        )

    def _sample_frames_gen_labels(self, info):
        # type(info)
        # exit(0)
        video_fp, rel_fp = self._get_video_path(info)
        if info['pnr_frame']:
            clip_path = os.path.join(
                self.cfg_DATA_CLIPS_SAVE_PATH,
                info['unique_id']
            )
        else:
            # Clip path for clips with no state change
            clip_path = os.path.join(
                self.cfg_DATA_NO_SC_SPLIT_PATH,
                info['unique_id']
            )
        message = f'Clip path {clip_path} does not exists...'
       # assert os.path.isdir(clip_path), message
        num_frames_per_video = (
                self.cfg_DATA_SAMPLING_FPS * self.cfg_DATA_CLIP_LEN_SEC
        )

        pnr_frame = info['parent_pnr_frame']
        if self.split == 'train':
            random_length_seconds = np.random.uniform(5, 8)
            #a random digit from 5 to 8
            random_start_seconds = info['parent_start_sec'] + np.random.uniform(
                8 - random_length_seconds
            ) #from parent start second+(0,8-length)
            random_start_frame = np.floor(
                random_start_seconds * 30
            ).astype(np.int32)
            #every 30 secs a frame
            random_end_seconds = random_start_seconds + random_length_seconds
            if random_end_seconds > info['parent_end_sec']:
                random_end_seconds = info['parent_end_sec']
            random_end_frame = np.floor(
                random_end_seconds * 30
            ).astype(np.int32)

            if pnr_frame:
                keyframe_after_end = pnr_frame > random_end_frame
                keyframe_before_start = pnr_frame < random_start_frame
                if keyframe_after_end:
                    random_end_frame = info['parent_end_frame']
                if keyframe_before_start:
                    random_start_frame = info['parent_start_frame']
                    #try to include the keyframe, by adjust the sample to the parent period

        elif self.split in ['test', 'val']:
            random_start_frame = info['parent_start_frame']
            random_end_frame = info['parent_end_frame']
            #val and test, do not need sample
        if pnr_frame:
            message = (f'Random start frame {random_start_frame} Random end '
                       f'frame {random_end_frame} info {info} clip path {clip_path}')#f:format
            assert random_start_frame <= pnr_frame <= random_end_frame, message
        else:
            message = (f'Random start frame {random_start_frame} Random end '
                       f'frame {random_end_frame} info {info} clip path {clip_path}')
            assert random_start_frame < random_end_frame, message

        candidate_frame_nums, keyframe_candidates_list = self._sample_frames(
            info['unique_id'],
            random_start_frame,
            random_end_frame,
            num_frames_per_video,
            pnr_frame,
            info
        )
        frames = list()
        #if os.path.isfile(video_fp):
        cap = cv2.VideoCapture(video_fp)
        assert (cap.isOpened())
        for frame_num in candidate_frame_nums:
            # frame_path = os.path.join(clip_path, f'{frame_num}.jpeg')
            # message = f'{frame_path}; {candidate_frame_nums}'
            # assert os.path.isfile(frame_path), message
          #if os.path.isfile(video_fp):
            #imgs, idxs = self.video_reader(video_fp, frame_num)
            #frames.append(self._load_frame(frame_path))
            # cap = cv2.VideoCapture(video_fp)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
        #read the index th frame by pos frames
            ret, frame = cap.read()
        #ret is bool variable
            if ret:
              frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              frame = torch.from_numpy(frame)
            #convert to tensor
            # (H x W x C) to (C x H x W)
              frame = frame.permute(2, 0, 1)
              frames.append(frame)
            else:
               pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')
        frames = torch.stack(frames).float() / 255
    #most cv mission makes the frames normalize to (0,1)
        cap.release()
        
  
    #tell the system exit capture the video
        if pnr_frame:
            keyframe_location = np.argmin(keyframe_candidates_list)
            hard_labels = np.zeros(len(candidate_frame_nums))
            hard_labels[keyframe_location] = 1
            labels = hard_labels
        else:
            labels = keyframe_candidates_list
        # Calculating the effective fps. In other words, the fps after sampling
        # changes when we are randomly clipping and varying the duration of the
        # clip
        final_clip_length = (random_end_frame / 30) - (random_start_frame / 30)
        effective_fps = num_frames_per_video / final_clip_length
        return np.concatenate(frames), np.array(labels), effective_fps

    def __getitem__(self, item):
        item = item
        sample = self.metadata.iloc[item]
        if(np.random.random()>0.5):
         state = sample['state']
         video_fp, rel_fp = self._get_video_path(sample) 
         if os.path.isfile(video_fp):       
         #imgs, labels, _ = self._sample_frames_gen_labels(sample)
          imgs, idxs = self.video_reader(video_path=video_fp, num_frames=self.num_frames, sample='uniform')
         #imgs = torch.as_tensor(imgs).permute(0, 3, 1, 2) / 255

        #   clip_len = sample['parent_end_sec'] - sample['parent_start_sec']
        #   clip_frame = sample['parent_end_frame'] - sample['parent_start_frame'] + 1
        #   fps = clip_frame / clip_len

          if self.transforms is not None:
            # if self.video_params['num_frames'] > 1:
            if self.num_frames > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)

          final = torch.zeros([self.num_frames, 3, self.video_params['input_res'],
                             self.video_params['input_res']])
          final[:imgs.shape[0]] = imgs
        #   labels= sample[4]
        else:
           state=0
           video_fp, rel_fp = self._get_video_path(sample)
           pnr=sample[4]+10
           if os.path.isfile(video_fp):
            imgs, idxs = self.video_reader(video_path=video_fp, num_frames=self.num_frames,sample='uniform',fix_start=pnr)
         #imgs = torch.as_tensor(imgs).permute(0, 3, 1, 2) / 255
             
           if self.transforms is not None:
            # if self.video_params['num_frames'] > 1:
            if self.num_frames > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)

           final = torch.zeros([self.num_frames, 3, self.video_params['input_res'],
                             self.video_params['input_res']])
           final[:imgs.shape[0]] = imgs
             
        data = {'video': final, 
                'state': state,
                #'idxs' :idxs,
                'parent_pnr_frame': sample['parent_pnr_frame'],
                'unique_id': sample['unique_id'],
                'clip_uid':sample['clip_uid']}
        return data
       
if __name__ == "__main__":
    split = 'train'
    kwargs = dict(
        dataset_name="Ego4D_OSCC",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 16,
        },
        data_dir="/mnt/hdd1/ego4d_proj/ego4d_data/v1/clips",
        meta_dir="/mnt/hdd1/ego4d_proj/ego4d_data/v1/annotations",
        tsfms=init_video_transform_dict()[split],
        reader='cv2',
        split=split
    )
    dataset = ObjectStateChangeClassification(**kwargs)
    for i in range(0,4):
        item = dataset[i]
        print(item.values())