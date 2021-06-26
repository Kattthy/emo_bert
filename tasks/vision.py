import os, glob
import cv2
import numpy as np
import tensorflow as tf
import collections
# from multiprocessing import Process

from utils import get_basename, mkdir
from .base_worker import BaseWorker

from tools.denseface.vision_network.models.dense_net import DenseNet

from tools.VAD import VAD
from FileOps import read_csv
import math

import time


class Video2Frame(BaseWorker):
    ''' 把视频帧抽出来存在文件夹中
        eg: 输入视频/root/hahah/0.mp4, save_root='./test/frame'
            输出视频帧位置: ./test/frame/hahah/0 (注意第29行, 按需求可修改)
    '''
    def __init__(self, fps=10, save_root='./test', logger=None):
        super().__init__(logger=logger)
        self.fps = fps
        self.save_root = save_root
    
    def __call__(self, video_path):
        basename = get_basename(video_path)
        basename = os.path.join(video_path.split('/')[-2], basename) # video_clip name xxxxx/1.mkv
        save_dir = os.path.join(self.save_root, basename)
        mkdir(save_dir)
        cmd = 'ffmpeg -i {} -r {} -q:v 2 -f image2 {}/'.format(video_path, self.fps, save_dir) + '%4d.jpg' + " > /dev/null 2>&1" #-r设置帧率，-q:v 图像质量, 2为保存为高质量，-f输出格式
        os.system(cmd)
        frames_count = len(glob.glob(os.path.join(save_dir, '*.jpg')))
        # self.print('Extract frames from {}, totally {} frames, save to {}'.format(video_path, frames_count, save_dir))
        return save_dir

class VideoFaceTracker(BaseWorker):
    ''' 使用openface工具抽取人脸
        eg: 输入视频帧位置: "./test/frame/hahah/0", 人脸图片位置save_root = './test/face'
            输出人脸位置: "./test/face/hahah/0 (注意第51行, 按需求可修改)
            其中./test/face/hahah/0/0_aligned文件夹中包含人脸图片
            ./test/face/hahah/0/0.csv中包含人脸关键点和AU等信息
    '''
    def __init__(self, save_root='test/track',
            openface_dir='/root/tools/OpenFace/build/bin', logger=None):
        #openface_dir='/root/tools/openface_tool/OpenFace/build/bin', logger=None):
        super().__init__(logger=logger)
        self.save_root = save_root
        self.openface_dir = openface_dir
    
    def __call__(self, frames_dir):
        basename = get_basename(frames_dir)
        basename = os.path.join(frames_dir.split('/')[-2], basename)
        save_dir = os.path.join(self.save_root, basename)
        mkdir(save_dir)
        cmd = '{}/FaceLandmarkVidMulti -fdir {} -mask -out_dir {} > /dev/null 2>&1'.format(
                    self.openface_dir, frames_dir, save_dir
                )
        os.system(cmd)
        # self.print('Face Track in {}, result save to {}'.format(frames_dir, save_dir))
        return save_dir

class DensefaceExtractor(BaseWorker):
    ''' 抽取denseface特征, mean是数据集图片的均值, std是数据集图片的方差(灰度图)
        device表示使用第几块gpu(从0开始计数), smooth=True则在一个视频序列中, 可能有些帧检测不到人脸, 对于这些缺失的帧的特征采用上一个有人脸的帧的特征填充, 通常smooth=False
    '''
    def __init__(self, mean=96.3801, std=53.615868, device=0, smooth=False, logger=None):
        """ extract densenet feature
            Parameters:
            ------------------------
            model: model class returned by function 'load_model'
        """
        super().__init__(logger=logger)
        #restore_path = '/data2/zjm/tools/FER_models/denseface/DenseNet-BC_growth-rate12_depth100_FERPlus/model/epoch-200'
        restore_path = '/data8/hzp/emo_bert/tools/denseface/pretrained_model/model/epoch-200'

        self.model = self.load_model(restore_path)
        self.mean = mean
        self.std = std
        self.previous_img = None        # smooth 的情况下, 如果没有人脸则用上一张人脸填充
        self.previous_img_path = None
        self.smooth = smooth
        self.dim = 342                  # returned feature dim
        self.device = device
    
    def load_model(self, restore_path):
        self.print("Initialize the model..")
        # fake data_provider
        growth_rate = 12
        img_size = 64
        depth = 100
        total_blocks = 3
        reduction = 0.5
        keep_prob = 1.0
        bc_mode = True
        model_path = restore_path
        dataset = 'FER+'
        num_class = 8

        DataProvider = collections.namedtuple('DataProvider', ['data_shape', 'n_classes'])
        data_provider = DataProvider(data_shape=(img_size, img_size, 1), n_classes=num_class)
        model = DenseNet(data_provider=data_provider, growth_rate=growth_rate, depth=depth,
                        total_blocks=total_blocks, keep_prob=keep_prob, reduction=reduction,
                        bc_mode=bc_mode, dataset=dataset)

        end_points = model.end_points
        model.saver.restore(model.sess, model_path)
        self.print("Successfully load model from model path: {}".format(model_path))
        return model
    
    def __call__(self, img_path):
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (64, 64))
            if self.smooth:
                self.previous_img = img
                self.previous_img_path = img_path

        elif self.smooth and self.previous_img is not None:
            # print('Path {} does not exists. Use previous img: {}'.format(img_path, self.previous_img_path))
            img = self.previous_img
        
        else:
            feat = np.zeros([1, self.dim]) # smooth的话第一张就是黑图的话就直接返回0特征, 不smooth缺图就返回0
            pred = np.zeros([1, 8])
            return feat, pred
        
        img = (img - self.mean) / self.std
        #print(img.shape)
        img = np.expand_dims(img, 2) # channel = 1
        img = np.expand_dims(img, 0) # batch_size=1
        with tf.device('/gpu:{}'.format(self.device)):
            feed_dict = {
                self.model.images: img,
                self.model.is_training: False
            }
            ft = self.model.sess.run(self.model.end_points['fc'], feed_dict=feed_dict)
            pred = self.model.sess.run(self.model.end_points['preds'], feed_dict=feed_dict)
            # ['neu', 'hap', 'sur', 'sad', 'ang', 'dis', 'fea', 'con']
        return ft, pred

class ActiveSpeakerSelector(BaseWorker):
    ''' 根据语音和landmark检测哪个人在说话, 输入videotracker输出的文件夹和对应的语音文件, 输出说话人id
    '''
    def __init__(self, diff_threshold=4, logger=None):
        super().__init__(logger=logger)
        self.diff_threshold = diff_threshold
    
    def get_clean_landmark(self, landmark_result_dir): #返回值：{faceId : [{frameId : 帧ID, score : 置信度, landmark : landmark坐标列表}, {}, ...] }
        '''
        :param alignment_landmark_result_filepath, FaceLandmarkVidMulti 的结果
        set faceId as personId, {'p1': [{'frameId':1, 'score':0.99, 'landmark':[[x1,y1], [x2,ye]]}, {'frameId':2}]}
        # b[:4] ['frame', ' face_id', ' timestamp', ' confidence']
        # b[299:435] 是 68 点 landmark 的坐标
        '''
        alignment_landmark_result_filepath = os.path.join(landmark_result_dir, #faces目录下的csv文件路径
                     get_basename(landmark_result_dir) + ".csv")
        if not os.path.exists(alignment_landmark_result_filepath):
            return None
        
        new_result = {}
        start = time.time()
        print(alignment_landmark_result_filepath)
        results = read_csv(alignment_landmark_result_filepath, delimiter=',', skip_rows=1) # 去掉 header
        end = time.time()
        print(end-start, 'read csv')

        start = time.time()
        for result in results:
            frameId, faceId, _, confidence = result[:4] #confidence：人脸landmark检测的置信度
            faceId = eval(faceId)
            raw_landmarks = result[299:435] #是68点landmark的坐标
            landmarks = []
            for i in range(68):
                landmarks.append([eval(raw_landmarks[i]), eval(raw_landmarks[i+68])])
            assert len(landmarks) == 68
            if new_result.get(faceId) is None:
                new_result[faceId] = []
            new_result[faceId].append({'frameId':eval(frameId), 'score':eval(confidence), 'landmark':landmarks})
        end = time.time()
        print(end-start, 'loop')
        return new_result
    
    def judge_mouth_open(self, landmark):
        '''
        第一个判断条件，嘴是否张开. 如果上下唇的高度大于唇的厚度，那么认为嘴是张开的
        param: 2Dlandmarks of one image, 人脸的2D关键点检测, 68 点的坐标
        '''
        is_open = False
        upperlip=landmark[62][1] - landmark[51][1] # 上唇的厚度
        height=landmark[66][1] - landmark[62][1] # 上下唇的高度
        if height > upperlip:
            is_open = True
        return is_open

    def get_mouth_open_score(self, landmarks): #返回值：嘴张开的次数，有效脸的个数
        '''
        如果某个人的嘴巴张开的次数作为 score, 注意过滤掉confidence小于0.5的脸。
        landmarks of all images of one person.
        '''
        count_open = 0
        count_valid = 0
        for idx in range(len(landmarks)):
            data = landmarks[idx] 
            if data['score'] > 0.5:
                count_valid += 1
                is_open = self.judge_mouth_open(data['landmark'])
                if is_open:
                    count_open += 1
        return count_open, count_valid
    
    def judge_continue_frames_change(self, landmark1, landmark2): #根据连续的两帧当中，内唇高度差和嘴角宽度差，判断是否发生明显的嘴部变化
        '''
        连续的两帧的人脸
        根据 62 66 计算内唇高度差，以及 60 64 计算嘴角宽度差，
        根据连续两帧的高度差和宽度差是否发生明显变化来判断.
        :param diff_threshold. = 4 的时候明显的嘴部变化，并且变化帧占总连续帧的25%左右。
        或者 diff_height_threshold > 2 or diff_width_threshold > 2
        '''
        is_change = False
        height1=landmark1[66][1] - landmark1[62][1] # 上下内唇的高度
        width1=landmark1[64][1] - landmark1[60][1] # 最后的内嘴角的宽度
        height2=landmark2[66][1] - landmark2[62][1] # 上下内唇的高度
        width2=landmark2[64][1] - landmark2[60][1] # 最后的内嘴角的宽度
        diff = abs(height2 - height1) + abs(width2 - width1)
        if diff > self.diff_threshold:
            is_change = True
        return is_change

    def get_continue_change_score(self, landmarks): #返回值：发生明显嘴部变化的数据数，有效的连续帧数据数，发生嘴部变化的连续帧的帧序号对列表
        '''
        第二个判断条件，嘴是否在动? 统计时间段内哪个人嘴部的动作最多。相对位置的变化，比如上唇和下唇的距离等.
        '''
        count_change = 0
        count_valid = 0
        change_frame_pairs = []
        for idx in range(len(landmarks)-1):
            data1 = landmarks[idx]
            data2 = landmarks[idx+1]
            frameId1 = data1['frameId']
            frameId2 = data2['frameId']
            if frameId2 - frameId1 <= 2:
                if data1['score'] > 0.5 and data2['score'] > 0.5:
                    count_valid += 1 #frameID1和frameID2的帧序号差不超过2，且两个帧的置信度都大于0.5，视作是一对有效的数据
                    is_change = self.judge_continue_frames_change(data1['landmark'], data2['landmark']) #根据连续的两帧当中，内唇高度差和嘴角宽度差，判断是否发生明显的嘴部变化
                    if is_change:
                        count_change += 1 #发生嘴部变化的数据
                        change_frame_pairs.append([frameId1, frameId2])
        return count_change, count_valid, change_frame_pairs
    
    def get_vad_face_match_score(self, wav_filepath, change_frame_pairs): #返回值：嘴部变化所在帧与vad label为1的帧重合的帧数量，vad label序列对应的总帧数
        '''
        audio=10ms/frame video=100ms/frame, 将video对齐到audio, 然后计算
        左右vad所在的帧，和 mouth_change 所在的帧, 的重合帧数。
        :param change_frameIds, [[frameId1, frameId2], ..., [frameId100, frameId101]]
        '''
        if len(change_frame_pairs) == 0: # 如果没有动作帧，那么
            return 0, 0
        vad = VAD()
        vad_fts = vad.gen_vad(wav_filepath)     # 获取每一帧的vad的label, 得到vad label序列
        total_frame = len(vad_fts)
        visual_frames = np.zeros_like(vad_fts, dtype=int)     # 构建 video change frames 对齐到 audio frame, 比如某帧
        for frame_pair in change_frame_pairs:
            frameId1, frameId2 = frame_pair
            # frameId start from 1, so frameId=1 到 frameId=2 其实0～100ms
            start_frame = (frameId1-1) * 10
            end_frame = (frameId2-1) * 10
            if end_frame > total_frame:
                end_frame = total_frame
            visual_frames[start_frame: end_frame] = 1 #对应于这段音频的帧范围
        count_match = 0
        for i in range(total_frame):
            if vad_fts[i] == visual_frames[i] == 1:
                count_match += 1
        return count_match, total_frame
    
    def get_ladder_rank_score(self, raw_person2scores): #返回字典：{personID : ladder rank 分数}
        '''
        rank score = 1 / sqrt(rank), ladder的含义是得分相同 的 rank 得分也一样
        raw_person2scores: {0:s1, 1:s2, 2:s3}
        '''
        sort_person2score_tups = sorted(raw_person2scores.items(), key=lambda asd: asd[1], reverse=True) #将各个字典项按值降序排列
        rank_ind = 1
        person2rank_score = {}
        previous_score = sort_person2score_tups[0][1] #previous_score：当前是值最大的字典项的值
        for tup in sort_person2score_tups:
            cur_person, cur_score = tup
            if cur_score == previous_score:
                rank_score = 1 / math.sqrt(rank_ind)
                rank_ind += 0
            else:
                rank_ind += 1
                rank_score = 1 / math.sqrt(rank_ind)
            person2rank_score[cur_person] = rank_score
            previous_score = cur_score
        return person2rank_score
    
    def get_final_decision(self, open_person2scores, change_person2scores, 
                match_person2scores):  #返回最终分数最高的那个人的personID
        '''
        条件1: 如果三个得分中有一个是0, 即如果张嘴次数是0 或者 动作次数是0 或者 match次数是0，那么该人不是说话人。
        条件2: 如果只剩下一个人, 那么直接该人的ID. 
        条件3: 如果最后没有合适的人，那么该视频丢弃. return None
        条件4: 如果正常的多个人，那么进行排序
        open_person2scores: {p1:s1, p2:s2, p3:s3}
        '''
        # 条件1
        persons = list(open_person2scores.keys())
        for person in persons:
            if open_person2scores[person] == 0 or change_person2scores[person] == 0 \
                    or match_person2scores[person] == 0:
                open_person2scores.pop(person)
                change_person2scores.pop(person)
                match_person2scores.pop(person)
        # 条件2 & 条件3
        if len(open_person2scores) == 0:
            return None
        if len(open_person2scores) == 1:
            return list(open_person2scores.keys())[0]
        # 条件4
        open_rank_scores = self.get_ladder_rank_score(open_person2scores)
        change_rank_scores = self.get_ladder_rank_score(change_person2scores)
        match_rank_scores = self.get_ladder_rank_score(match_person2scores)

        person2final_score = {}
        for person in open_rank_scores.keys():
            final_score = open_rank_scores[person] + change_rank_scores[person] + match_rank_scores[person] #最后每个人的分数就是三个分数加和
            person2final_score[person] = final_score
        sort_person2score_tups = sorted(person2final_score.items(), key=lambda asd: asd[1], reverse=True)
        return sort_person2score_tups[0][0] #返回分数最高的那个人的personID

    def __call__(self, face_dir, audio_path):
        '''
        通过 openface 的结果判断一个人是否在说话，很有可能不是屏幕中的人在说话。
        Q1: 得到次数之后如何计算得分？ rank_score = 1 / sqrt(rank)
        Q2: 三个得分的融合决策的策略是什么？ rank_score 直接相加。
        :param, landmarks_path, {'p1': [{'frameId':1, 'score':0.99, 'landmark':[[x1,y1], [x2,y2]]}, {'frameId':2}]}
        '''
        start = time.time()
        landmarks = self.get_clean_landmark(face_dir) #在上一步（抽取人脸）过程生成的csv文件中提取出想要的信息，得到的是一个人脸ID和对应信息的字典：{faceId : [{frameId : 帧ID, score : 置信度, landmark : landmark坐标列表}, {}, ...] }
        end = time.time()
        #print('f0', end-start)
        if landmarks is None:
            return None
        
        open_person2scores = {}
        change_person2scores = {}
        match_person2scores = {}
        for person in landmarks.keys(): #landmarks.keys：faceID
            person_frames = landmarks[person] #person_frames：人脸对应信息（帧ID、置信度、landmark坐标）
            start = time.time()
            count_open, count_valid = self.get_mouth_open_score(person_frames) #count_open：嘴巴张开的个数；count_valid：有效脸（置信度>0.5）的个数
            end = time.time()
            #print('f1', end-start)
            # self.print('person{} open: {}/{}'.format(person, count_open, count_valid))
            start = time.time()
            count_change, count_valid, change_frame_pairs = self.get_continue_change_score(person_frames) #count_change：发生明显嘴部变化的数据数；count_valid：有效的连续帧数据数；change_frame_pairs：发生嘴部变化的连续帧的帧序号对列表
            end = time.time()
            #print('f2', end-start)
            # self.print('person{} change: {}/{}'.format(person, count_change, count_valid))
            start = time.time()
            count_match, total_frame = self.get_vad_face_match_score(audio_path, change_frame_pairs) #count_match：嘴部变化所在帧与vad label为1的帧重合的帧数量；total_frame：vad label序列对应的总帧数
            end = time.time()
            #print('f3', end-start)
            # self.print('person{} match: {}/{}'.format(person, count_match, total_frame))
            open_person2scores[person] = count_open
            change_person2scores[person] = count_change
            match_person2scores[person] = count_match

        active_speaker = self.get_final_decision(open_person2scores, change_person2scores, match_person2scores) #计算所有人的最终分数，并返回最终分数最高的那个人的personID，将其视作active speaker
        return active_speaker
    
class FaceSelector(BaseWorker):
    def __init__(self, logger=None):
        super().__init__(logger=logger)
    
    def __call__(self, face_dir, active_spk_id):
        if active_spk_id == None:
            return []
        basename = get_basename(face_dir)
        face_img_dir = os.path.join(face_dir, basename + '_aligned')
        face_list = glob.glob(os.path.join(face_img_dir, f'*_det_{active_spk_id:02d}_*.bmp'))
        face_list = sorted(face_list, key=lambda x:int(x.split('/')[-1].split('.')[0].split('_')[-1]))
        return face_list

if __name__ == '__main__':
    # get_frame = Video2Frame()
    # frame_dir = get_frame('../resources/output1.mkv')
    # face_track = VideoFaceTracker()
    # a = face_track(frame_dir)
    # print(a)

    face_path = '/data6/zjm/emobert/preprocess/test/track/resources/output1/output1_aligned/frame_det_00_000001.bmp'
    mean = 96.3801
    std = 53.615868
    get_denseface = DensefaceExtractor(mean=mean, std=std, device=0)
    feature, pred = get_denseface(face_path)
    print(feature.shape)
    print(pred.shape)
    print(pred)

    # import time
    # select_activate_spk = ActiveSpeakerSelector()
    # # select_faces = FaceSelector()
    # start = time.time()
    # active_spkid = select_activate_spk("/data6/zjm/emobert/preprocess/data/faces/No0001.The.Shawshank.Redemption/18", "/data6/zjm/emobert/preprocess/data/audio_clips/No0001.The.Shawshank.Redemption/18.wav")
    # # face_lists = select_faces("test/track/output1", active_spkid)
    # end = time.time()
    # print("Total:", end-start)
    # # print(active_spkid)
    # # print(face_lists)

    # import pandas as pd
    # start = time.time()
    # pd.read_csv("/data6/zjm/emobert/preprocess/data/faces/No0001.The.Shawshank.Redemption/18/18.csv")
    # end = time.time()
    # print('pd read_csv', end-start)

    # start = time.time()
    # read_csv("/data6/lrc/18.csv", delimiter=',', skip_rows=1)
    # end = time.time()
    # print('read_csv', end-start)
    