from tasks.vision import *
from utils import get_basename, mkdir
from tqdm import tqdm
import os
import numpy as np
import math
from functools import total_ordering
import json
from collections import deque, Counter
from shutil import copyfile, copytree
from sys import exit
import sys
from tasks.text import TranscriptPackager
import csv
import liwc
import re

# 加载情绪词典
parse, category_names = liwc.load_token_parser('/data8/hzp/emo_bert/LIWC2015Dictionary.dic')

def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)


run_denseface = DensefaceExtractor() #加载denseface模型

emo_class_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disgust', 6: 'fear', 7: 'contempt'}

def pred2result(pred): #根据识别模型输出的各类的概率，给出模型判定的情感类别名称
    return emo_class_dict[np.argmax(np.array(pred))]

def num2idstr(num): #将整数数字转化为六位字符串，位数不够前面补0
    return str(num).zfill(6)

def copy(source, target):
    try:
        copyfile(source, target)
    except IOError as e:
        print('Unable to copy file. %s' % e)
        exit(1)
    except:
        print('Unexpected error:', sys.exc_info())
        exit(1)



@total_ordering
class EmotionSegment(object):
    '''
    用于记录连续情感帧的数据结构
    注：startId，endId为int类型，emotion均为str类型。length是指这个片段中实际的帧数量，而不是endId-startId（因为中间的序号不一定连续）。
    frames为片段中帧的列表
    排序规则：按length排，大的在后
    '''
    def __init__(self, frames, emo):
        self.frames = frames
        self.emotion = emo

    def __len__(self):
        return len(self.frames)

    def __eq__(self, other):
        return len(self.frames)==len(other.frames)

    def __gt__(self, other):
        return len(self.frames)>len(other.frames)



class AnnotateEmotion(BaseWorker):
    '''
    对给定的电影通过denseface预测结果得到情感伪标注并写入文件
    '''
    def __init__(self, source_root, frame_root, audio_root, save_root, logger=None):
        '''
        source_root举例：/data1/hzp/emo_bert/data/faces
        save_root举例：/data1/hzp/emo_bert/data/pseudo_annotations_v3
        '''
        super().__init__(logger=logger)
        self.source_root = source_root
        self.save_root = save_root
        self.frame_root = frame_root
        self.audio_root = audio_root

    def choose_active_speaker_face(self, frame_dir, face_dir, audio_path):
        """
        输入：某个clip的帧目录、脸目录、音频目录
        1.找该片段的有效说话人
        2.如果存在有效说话人，则判断该说话人的人脸数量是否足够
        返回值：是否能找到有效说话人且该人的人脸是否足够（布尔值）、有效说话人id、有效说话人的所有人脸图像的帧的路径列表
        """
        get_activate_spk = ActiveSpeakerSelector()
        select_faces = FaceSelector()

        active_spk_id = get_activate_spk(face_dir, audio_path) #找到有效说话人id
        if active_spk_id is None: #没有有效说话人的情况
            return False, None, []
        else:
            face_paths = select_faces(face_dir, active_spk_id) #在faces/....../xx_aligned目录下，找出所有active_spk_id对应的人的脸的图像，并将这些图像的目录按该目录下的图像序号排好序。返回指定人的有序的脸部图像目录列表
            if len(face_paths) < 0.4 * len(glob.glob(os.path.join(frame_dir, '*.jpg'))): #有效说话人的人脸不足的情况
                return False, active_spk_id, face_paths
            else:
                return True, active_spk_id, face_paths
    
    def faceID2num(self, faceID):
        '''
        faceID: frame_det_00_000001
        num:1
        '''
        return int(faceID.split('_')[-1])


    def __call__(self, valid_item):
        '''
        valid_item是check_data目录下的txt文档记录的每一项，如：No0001.The.Shawshank.Redemption/139
        '''
        save_dir = os.path.join(self.save_root, valid_item)
        #mkdir(save_dir) #建立对应片段的目录

        valid_id = valid_item.split('/')[-1] #str类型
        frame_dir = os.path.join(self.frame_root, valid_item)
        face_dir = os.path.join(self.source_root, valid_item)
        audio_path = os.path.join(self.audio_root, valid_item+'.wav')
        
        success, active_spk_id, faceFrame_list = self.choose_active_speaker_face(frame_dir, face_dir, audio_path)
        faceFrame_list = sorted(faceFrame_list)

        #record_file_path = os.path.join(save_dir, 'pseudo_anno_' + valid_id + '.txt') #记录该片段的有效说话人信息和每一帧的伪标注
        #continuous_anno_path = os.path.join(save_dir, 'continuous_anno_' + valid_id + '.txt') #记录该片段中的连续情感帧伪标注
        
        denseface_result = {} #{clip_id的值:[{'ID':frame_id,'pred':[],'emo':''},{}]}
        continuous_result = {} #{clip_id, frames:[], denseface_emo}


        #dict_id_emo = {} #每个faceFrame的序号与该帧的伪标注的类别对应关系：{faceFrameId（int类型） : emotion（str类型）}

        if success==False: #无说话人或者说话人对应的人脸过少的情况
            return None, None
        else:
            #得到该clip的denseface_result：
            denseface_faceInfo_list = []
            for faceFrame in faceFrame_list:
                denseface_faceInfo = {}

                personID = get_basename(faceFrame).split('_')[2]
                faceFrameID = get_basename(faceFrame).split('_')[3].split('.')[0]
                ft, pred = run_denseface(faceFrame)

                #记录并存放中间数据
                #intermediate_data_basename = os.path.join(save_dir, 'pseudo_anno_' + personID + '_' + faceFrameID) #存放中间数据（即denseface模型识别每一帧人脸得出的特征和输出）
                #np.save(intermediate_data_basename + '_ft.npy', ft)
                #np.save(intermediate_data_basename + '_pred.npy', pred)

                denseface_faceInfo['ID'] = get_basename(faceFrame)
                denseface_faceInfo['pred'] = np.squeeze(pred).tolist() #json文件无法直接输出ndarray，需要转换成普通列表
                denseface_faceInfo['emo'] = pred2result(pred)
                denseface_faceInfo_list.append(denseface_faceInfo) #添加一帧信息

                #dict_id_emo[int(faceFrameID)] = pred2result(pred) #将faceFrameID转换为int类型作为词典的key

            denseface_result[valid_id] = denseface_faceInfo_list




            #得到连续情感帧的过程
            #faceFrameIdList = sorted(dict_id_emo.keys()) #得到一个排序后的faceFrame序号列表

            #------------------------------------------
            #一、如果某两个序号之间的差>2，则在该处分为两段。得到队列 paraQueue：[段1，...，段n]。其中元素“段m”是该段中所有denseface_faceInfo的列表
            #    如果某段长度小于10则直接丢弃
            #------------------------------------------
            paraQueue = []
            paraItem = [denseface_faceInfo_list[0]]
            for i in range(len(denseface_faceInfo_list) - 1):
                if self.faceID2num(denseface_faceInfo_list[i+1]['ID']) - self.faceID2num(denseface_faceInfo_list[i]['ID']) <= 2:
                    paraItem.append(denseface_faceInfo_list[i+1])
                else:
                    if len(paraItem) >= 10: #如果某段长度小于10则直接丢弃
                        paraQueue.append(paraItem)
                    paraItem = [denseface_faceInfo_list[i+1]]
            if len(paraItem) >= 10: #最后一段也要考虑加入队列
                paraQueue.append(paraItem) #最后一段也要加入队列

            selected = [] #同一个视频clip的所有段最终选出的片段列表（每一段最多选出一个片段）

            for para in paraQueue:
                #------------------------------------------
                #二、
                #在每段中，统计除neutral外每个情感类别的帧数量。如果最高频的情感帧数量>=3，则选出该最高频情感（记为HF_emotion，简记为E）
                #（如果最高频有多个相同数量的情感类别就随便选一个）
                #如果最高频的情感帧数量<3则不考虑情感帧（跳过第四步中的平滑部分以及第五、六步）直接考虑neutral（第七步）。
                #------------------------------------------

                #对段para列表中的每个元素（帧）做情感类别的统计
                count_emotion = {} #记录各情感类别的帧数量
                for i in range(1, 8): #初始化
                    count_emotion[emo_class_dict[i]] = 0
                for item in para:
                    if item['emo'] != 'neutral': #是情感类别帧
                        count_emotion[item['emo']] += 1

                emotion_sort = sorted(count_emotion.items(), key = lambda kv:(kv[1], kv[0]), reverse=True) #降序。排序时如果两个情感的帧数量相同，就按情感名决定先后
                
                if emotion_sort[0][1] >= 3:
                    HF_emotion = emotion_sort[0][0] #最高频情感的名称
                else:
                    HF_emotion = 0


                #------------------------------------------
                #三、
                #每一段中连续的相同情感的帧划分为一个连续情感片段EmotionSegment。
                #连续情感片段数据结构：{帧列表, 情感类别}
                #得到栈 segStack：[片段n，片段n-1，...，片段1]，需确保最前面的片段在栈顶
                #------------------------------------------
                frameList = [para[0]['ID']]
                segList = []
                for i in range(len(para) - 1):
                    if para[i+1]['emo'] != para[i]['emo']: #前后两帧情感不相同
                        emoSeg = EmotionSegment(frameList, para[i]['emo'])
                        segList.append(emoSeg)
                        frameList = [para[i+1]['ID']]
                    else:
                        frameList.append(para[i+1]['ID'])
                emoSeg = EmotionSegment(frameList, para[-1]['emo']) #最后一段也要加入
                segList.append(emoSeg)

                segStack = list(reversed(segList)) #使最前面的片段在栈顶

                #------------------------------------------
                #四、
                #检查所有片段。如果有长度大于等于10的neutral片段，将它加入到一个列表中。如果有平滑之后长度大于等于10的E情感片段，将它加入到另一个列表中。
                #可能需平滑的情况：连续的3个片段的情感分别是 E, b, E（即只考虑对最高频情感的平滑）
                #平滑规则：
                #    1）若 b 是neutral，则 b.length<=4 的情况下将其与前后两个 a 片段合并；
                #    2）若 b 片段是其他情感，则 b.length<=2 的情况下将其与前后两个 a 片段合并
                #------------------------------------------
                neutral_list = [] #存放该段中所有选出的neutral片段
                HF_emotion_list = [] #存放该段中所有选出的平滑后的E情感 片段
                while len(segStack): #栈不空
                    top = segStack[-1]
                    segStack.pop()

                    if top.emotion == 'neutral':
                        if len(top) >= 10:
                            neutral_list.append(top)
                        continue
                    elif top.emotion != HF_emotion:
                        continue

                    #top是情感E的情况：

                    if len(segStack)==0: #栈空
                        if len(top) >= 10:
                            HF_emotion_list.append(top)
                        break

                    s_top = segStack[-1]
                    if s_top.emotion == 'neutral':
                        if len(s_top) > 4:
                            if len(top) >= 10:
                                HF_emotion_list.append(top)
                            continue
                    else:
                        if len(s_top) > 2:
                            if len(top) >= 10:
                                HF_emotion_list.append(top)
                            continue

                    segStack.pop() #s_top出栈
                    if len(segStack)==0: #栈空
                        if len(top) >= 10:
                            HF_emotion_list.append(top)
                        segStack.append(s_top) #s_top重新入栈
                        continue

                    t_top = segStack[-1]
                    if t_top.emotion == HF_emotion:
                        #合并 E b E 三个片段
                        segStack.pop() #t_top出栈
                        #merge = EmotionSegment(top.startId, t_top.endId, top.length + s_top.length + t_top.length, top.emotion)
                        merge_frames = sorted(top.frames + s_top.frames + t_top.frames)
                        merge = EmotionSegment(merge_frames, top.emotion)
                        segStack.append(merge) #合并后的片段重新入栈，继续与后两段检查是否可以平滑
                    else:
                        if len(top) >= 10:
                            HF_emotion_list.append(top)
                        segStack.append(s_top) #s_top重新入栈，继续与后两段检查是否可以平滑
                

                
                #------------------------------------------
                #五、
                #如果有选出的平滑后E情感片段，则挑出最长的那个。如果它的长度>=20，则把它两侧同时去掉一部分使得它最终的长度不超过20。
                #------------------------------------------
                if HF_emotion and len(HF_emotion_list):
                    selected_seg = sorted(HF_emotion_list)[-1]

                    '''
                    #裁剪
                    if len(selected_seg) >= 20:
                        selected_seg = self.cut(selected_seg)
                    '''

                    selected.append(selected_seg)


                #------------------------------------------
                #六、
                #如果平滑后E情感片段列表为空，则考虑使用窗口的方法选E情感片段。
                #窗口的想法：
                #    1、固定窗口的长度为10
                #    2、将窗口从这一段的头滑到尾，在每个位置上分别计算出两个量：
                #        ①E情感帧的数量 count_E
                #        ②每个E情感帧距离这10帧的中心的距离的平方 之和 sum_distance（比如第5帧是E那距离可以设为0.5²，第1帧是E距离可以设为4.5²）
                #    3、如果在某个位置上的count_E >= 4，则将其加入候选窗口列表
                #    4、在所有候选窗口中，选出count_E值最大的那个或那些
                #    5、如果第4步得到的窗口不唯一，再在这些窗口中选出sum_distance值最小的窗口。如果还有值相同的那就随便选一个。
                #    （这样做的目的是为了让窗口框住的部分中情感E帧尽可能都出现在中间而不是某一侧）
                #------------------------------------------
                if HF_emotion and len(HF_emotion_list)==0:
                    window_len = 10
                    candiWindow = [] #用于存放候选窗口的列表

                    assert len(para) >= 10
                    s_idx = 0
                    e_idx = window_len - 1
                    while e_idx < len(para):
                        count_E = 0
                        sum_distance = 0
                        window_frames = []
                        for i in range(s_idx, e_idx + 1):
                            window_frames.append(para[i]['ID'])
                            if para[i]['emo'] == HF_emotion:
                                count_E += 1
                                sum_distance += abs(i - s_idx - 4.5) * abs(i - s_idx - 4.5)
                        if count_E >= 4:
                            #win = Window(para[s_idx], para[e_idx], count_E, sum_distance)
                            win = Window(window_frames, count_E, sum_distance)
                            candiWindow.append(win) #加入候选窗口列表
                        s_idx += 1
                        e_idx += 1
                    
                    if len(candiWindow):
                        selected_win = sorted(candiWindow)[-1]
                        #selected.append(EmotionSegment(selected_win.startId, selected_win.endId, window_len, HF_emotion))
                        selected.append(EmotionSegment(selected_win.frames, HF_emotion))


                #------------------------------------------
                #七、
                #如果第五、六步无法选出任何E情感片段，则再考虑是否可选出neutral片段。
                #如果第四步得到的neutral片段列表不空，则挑出最长的那个。如果它的长度>=20，则把它两侧同时去掉一部分使得它最终的长度不超过20。
                #如果也没有选出的neutral片段，则丢弃这一段。
                #------------------------------------------
                if len(selected)==0:
                    if len(neutral_list):
                        selected_seg = sorted(neutral_list)[-1]

                        '''
                        #裁剪
                        if len(selected_seg) >= 20:
                            selected_seg = self.cut(selected_seg)
                        '''

                        selected.append(selected_seg)

                
            #输出文件：
            #with open(continuous_anno_path, 'w') as cf:
            #    cf.write("active_speaker_id:" + str(active_spk_id) + "\n")
            #    for item in selected:
            #        cf.write(str(item))

            if len(selected)==0:
                return denseface_result, None
            else:
                #------------------------------------------
                #八、
                #每个clip如果因有多段而选出多个片段，只选最长的那个片段
                #------------------------------------------
                final_seg = sorted(selected)[-1]
                continuous_result['clip_id'] = valid_id #str类型
                continuous_result['frames'] = final_seg.frames
                continuous_result['denseface_emo'] = final_seg.emotion
                return denseface_result, continuous_result


def load_transcript(transcripts_dir, valid_item):
    '''
    valid_item是check_data目录下的txt文档记录的每一项，如：No0001.The.Shawshank.Redemption/139
    返回当前句文本
    '''
    movie_name = valid_item.split('/')[0]
    transcript_id = int(valid_item.split('/')[-1])
    transcript_ass = os.path.join(transcripts_dir, movie_name + '.ass')
    transcript_srt = os.path.join(transcripts_dir, movie_name + '.srt')
    if os.path.exists(transcript_ass):
        transcript_path = transcript_ass
    else:
        transcript_path = transcript_srt
    package_transcript = TranscriptPackager()
    #返回一个字典列表，列表中的每一项是记录每条字幕信息的字典：
    #   {start: 开始时间，end: 结束时间，content: 处理后的文本内容，index: 字幕序号}。
    transcript_info = package_transcript(transcript_path)
    current_sentence = transcript_info[transcript_id]['content'].lower()
    return current_sentence

def roberta_predict(source_path):
    id2emo = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'fear'}
    roberta_root = '/data1/hzp/RoBERTa'
    roberta_data_dir = os.path.join(roberta_root, 'movie_data')
    bash_file_path = os.path.join(roberta_root, 'run_classifier_for_anno.sh')
    copy(source_path, os.path.join(roberta_data_dir, 'test.tsv'))
    #_cmd = 'bash {} > /dev/null 2>&1'
    _cmd = 'bash {}'
    os.system(_cmd.format(bash_file_path))
    pred_file = os.path.join(roberta_root, 'output', 'roberta_3', 'predicted.csv')

    with open(pred_file, "r") as f:
        reader = csv.reader(f)
        lines = []
        for line in reader:
            lines.append(id2emo[int(line[0])])

    return lines


def text_filter(continuous_result_list, text_pos_neg, final_result_list):
    '''
    用文本模态的情感词典和roberta结果辅助denseface结果得出最后的伪标注情感。
    通过continuous_result_list读入denseface和roberta预测结果，并将最终的伪标注情感记入它每个元素的'emo'项（如果无效则记为None）
    同时将所有最终有效的样例数据记入final_result_list

    策略：
    if (roberta预测emotion == denseface预测emotion)，则该样例有效
    elif (roberta预测emotion == neutral)，则该样例有效
    else:
        if (情感词'pos'为正)&&(denseface预测emotion==happiness)，则该样例有效
        elif (情感词'anger'为正)&&(denseface预测emotion==anger)，则该样例有效
        elif (情感词'sad'为正)&&(denseface预测emotion==sadness)，则该样例有效
        else 无效样例
    '''
    for item in continuous_result_list:
        denseface_pred = item['denseface_emo']
        roberta_pred = item['roberta_emo']
        transcript_id = int(item['clip_id'])
        final_emo = None
        if roberta_pred == denseface_pred:
            final_emo = denseface_pred
        elif roberta_pred == 'neutral':
            final_emo = denseface_pred
        else:
            if (text_pos_neg[transcript_id]['pos'] > 0) and (denseface_pred == 'happiness'):
                final_emo = denseface_pred
            elif (text_pos_neg[transcript_id]['anger'] > 0) and (denseface_pred == 'anger'):
                final_emo = denseface_pred
            elif (text_pos_neg[transcript_id]['sad'] > 0) and (denseface_pred == 'sadness'):
                final_emo = denseface_pred

        if final_emo == None:
            item.update({'emo': 'None'})
        else:
            item.update({'emo': final_emo})
            final_dict = {'clip_id': item['clip_id'], 'frames': item['frames'], 'emo': final_emo}
            final_result_list.append(final_dict)



if __name__ == '__main__':

    data_source = 'wjq' #hzp #用谁的预处理数据
    
    pseudo_annotation_dir = './data/pseudo_annotations_v4' #用于存放情感伪标注信息的根目录
    
    assert(data_source in ['hzp', 'wjq'])
    if data_source == 'hzp':
        face_root_dir = "./data/faces" #预处理过程提取出的脸部图像数据的根目录
        check_data_root_dir = "./data/check_data" #预处理过程记录抽脸信息的根目录
        frame_root_dir = './data/frames' #预处理过程抽出的帧的根目录
        audio_root_dir = './data/audio_clips' #预处理过程抽出的音频的根目录
        transcripts_dir = './data/transcripts' #电影字幕文件的根目录
    else:
        #wjq的数据
        face_root_dir = "/data1/wjq/data/faces" #预处理过程提取出的脸部图像数据的根目录
        check_data_root_dir = "/data1/wjq/data/check_data" #预处理过程记录抽脸信息的根目录
        frame_root_dir = '/data1/wjq/data/frames' #预处理过程抽出的帧的根目录
        audio_root_dir = '/data1/wjq/data/audio_clips' #预处理过程抽出的音频的根目录
        transcripts_dir = '/data1/wjq/raw_movies_10' #电影字幕文件的根目录


    mkdir(pseudo_annotation_dir)

    #情感伪标注
    get_emotion_annotation = AnnotateEmotion(source_root=face_root_dir, frame_root=frame_root_dir, audio_root=audio_root_dir, save_root=pseudo_annotation_dir)
    movie_folder_list = sorted(os.listdir(face_root_dir)) #获取所有电影文件夹列表



    #-----test----------
    #movie_folder_list = ['No0001.The.Shawshank.Redemption']
    #-------------------


    if data_source == 'wjq':
        transcripts_tmp = transcripts_dir

    for movie_name in movie_folder_list:
        if data_source == 'wjq':
            transcripts_dir = os.path.join(transcripts_tmp, movie_name)

        mkdir(os.path.join(pseudo_annotation_dir, movie_name))
        denseface_result_list = [] #该电影中所有clips的denseface识别结果
        continuous_result_list = [] #该电影中所有clips的伪标注结果。{clip_id, frames:[], denseface_emo, roberta_emo, emo}
        final_result_list = [] #视觉+文本筛选后所有有效的伪标注结果。{clip_id, frames:[], emo}
        text_pos_neg = {} #情感词典。{台词号:{pos, neg, anger, sad}}
        transcripts_list = [] #所有denseface伪标注有效的clip对应的句子

        print("Start processing {}:".format(movie_name))

        valid_face_text_file = os.path.join(check_data_root_dir, movie_name + ".txt")
        valid_list = []
        with open(valid_face_text_file, 'r') as vf:
            for line in vf.readlines():
                valid_list.append(line.replace('\n', ''))

        if not valid_list: #列表为空，说明这整部电影都没有任何脸部有效的帧
            print("该电影没有任何脸部有效的帧！")
            continue

        for valid_item in tqdm(valid_list):
            denseface_result, continuous_result = get_emotion_annotation(valid_item)
            if denseface_result != None:
                denseface_result_list.append(denseface_result)
            if continuous_result != None: #denseface伪标注存在
                continuous_result_list.append(continuous_result)
                
                #提取该句文本，并建立该句的情感词典
                current_sentence = load_transcript(transcripts_dir, valid_item)
                transcripts_list.append(current_sentence)
                tokens = tokenize(current_sentence)
                category_count = Counter(category for token in tokens for category in parse(token))
                transcript_id = int(valid_item.split('/')[-1])
                text_pos_neg[transcript_id] = {}
                text_pos_neg[transcript_id]['pos'] = category_count['posemo']
                text_pos_neg[transcript_id]['neg'] = category_count['negemo']
                text_pos_neg[transcript_id]['anger'] = category_count['anger']
                text_pos_neg[transcript_id]['sad'] = category_count['sad']


        #denseface对每一帧做情感预测的结果写json文件
        denseface_result_path = os.path.join(pseudo_annotation_dir, movie_name, 'denseface_result.json')
        with open(denseface_result_path,'w') as f:
            json.dump(denseface_result_list, f)

        #将该电影中所有denseface伪标注有效的clip对应的句子写入tsv文件
        transcripts_path = os.path.join(pseudo_annotation_dir, movie_name, 'denseface_valid_transcripts.tsv')
        with open(transcripts_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            for sentence in transcripts_list:
                writer.writerow([sentence]) 

        
        #得到roberta预测结果
        lines = roberta_predict(transcripts_path)
        assert(len(lines)==len(continuous_result_list))
        for dic, line in zip(continuous_result_list, lines):
            dic.update({'roberta_emo': line})
        

        #用文本模态的情感词典和roberta结果辅助denseface结果得出最后的伪标注情感。
        #通过continuous_result_list读入denseface和roberta预测结果，并将最终的伪标注情感记入它每个元素的'emo'项（如果无效则记为None）
        #同时将所有最终有效的样例数据记入final_result_list
        text_filter(continuous_result_list, text_pos_neg, final_result_list)


        #continuous_result_list和final_result_list写json文件
        continuous_result_path = os.path.join(pseudo_annotation_dir, movie_name, 'continuous_result.json')
        with open(continuous_result_path,'w') as f:
            json.dump(continuous_result_list, f)
        final_result_path = os.path.join(pseudo_annotation_dir, movie_name, 'final_result.json')
        with open(final_result_path,'w') as f:
            json.dump(final_result_list, f)
        

        print("{} is successfully finished.".format(movie_name))


            