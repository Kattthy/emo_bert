"""
在指定的n部电影中，分别从每部电影里随机挑出m个样例数据。
把所有选出的数据的视频片段、帧图像、人脸图像、标签文件（两个）、文本（包含该句以及几句上下文）
将每个例子对应的那句文本再同时加入一个列表，之后根据这个列表生成tsv文件用于文本模型对每句话的情感预测
自动写一张csv表，信息包括挑选出的数据的序号、所在的电影名、数据片段id、预测的有效说话人id、帧id号范围、denseface模型预测标签
"""

from utils import get_basename, mkdir
import os
import numpy as np
import random
from functools import total_ordering
from shutil import copyfile, copytree
from sys import exit
import sys
from tasks.text import TranscriptPackager
import csv
from tqdm import tqdm

video_clips_dir = '/data1/hzp/emo_bert/data/video_clips' #预处理过程切的视频片段的根目录
face_root_dir = "/data1/hzp/emo_bert/data/faces" #预处理过程提取出的脸部图像数据的根目录
frame_root_dir = '/data1/hzp/emo_bert/data/frames' #预处理过程抽出的帧的根目录
pseudo_annotation_dir = '/data1/hzp/emo_bert/data/pseudo_annotations_v3' #用于存放情感伪标注信息的根目录
transcripts_dir = '/data1/hzp/emo_bert/data/transcripts' #电影字幕文件的根目录

#emo_class_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disgust', 6: 'fear', 7: 'contempt'}
valid_category = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'fear'] #对于不在该列表中的类，不选择预测为该类情感的样例

def copy(source, target, type='file'):
    """
    拷贝文件或目录
    type：  'file'：文件（默认）；  'dir'：目录
    """
    assert (type == 'file') or (type == 'dir')
    if type == 'file':
        try:
            copyfile(source, target)
        except IOError as e:
            print('Unable to copy file. %s' % e)
            exit(1)
        except:
            print('Unexpected error: ', sys.exc_info())
            exit(1)
    else:
        try:
            copytree(source, target)
        except IOError as e:
            print('Unable to copy dir. %s' % e)
            exit(1)
        except:
            print('Unexpected error: ', sys.exc_info())
            exit(1)


class EmotionSegmentInfo(object):
    '''
    用于记录情感帧的信息的数据结构
    注：
    data_id为该组数据的序号，int类型
    其余参数均为str类型
    frame_id_range（片段的帧id号范围）为str类型，格式为：000001~000010
    '''
    def __init__(self, data_id, movie_name, clip_id, pred_speaker_id, frame_id_range, pred_label):
        self.data_id = data_id
        self.movie_name = movie_name
        self.clip_id = str(clip_id)
        self.pred_speaker_id = pred_speaker_id
        self.frame_id_range = frame_id_range
        self.pred_label = pred_label

    def __repr__(self):
        emoSegInfoStr = self.movie_name + '\t' + self.clip_id + '\t' + self.pred_speaker_id + '\t' + self.frame_id_range + '\t' + self.pred_label + '\n'
        return emoSegInfoStr


def sample_data(movie_list, num_per_movie, sample_dir, start_id = 1, context_len = 2):
    """
    start_id表示选出的数据的序号从几开始，默认为1。
    context_len表示在提供给标注者的文本文件中，分别给出当前句前面的几句以及后面的几句作为上下文。
    """
    output_text_file = os.path.join(sample_dir, 'text.tsv') #所有样例的文本
    output_info_file = os.path.join(sample_dir, 'info.csv') #所有样例的信息文件（数据的序号、所在的电影名、数据片段id、预测的有效说话人id、帧id号范围、denseface模型预测标签）

    data_id = start_id
    sample_list = [] #所有挑出的样例
    text_list = [] #所有挑出样例的文本列表

    print("----- sample data: -----")
    for movie in tqdm(movie_list):
        movie_emoseginfo_list = [] #存放该部电影中所有情感片段：[emotionsegment, active_speaker_id]
        movie_anno_folder = os.path.join(pseudo_annotation_dir, movie)
        anno_folder_list = os.listdir(movie_anno_folder)
        for clip_id in anno_folder_list:
            with open(os.path.join(movie_anno_folder, clip_id, 'continuous_anno_' + clip_id + '.txt'), 'r') as f:
                line = f.readline() #第一行是speaker_id的信息
                pred_speaker_id = int(line.strip().split(':')[1])
                line = f.readline()
                while line:
                    frame_id_range = line.strip().split()[0].split(':')[0]
                    pred_label = line.strip().split()[-1].split(':')[1]
                    if pred_label in valid_category: #预测的情感类别是我们想要保留的类别，则添加该条数据
                        emoseginfo = EmotionSegmentInfo(-1, movie, clip_id, pred_speaker_id, frame_id_range, pred_label)
                        movie_emoseginfo_list.append(emoseginfo)
                    line = f.readline()
        movie_sample_list = random.sample(movie_emoseginfo_list, num_per_movie)
        
        #为选出的片段分配id
        for item in movie_sample_list:
            item.data_id = data_id
            data_id += 1

        sample_list += movie_sample_list

    #创建每组数据的存放目录，并拷贝文件
    print("----- copy files: -----")
    for data in tqdm(sample_list):
        data_root_dir = os.path.join(sample_dir, str(data.data_id))
        mkdir(data_root_dir)
        #视频片段
        video_clip_source = os.path.join(video_clips_dir, data.movie_name, data.clip_id + '.mkv')
        video_clip_target = os.path.join(data_root_dir, str(data.data_id) + '.mkv')
        copy(video_clip_source, video_clip_target)
        #帧图像
        frame_source = os.path.join(frame_root_dir, data.movie_name, data.clip_id)
        frame_target = os.path.join(data_root_dir, str(data.data_id) + '_frame')
        copy(frame_source, frame_target, type='dir')
        #人脸图像
        face_source = os.path.join(face_root_dir, data.movie_name, data.clip_id, data.clip_id + '_aligned')
        face_target = os.path.join(data_root_dir, str(data.data_id) + '_face')
        copy(face_source, face_target, type='dir')
        #标签
        continuous_anno_source = os.path.join(pseudo_annotation_dir, data.movie_name, data.clip_id, 'continuous_anno_' + data.clip_id + '.txt')
        pseudo_anno_source = os.path.join(pseudo_annotation_dir, data.movie_name, data.clip_id, 'pseudo_anno_' + data.clip_id + '.txt')
        continuous_anno_target = os.path.join(data_root_dir, str(data.data_id) + '_continuous_anno.txt')
        pseudo_anno_target = os.path.join(data_root_dir, str(data.data_id) + '_pseudo_anno.txt')
        copy(continuous_anno_source, continuous_anno_target)
        copy(pseudo_anno_source, pseudo_anno_target)
        #文本
        transcript_ass = os.path.join(transcripts_dir, data.movie_name + '.ass')
        transcript_srt = os.path.join(transcripts_dir, data.movie_name + '.srt')
        if os.path.exists(transcript_ass):
            transcript_path = transcript_ass
        else:
            transcript_path = transcript_srt
        package_transcript = TranscriptPackager()
        #返回一个字典列表，列表中的每一项是记录每条字幕信息的字典：
        #   {start: 开始时间，end: 结束时间，content: 处理后的文本内容，index: 字幕序号}。
        transcript_info = package_transcript(transcript_path)
        cur_id = int(data.clip_id)
        previous_sentence = []
        posterior_sentence = []
        for i in range(1, context_len + 1):
            if cur_id - i < 0:
                previous_sentence.append('*****out of bounds*****')
            else:
                previous_sentence.append(transcript_info[cur_id - i]['content'])
            if cur_id + i >= len(transcript_info):
                posterior_sentence.append('*****out of bounds*****')
            else:
                posterior_sentence.append(transcript_info[cur_id + i]['content'])
        current_sentence = transcript_info[cur_id]['content']
        text_list.append(current_sentence)
        text_file_path = os.path.join(data_root_dir, str(data.data_id) + '_transcript.txt')
        with open(text_file_path, 'w') as f:
            for i in range(context_len - 1, -1, -1):
                f.write('cur - ' + str(i+1) + ':\t' + previous_sentence[i] + "\n")
            f.write('current:\t' + current_sentence + "\n")
            for i in range(context_len):
                f.write('cur + ' + str(i+1) + ':\t' + posterior_sentence[i] + "\n")
        
    #将所有样例的文本句子写入一个文件中
    with open(output_text_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for sentence in text_list:
            writer.writerow([sentence])

    #将所有样例信息写入csv文件中
    with open(output_info_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["data_id", "movie_name", "clip_id", "pred_speaker_id", "frame_id_range", "pred_label"]) #构建表头
        for data in sample_list:
            writer.writerow([data.data_id, data.movie_name, data.clip_id, data.pred_speaker_id, data.frame_id_range, data.pred_label])
        


if __name__ == '__main__':
    sample_dir = './sample'
    mkdir(sample_dir)

    movie_folder_list = sorted(os.listdir(pseudo_annotation_dir)) #获取所有标注目录下的电影文件夹列表
    #-----test----------
    movie_folder_list = ['No0009.The.Great.Gatsby', 'No0010.Love.in.the.Time.of.Cholera']
    #-------------------
    
    sample_data(movie_folder_list, 10, sample_dir, start_id = 81) #start_id = ...






