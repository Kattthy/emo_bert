'''
构建“denseface+RoBERTa+情感词”策略（v7版本）下的电影数据集。
（注意这里就是简单地将所有伪标注有效的数据放进去，所以类别极其不均衡，后面get target的时候需要解决这个问题）
根目录：/data8/hzp/datasets/MovieData_v7
视觉模态：face/[movie name]/Movie000x_Clip000x/frame_det_00_00000x.bmp 
        每个clip目录下的是final_result.json文件中记录的该clip的"frames"项中的所有图片
语音模态：audio/[movie name]/Movie000x_Clip000x.wav
文本模态：text/[movie name].tsv
        tsv文件格式：Movie000x_Clip000x; [transcript]; [emotion]
'''

import numpy as np
import json
import os
from shutil import copyfile
from sys import exit
import sys
import csv
from tqdm import tqdm

#valid_emotion_list = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'fear'] #有效的类
valid_emotion_list = ['neutral', 'happiness', 'surprise', 'sadness', 'anger'] #有效的类


def copy(source, target):
    try:
        copyfile(source, target)
    except IOError as e:
        print('Unable to copy file. %s' % e)
        exit(1)
    except:
        print('Unexpected error:', sys.exc_info())
        exit(1)

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

def get_basename(path):
    basename = os.path.basename(path)
    if os.path.isfile(path):
        basename = basename[:basename.rfind('.')]
    return basename


dataset_root = '/data8/hzp/datasets/MovieData_v7'
target_text_dir = os.path.join(dataset_root, 'text') #数据集文本模态数据根目录
mkdir(target_text_dir)
pseudo_annotation_dir = '/data8/hzp/emo_bert/data/pseudo_annotations_v7' #用于存放情感伪标注信息的根目录


face_root_dir = "/data7/emobert/data_nomask_new/faces" #预处理过程提取出的脸部图像数据的根目录
audio_root_dir = '/data7/emobert/data_nomask_new/audio_clips' #预处理过程抽出的音频的根目录


 #获取所有电影文件夹列表
movie_list_file = '/data8/hzp/emo_bert/data/movie_list.txt'
movie_folder_list = []
with open(movie_list_file, 'r') as f:
        for line in f.readlines():
                movie_folder_list.append(line.strip())
movie_folder_list = sorted(movie_folder_list)

#-----test----------
#movie_folder_list = ['No0001.The.Shawshank.Redemption']
#-------------------

for movie in tqdm(movie_folder_list):
        #print('--------------------------')
        #print('process movie: ', movie)
        movie_id = movie[2:6] #000x
        target_face_dir = os.path.join(dataset_root, 'face', 'Movie' + movie_id) #数据集视觉模态数据根目录
        target_audio_dir = os.path.join(dataset_root, 'audio', 'Movie' + movie_id) #数据集语音模态数据根目录
        mkdir(target_face_dir)
        mkdir(target_audio_dir)
        annotation_path = os.path.join(pseudo_annotation_dir, movie, 'continuous_result.json')
        transcripts_path = os.path.join(pseudo_annotation_dir, movie, 'denseface_valid_transcripts.tsv')

        #读入数据
        annotation_list = []
        transcripts_list = []
        with open(annotation_path, 'r') as f:
                load_annotation = json.load(f)
        with open(transcripts_path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                load_transcripts = [i[0] for i in list(reader)]
        assert(len(load_annotation) == len(load_transcripts))
        for anno, tran in zip(load_annotation, load_transcripts):
                if anno['emo'] in valid_emotion_list:
                        annotation_list.append(anno)
                        transcripts_list.append(tran)
        
        #复制相应文件到数据集，并写入文本信息
        text_target_path = os.path.join(target_text_dir, 'Movie' + movie_id + '.tsv')
        with open(text_target_path, 'w') as f:
                writer = csv.writer(f, delimiter='\t')
        
                for anno, tran in zip(annotation_list, transcripts_list):
                        movie_id = movie[2:6] #000x
                        clip_origin_id = anno['clip_id'] #x
                        clip_id = clip_origin_id.zfill(4) #000x
                        data_name = 'Movie' + movie_id + '_Clip' + clip_id
                
                        #视觉：
                        mkdir(os.path.join(target_face_dir, data_name))
                        face_source_dir = os.path.join(face_root_dir, movie, clip_origin_id, clip_origin_id + '_aligned')
                        for frame in anno['frames']:
                                face_source_path = os.path.join(face_source_dir, frame + '.bmp')
                                face_target_path = os.path.join(target_face_dir, data_name, frame + '.bmp')
                                copy(face_source_path, face_target_path)

                        #语音:
                        audio_source_path = os.path.join(audio_root_dir, movie, clip_origin_id + '.wav')
                        audio_target_path = os.path.join(target_audio_dir, data_name + '.wav')
                        copy(audio_source_path, audio_target_path)

                        #文本：
                        data = [data_name, tran, anno['emo']]
                        writer.writerow(data)
        

print('dataset created successfully!')

