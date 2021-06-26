import numpy as np
import json
import os


pseudo_annotation_dir = '/data8/hzp/emo_bert/data/pseudo_annotations_v7' #用于存放情感伪标注信息的根目录
movie_folder_list = os.listdir(pseudo_annotation_dir) #获取所有电影文件夹列表
movie_folder_list = sorted(movie_folder_list)
#-----test----------
#movie_folder_list = ['No0001.The.Shawshank.Redemption']
movie_folder_list = movie_folder_list
#-------------------

vision_result = {}
final_result = {}

valid_emotion_list = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'fear']

for emo in valid_emotion_list:
    vision_result[emo] = 0
    final_result[emo] = 0

vision_result['total'] = 0
final_result['total'] = 0

for movie in movie_folder_list:
    movie_path = os.path.join(pseudo_annotation_dir, movie)
    vision_result_path = os.path.join(movie_path, 'continuous_result.json')
    final_result_path = os.path.join(movie_path, 'final_result.json')

    with open(vision_result_path, 'r') as f:
        vision_result_list = json.load(f)
        for item in vision_result_list:
            if item['denseface_emo'] in valid_emotion_list:
                vision_result[item['denseface_emo']] += 1
                vision_result['total'] += 1

    with open(final_result_path, 'r') as f:
        final_result_list = json.load(f)
        for item in final_result_list:
            if item['emo'] in valid_emotion_list:
                final_result[item['emo']] += 1
                final_result['total'] += 1

print('--------------------------------------')
print('vision result:')
print('total:', vision_result['total'])
for emo in valid_emotion_list:
    print(emo + ': ' + str(vision_result[emo]), end='\t')
print('\n--------------------------------------')
print('final result:')
print('total:', final_result['total'])
for emo in valid_emotion_list:
    print(emo + ': ' + str(final_result[emo]), end='\t')
print('\n')