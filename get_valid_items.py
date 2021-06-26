from tqdm import tqdm
import os
import json

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


face_root_dir = "/data7/emobert/data_nomask_new/faces" #预处理过程提取出的脸部图像数据的根目录
#check_data_root_dir = '/data8/hzp/emo_bert/data/check_data' #预处理过程记录有效抽脸信息的根目录
#frame_root_dir = '/data7/emobert/data_nomask_new/frames' #预处理过程抽出的帧的根目录
audio_root_dir = '/data7/emobert/data_nomask_new/audio_clips' #预处理过程抽出的音频的根目录
transcripts_dir = '/data7/emobert/data_nomask_new/transcripts/raw' #电影字幕文件的根目录


checkdata_json_root = '/data7/emobert/data_nomask_new/check_data'
target_data_root = '/data8/hzp/emo_bert/data'
target_checkdata_root = '/data8/hzp/emo_bert/data/check_data'
mkdir(target_checkdata_root)

#先将各个目录的文件对齐一下，将各个目录里都存在的电影名记录到文件中
checkdata_movie_list = ['.'.join(i.split('.')[:-1]) for i in os.listdir(checkdata_json_root)] #原check_data目录下的所有电影名称
face_movie_list = os.listdir(face_root_dir)
#frame_movie_list = os.listdir(frame_root_dir)
audio_movie_list = os.listdir(audio_root_dir)
transcripts_movie_list = ['.'.join(i.split('.')[:-1]) for i in os.listdir(transcripts_dir)]

movie_list = []
for movie in checkdata_movie_list:
    if (movie in face_movie_list) and (movie in audio_movie_list) and (movie in transcripts_movie_list):
        movie_list.append(movie)
movie_list = sorted(movie_list)

#print(len(checkdata_movie_list))
#print(len(face_movie_list))
#print(len(audio_movie_list))
#print(len(transcripts_movie_list))


with open(os.path.join(target_data_root, 'movie_list.txt'), 'w') as f:
    for movie in movie_list:
        f.write(movie + '\n')


#-----test----------
#movie_list = ['No0001.The.Shawshank.Redemption']
#-------------------

for movie in tqdm(movie_list):
    face_dir = os.path.join(face_root_dir, movie)
    all_face_list = [int(i) for i in os.listdir(face_dir)] #所有clip的id列表
    
    check_data_file = os.path.join(checkdata_json_root, movie + '.json')
    with open(check_data_file, 'r') as f:
        dic = json.load(f)
        #将各种类型的无效的clip的id列表合并并去重
        invalid_face_list = list(set(dic['no_sentence'] + dic['too_short'] + dic['no_face'] + dic['face_too_less'] + dic['no_active_spk']))
        valid_face_list = [i for i in all_face_list if i not in invalid_face_list]
        valid_face_list = sorted(valid_face_list) #对有效的列表做排序

        target_checkdata_file = os.path.join(target_checkdata_root, movie + '.txt')
        with open(target_checkdata_file, 'w') as f:
            for item in valid_face_list:
                f.write(movie + '/' + str(item) + '\n')



