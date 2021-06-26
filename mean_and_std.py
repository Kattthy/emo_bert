import os, glob
import shutil
import h5py
import numpy as np
import tools.istarmap
import multiprocessing
from tqdm import tqdm
from utils import get_basename, mkdir
from PIL import Image

#IEMOCAP_denseface_path = '/data1/wjq/IEMOCAP_denseface'
#IEMOCAP_features_path = '/data1/wjq/IEMOCAP_features_npy'

#get_denseface = DensefaceExtractor()
#a = get_denseface('/data1/wjq/IEMOCAP_frames_and_faces/Session1/face/Ses01F_impro01_F000/00001.jpg')
#print(a)
# 提取denseface特征
def extract_denseface_feature():
    min_count = 1000000
    faces_path = '/data8/hzp/datasets/MovieData/face'
    movies = os.listdir(faces_path)
    mean_ = []
    std_ = []
    for movie in movies:
        if movie.find('Movie') != -1:           
            movie_faces_path = os.path.join(faces_path, movie)
            faces_file_paths = os.listdir(movie_faces_path)
            for faces_file_path in faces_file_paths:
                faces_path_ = os.path.join(movie_faces_path, faces_file_path)
                faces = os.listdir(faces_path_)
                for idx,face in enumerate(faces):
                    face_path = os.path.join(faces_path_, face)
                    face_img = Image.open(face_path)
                    face_img_grey = face_img.convert('L')
                    matrix = np.asarray(face_img_grey)  
                    mean_.append(np.mean(matrix))
                    std_.append(np.std(matrix))
                    #ft_test, pred_test = get_denseface(face_path)
                    #np.save(denseface_dir+'-ft.npy',ft_test)
                    #np.save(denseface_dir+'-pred.npy',pred_test) 
    avg_mean = np.mean(np.array(mean_))    
    avg_std = np.mean(np.array(std_))


    print(avg_mean, avg_std)

extract_denseface_feature()
