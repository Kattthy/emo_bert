'''
每次处理一个视频, 处理之前要检查是否包含字幕文件
Step1: 抽取字幕文件.
Step2: 根据字幕文件进行时间戳和文本信息的整理.
Step3: 根据时间戳进行视频和音频信息的抽取.
--- 以上三步用FFMPEG进行操作, 按照文档中写 https://docs.qq.com/doc/DU0N0TFZCR2h0alJE
--- 保存目录结构，你设计一下
'''
import torch
import os, glob
import subprocess
import h5py
import json
import re, time
import numpy as np
from typing import List
from numpy import ndarray
from langdetect import detect, detect_langs
from langdetect import DetectorFactory
DetectorFactory.seed = 0
from transformers import (
    BertModel, AlbertModel,
    BertTokenizer, TransfoXLTokenizer, AlbertTokenizer
)

from .base_worker import BaseWorker
from utils import get_basename, mkdir

class TranscriptExtractor(BaseWorker):
    def __init__(self, save_root, logger=None):
        super().__init__(logger=logger)
        self.save_root = save_root
        mkdir(self.save_root)

    def __call__(self, video_path):
        basename = get_basename(video_path)
        save_path_ass = os.path.join(self.save_root, basename+'.ass')
        save_path_srt = os.path.join(self.save_root, basename+'.srt')
        if not os.path.exists(save_path_ass) and not os.path.exists(save_path_srt):
            p = subprocess.Popen([f'ffmpeg -i {video_path}'], stderr=subprocess.PIPE, shell=True) #获取视频流和音频流信息
            lines = p.stderr.readlines() #读入信息
            duration = {}
            cur_id = -1
            for line in lines:
                line = line.decode() #解码字符串
                if 'Stream' in line and "Subtitle" in line: #找到类似这样的行：“Stream #0:2(eng): Subtitle: subrip”
                    stream_num = int(line[line.find(':')+1:line.find(':')+2]) #找到上例中的“2”
                    cur_id = stream_num
                    continue
                if cur_id != -1 and 'DURATION' in line and ':' in line[line.find('DURATION'):]: #在找到字幕stream的情况下，如果当前行包含“DURATION”，且之后有“:”。如：“DURATION        : 00:18:56.880000000”
                    _duration = line[line.find(':')+1:].strip().split('.')[0] #提取出时间字符串，并忽略掉秒数的小数部分
                    _duration = time.mktime(time.strptime(_duration, '%H:%M:%S')) #time.strptime：把时间字符串解析为时间元组。time.mktime：返回用秒数表示时间的浮点数
                    duration[cur_id] = _duration #字典{字幕id：持续时间}

            ranked_duration = sorted(duration.items(), key=lambda x: x[1], reverse=True) #按duration把所有字幕字典项降序排列。items()：以列表返回可遍历的(键, 值) 元组数组。
            if len(ranked_duration) > 0:
                longest_id = ranked_duration[0][0] #取持续时间最长的字幕
            else:
                longest_id = 0
            
            #_cmd = "ffmpeg -i {} -an -vn -map 0:s:{} {} -y > /dev/null 2>&1"
            _cmd = "ffmpeg -i {} -an -vn -map 0:{} {} -y > /dev/null 2>&1" #提取出那条持续时间最长的字幕
            self.print(f"{video_path}->{save_path_ass}: Using Stream number {longest_id}. ")
            os.system(_cmd.format(video_path, longest_id, save_path_ass))
            return save_path_ass
        elif not os.path.exists(save_path_ass):
            self.print(f'Found in {save_path_srt}, skip')
            return save_path_srt
        else:
            self.print(f'Found in {save_path_ass}, skip')
            return save_path_ass
        
    
class DetectLanguage(BaseWorker):
    def __init__(self, logger=None):
        super().__init__(logger=logger)   

    def __call__(self, sentence):
        ans = detect(sentence)
        self.print(f'{ans}: {sentence}')
        return ans

class TranscriptPackager(BaseWorker):
    def __init__(self, logger=None):
        super().__init__(logger=logger)
    
    def process_text(self, text, mode='en_ch'): #去除各种符号、非ascii字符、换行符。
        if mode == 'en_ch':
            if "\\N" in text:
                text = text.split('\\N')[-1]

            # 去除非ascii字符(比如 '∮ if i dont care ∮' )
            text = ''.join(filter(lambda x: ord(x) < 128, text)).strip()
            

        elif mode == 'en':
            text = ''.join(filter(lambda x: ord(x) < 128, text)).strip()
            text = text.replace('\\N', ' ')
        # 去除-_@#¥%^*()_+={}[]\|<>/
        text = re.sub(u'[-_@#¥%^*()_+={}\[\]\\\|<>/\n]+', "", text)

        return text.strip()
    
    def get_transcripts_mode(self, lines): #判断字幕类型是“中英”还是“英”
        # 包含中文的行大于90%则为en_ch, 否则为en
        ch_count = 0
        for line in lines:
            if any(u'\u4e00' <= ch <= u'\u9fff' for ch in line): #统计中文字符数。在Unicode编码中4E00-9FFF为中文字符编码区。
                ch_count += 1
        
        if ch_count > int(len(lines)*0.9):
            return "en_ch"
        else:
            return "en"
        
    def process_ass(self, lines):
        lines = list(filter(lambda x: x.startswith('Dialogue:'), lines)) #只保留所有以“Dialogue:”开头的行。filter()函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。该函数接收两个参数，第一个为函数，第二个为序列。
        ans = []
        mode = self.get_transcripts_mode(lines) #判断字幕类型是“中英”还是“英”
        for i, line in enumerate(lines):
            start_time, end_time = line.split(',')[1:3] #获得该行字幕的起始时间和结束时间
            text_content = ''.join(line.split(',')[9:])
            text_content = re.sub(u"\\{.*?}", "", text_content) #去掉大括号及里面的内容
            _text_content = self.process_text(text_content, mode) #去除各种符号、非ascii字符、换行符。
            ans.append({'start': start_time, 'end': end_time, "content": _text_content, 'index':i})
        return ans

    def process_srt(self, lines):
        '''
        srt字幕示例：
        3
        00:01:18,614 --> 00:01:23,614
        死时还未懂得爱的真谛 那就比死亡本身更可怕了
        The fear of dying without ever having known
        love was greater than the fear of death itself
        '''
        ans = []
        sentence_line_numbers = []
        for i, line in enumerate(lines):
            line = line.strip().replace('\ufeff', '') #去除'\ufeff'非法字符
            if line.isnumeric(): #如果字符串只由数字组成，这说明它是该句字幕的序号
                sentence_line_numbers.append(i) #把字幕序号所在行的行号加入列表
        sentence_line_numbers.append(None)
        mode = self.get_transcripts_mode(lines) #判断字幕类型是“中英”还是“英”
        for i in range(len(sentence_line_numbers)-1):
            contents = lines[sentence_line_numbers[i]: sentence_line_numbers[i+1]]
            duration_info = contents[1]
            start = duration_info.split('-->')[0].strip().replace(',', '.')
            end = duration_info.split('-->')[1].strip().replace(',', '.')
            sentence = ' '.join(contents[2:])
            sentence = self.process_text(sentence, mode) #去除各种符号、非ascii字符、换行符。
            ans.append({'start': start, 'end': end, "content": sentence, "index":i})
        return ans

    def __call__(self, transcript_file_path):
        if not os.path.exists(transcript_file_path):
            self.print(f"Error: {transcript_file_path} not exists")
            return
        
        lines = open(transcript_file_path, encoding='utf8').readlines()
        #lines = open(transcript_file_path, encoding='unicode_escape').readlines()
        if len(lines) == 0:
            self.print("Error in {}, empty file.".format(transcript_file_path))
            return 
        
        file_format = transcript_file_path.split('.')[-1] #通过字幕文件名后缀获取字幕格式
        if file_format == 'ass':
            ans = self.process_ass(lines)
        elif file_format == 'srt':
            ans = self.process_srt(lines)
        else:
            raise ValueError(f'[TranscriptPackager]: file format not supported: {transcript_file_path}')
        return ans
    
class BertExtractor(object):

    def __init__(self, device=0, model_name='bert_base'):
        """
        :param model_name: bert size: base(768), small(512), medium(512), tiny(128), mini(256)
        Choice in [bert_base, bert_medium, bert_mini, bert_tiny, albert_base]
        """
        assert model_name in ['bert_base', 'bert_medium', 'bert_small', 
                'bert_mini', 'bert_tiny', 'albert_base', 'albert_large',
                'bert_base_arousal', 'bert_base_valence'], f'Model type not supported: {model_name}'
        self.model_name = model_name
        self.device = torch.device('cuda:{}'.format(device))
        self.pretrained_path = self.get_pretrained_path()
        self.get_tokenizer(self.model_name.split('_')[0])
        self.max_length = 512
        if self.model_name.split('_')[0] == 'bert':
            self.model = BertModel.from_pretrained(self.pretrained_path).to(self.device)
        else:
            print('Albert')
            self.model = AlbertModel.from_pretrained('albert-base-v2').to(self.device)

        self.model.eval()
    
    def get_pretrained_path(self):
        path_config = {
            'bert_base': '/data7/hjw/Bert_pretrn/bert_base_uncased_L-12_H-768_A-12',
            'bert_medium': '/data7/hjw/Bert_pretrn/bert_medium_uncased_L-8_H-512_A-8',
            'bert_small': '/data7/hjw/Bert_pretrn/bert_small_uncased_L-4_H-512_A-8',
            'bert_mini': '/data7/hjw/Bert_pretrn/bert_mini_uncased_L-4_H-256_A-4',
            'bert_tiny': '/data7/hjw/Bert_pretrn/bert_tiny_uncased_L-2_H-128_A-2',
            'albert_base': '/data7/lrc/MuSe2020/hhh/pretrain_model/albert_base',
            'albert_large': '/data7/lrc/MuSe2020/hhh/pretrain_model/albert_large',
            # finetune on MuSe arousal label
            'bert_base_arousal': '/data7/lrc/MuSe2020/MuSe2020_features/code/finetune_bert/output/arousal',
            # finetune on MuSe valence label
            'bert_base_valence': '/data7/lrc/MuSe2020/MuSe2020_features/code/finetune_bert/output/valence',
        }
        return path_config[self.model_name]

    def get_tokenizer(self, model_name):
        if model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_path)
        elif model_name == 'transformer-xl':
            self.tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
            self.tokenizer.pad_token = '[PAD]'
        elif model_name == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2') # 'albert-base-v2'
        else:
            raise ValueError('model name not supported')

    def extract_words(self, word_list: List[str]):
        # since the model is uncased
        word_list = list(map(lambda x: x.lower(), word_list))
        ids = self.tokenizer.convert_tokens_to_ids(word_list)
        ids = torch.tensor(ids).unsqueeze(0)
        ans = []
        for i in range(0, ids.shape[1], self.max_length):
            _sub_ids = ids[:, i: i+self.max_length]
            if i != 0 and _sub_ids.shape[1] < self.max_length: # encounters the tail
                _sub_ids = ids[:, -self.max_length:]
                start = -(ids.shape[1] % self.max_length)
            else:
                start = 0
            with torch.no_grad():
                feats = self.model(_sub_ids.to(self.device))[0]
            feats = feats.detach().cpu().numpy().squeeze()
            ans.append(feats[start:])
        return np.concatenate(ans)
    
    def extract_sentence(self, sentence: str):
        ids = self.tokenizer.encode(sentence)
        ids = torch.tensor(ids).unsqueeze(0)
        feats = self.model(ids.to(self.device))[0]
        return feats.detach().cpu().numpy()
    
    def __call__(self, sentence: str):
        return self.extract_sentence(sentence)


if __name__ == "__main__":
    # text_pipeline('../resources/raw_movies')
    # get_transcript = tor('./transcripts/ass')
    # package_transcript = TranscriptPackager()
    # ass_path = get_transcript("../resources/raw_movies/No0001.The.Shawshank.Redemption.ass")
    # transcripts = package_transcript(ass_path)
    # sentence = transcripts[0]['content']
    # print(sentence)
    # extract_bert = BertExtractor(device=0)
    # feature = extract_bert(sentence)
    # print(feature.shape)
    # print(transcripts)
    # package_transcript('transcripts/ass/No0001.The.Shawshank.Redemption.ass')

    # tokenizer = BertTokenizer.from_pretrained('/data7/hjw/Bert_pretrn/bert_base_uncased_L-12_H-768_A-12')
    # sentence = "Nisa be-ru'ach ha'arbaim Im kol pa'amonim"
    # ids = tokenizer.encode(sentence)
    # print(ids)
    # tokens = tokenizer.convert_ids_to_tokens(ids)
    # print(tokens)
    # text = "Nisa []<>be-ru'ach ha'arbaim Im |kol\ pa'amonim"
    # text = re.sub(u'[-_@#¥%^*()_+={}\[\]\\\|<>/]+', "", text)
    # print(text)
    process_transcripts = TranscriptPackager()
    #a = process_transcripts('/data6/zjm/emobert/resources/raw_movies/No0022.Rent.srt')
    a = process_transcripts('/data1/hzp/emo_bert/resources/raw_movies/No0001.The.Shawshank.Redemption.ass')
    json.dump(a, open('test.json', 'w'), indent=4)
    print(a)