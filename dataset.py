import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

import unicodedata
import re

# LJdataset size: 13,100

def file_to_list(fdir):
	data_path=[]
	with open(os.path.join(fdir,'metadata.csv'),encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split('|')
			wav_path=os.path.join(fdir,'wavs',parts[0]+'.wav')
			data_path.append([parts[2],wav_path])

	return data_path

def text_normalization(text):
	text = ''.join(char for char in unicodedata.normalize('NFD', text)
				   if unicodedata.category(char) != 'Mn')  # Strip accents
	text = text.lower()
	text = re.sub("[^{}]".format(vocab), " ", text)
	text = re.sub("[ ]+", " ", text)
	return text

class LJDataset(Dataset):
	def __init__(self):
		super(LJDataset,self).__init__()

		self.file_list=file_to_list('LJSpeech-1.1')
		self.wav_to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=80, win_length=1024, hop_length=256, f_min=0.0, f_max=8000.0, n_fft=1024)

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self,index):
		wavform,sr=torchaudio.load(self.file_list[index][1])
		self.mel=self.wav_to_mel(wavform.squeeze())
		self.text=text_normalization(self.file_list[index][0])
		return self.text, self.mel

class Collate():
	def __init__(self):
		self.nmel=80

	def __call__(self,batch):
		
		txt_lengths, ids_sorted_txt = torch.sort(torch.LongTensor([len(data[0]) for data in batch]), dim=0, descending=True)
		sorted_mel_lengths, ids_sorted_mel = torch.sort(torch.LongTensor([data[1].size(1) for data in batch]), dim=0, descending=True)
		max_txt_len=txt_lengths[0]
		max_mel_len=sorted_mel_lengths[0]
		batch_size=len(ids_sorted_txt)

		text_padded=torch.IntTensor(batch_size,max_txt_len)
		mel_padded=torch.FloatTensor(batch_size,self.nmel,max_mel_len)
		gate_padded=torch.FloatTensor(batch_size,max_mel_len)
		mel_lengths=torch.LongTensor(batch_size)

		text_padded.zero_()
		mel_padded.zero_()
		gate_padded.zero_()

		vocab = " abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding, E: EOS.
		char2idx = {char: idx for idx, char in enumerate(vocab)}
		idx2char = {idx: char for idx, char in enumerate(vocab)}
		
		for i in range(batch_size):
			txt=batch[ids_sorted_txt[i]][0]
			mel=batch[ids_sorted_txt[i]][1]
			mel=self.dynamic_range_compression(mel)
			txt_norm=[char2idx[char] for char in txt]
			txt_norm=torch.IntTensor(txt_norm)
			text_padded[i,:len(txt_norm)]=txt_norm
			mel_padded[i,:,:mel.size(1)]=mel
			gate_padded[i,mel.size(1):]=1
			mel_lengths[i]=mel.size(1)

		return text_padded,mel_padded,gate_padded,txt_lengths,mel_lengths

	def dynamic_range_compression(self,x,C=1,clip_val=1e-5):
		return torch.log(torch.clamp(x,min=clip_val)*C)

dataset=LJDataset()
collate_fn=Collate()
train_loader=DataLoader(dataset,batch_size=4,shuffle=False,drop_last=False,collate_fn=collate_fn)