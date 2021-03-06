import pickle as pkl
import numpy as np
import PIL
import os

def format_aux(tab, new_shape):
  n, m, _= tab.shape
  return  np.pad(np.multiply(tab,1/255.,out=tab), ((0, new_shape[0]-n), (0, new_shape[1]-m), (0, 0)), constant_values = 1.)

class div2k:

  def __init__(self):
    self.HR_train = np.zeros((800, 2048, 2048, 3), dtype=float)
    self.LR_train = np.zeros((800, 512, 512, 3), dtype=float)
    self.HR_val = np.zeros((100, 2048, 2048, 3), dtype=float)
    self.LR_val = np.zeros((100, 512, 512, 3), dtype=float)
  
  def process(self, data_dir):
    paths=[data_dir+'DIV2K_train_HR/', data_dir+'DIV2K_train_LR_bicubic/X4/', data_dir+'DIV2K_valid_HR/', data_dir+'DIV2K_valid_LR_bicubic/X4/']
    i=0
    hrt_count = 0
    hrv_count = 0
    lrt_count = 0
    lrv_count = 0
    new_shapeH = (2048, 2048)
    new_shapeL = (512, 512)
    for path in paths:
      for image in os.listdir(path):
        if i==0:
          self.HR_train[hrt_count, :, :, :] = format_aux(np.array(PIL.Image.open(path+image)).astype(float), new_shapeH)
          if hrt_count % 50  == 0 :
            print('loading train high resolution images : {}/800'.format(hrt_count))  
          hrt_count+=1
        elif i==1:
          self.LR_train[lrt_count, :, :, :] = format_aux(np.array(PIL.Image.open(path+image)).astype(float), new_shapeL)
          if lrt_count % 50  == 0 :
            print('loading train low resolution images : {}/800'.format(lrt_count))  
          lrt_count+=1
        elif i==2:
          self.HR_val[hrv_count, :, :, :] = format_aux(np.array(PIL.Image.open(path+image)).astype(float), new_shapeH)
          if hrv_count % 50  == 0 :
            print('loading validation high resolution images : {}/100'.format(hrv_count))  
          hrv_count+=1
        elif i==3:
          self.LR_val[lrv_count, :, :, :] = format_aux(np.array(PIL.Image.open(path+image)).astype(float), new_shapeL)
          if lrv_count % 50  == 0 :
            print('loading validation low resolution images : {}/100'.format(lrv_count))  
          lrv_count+=1
      i+=1
  
  def save_data(self, dir_path):
    with open('{}dataset.pkl'.format(dir_path),'wb') as f:
      pkl.dump((self.HR_train, self.LR_train, self.HR_val, self.LR_val), f)
    f.close()
  
  def load_data(self, dir_path):
    with open('{}dataset.pkl'.format(dir_path),'rb') as f:
      self.HR_train, self.LR_train, self.HR_val, self.LR_val = pkl.load(f)
    f.close()
  
  def get_data(self):
    return self.HR_train, self.LR_train, self.HR_val, self.LR_val
