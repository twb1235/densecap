import os
import h5py
import json
import pickle
import argparse

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

from imgapp.algrithm_pkg.model.densecap import densecap_resnet50_fpn

import json
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle

def load_model(console_args,model_checkpoint):
#传递两个参数  config.json和checkpoints
    with open(console_args, 'r') as f:
        model_args = json.load(f)
    print('加载模型')
    model = densecap_resnet50_fpn(backbone_pretrained=model_args['backbone_pretrained'],
                                  return_features=False,
                                  feat_size=model_args['feat_size'],
                                  hidden_size=model_args['hidden_size'],
                                  max_len=model_args['max_len'],
                                  emb_size=model_args['emb_size'],
                                  rnn_num_layers=model_args['rnn_num_layers'],
                                  vocab_size=model_args['vocab_size'],
                                  fusion_type=model_args['fusion_type'],
                                  box_detections_per_img=100)

    print('正在加载')
    checkpoint = torch.load(model_checkpoint)
    print('123')
    model.load_state_dict(checkpoint['model'])
    print('模型加载完成')

    

    return model


def get_image_path(img_url):

    img_list = []

    if os.path.isdir(img_url):
        for file_name in os.listdir(img_url):
            img_list.append(os.path.join(img_url, file_name))
    else:
        img_list.append(img_url)

    return img_list


def img_to_tensor(img_list):

    assert isinstance(img_list, list) and len(img_list) > 0

    img_tensors = []

    for img_path in img_list:

        img = Image.open(img_path).convert("RGB")

        img_tensors.append(transforms.ToTensor()(img))

    return img_tensors


def describe_images(model, img_list, device, batch_size):

    assert isinstance(img_list, list)
    assert isinstance(batch_size, int) and batch_size > 0

    all_results = []
    print('描述')
    with torch.no_grad():

        model.to(device)
        model.eval()

        for i in tqdm(range(0, len(img_list), batch_size), disable=False):

            image_tensors = img_to_tensor(img_list[i:i+batch_size])
            input_ = [t.to(device) for t in image_tensors]

            results = model(input_)

            all_results.extend([{k:v.cpu() for k,v in r.items()} for r in results])

    return all_results


def save_results_to_file(img_list, all_results, lut_path,result_dir):

    with open(os.path.join(lut_path), 'rb') as f:
        look_up_tables = pickle.load(f)

    idx_to_token = look_up_tables['idx_to_token']

    results_dict = {}

    for img_path, results in zip(img_list, all_results):

        

        results_dict[img_path] = []
        for box, cap, score in zip(results['boxes'], results['caps'], results['scores']):

            r = {
                'box': [round(c, 2) for c in box.tolist()],
                'score': round(score.item(), 2),
                'cap': ' '.join(idx_to_token[idx] for idx in cap.tolist()
                                if idx_to_token[idx] not in ['<pad>', '<bos>', '<eos>'])
            }

            results_dict[img_path].append(r)

        

    

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    with open(os.path.join(result_dir, 'result.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)

    

def main(console_args,img_url,checkpoints,batch_size,lut_path,result_dir):
#json，图片url，checkpoints,batch_size,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === prepare images ====
    img_list = get_image_path(img_url)

    # === prepare model ====
    model = load_model(console_args,checkpoints)

    # === inference ====
    #一次可以上传多张图片？？？
    all_results = describe_images(model, img_list, device, batch_size)

    # === save results ====
    save_results_to_file(img_list, all_results, lut_path,result_dir)
    
    



def caption(img_url):
	#img_url放到某一位置 
    img_url='.'+img_url
    print(img_url)
    
    
    #parser.add_argument('--img_path', type=str,default=img_url,help="path of images, should be a file or a directory with only images")
    
        
    batch_size=2
    #print()
    result_dir='imgapp/algrithm_pkg/image/'  #result.json文件
    lut_path='imgapp/algrithm_pkg/data/VG-regions-dicts-lite.pkl'
    config='imgapp/algrithm_pkg/model_params/config.json'
    checkpoints='imgapp/algrithm_pkg/model_params/train_all_val_all_bz_2_epoch_10_inject_init.pth.tar'
    main(config,img_url,checkpoints,batch_size,lut_path,result_dir)
    
    print('result生成')
    TO_K = 8
    RESULT_JSON_PATH = 'imgapp/algrithm_pkg/image/result.json'
    with open(RESULT_JSON_PATH, 'r') as f:
        results = json.load(f)
    print(results.keys())
    IMG_FILE_PATH=img_url
    result=results[IMG_FILE_PATH][:TO_K]
    assert IMG_FILE_PATH in results.keys()
    image_file_path=IMG_FILE_PATH
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    assert isinstance(result, list)

    img = Image.open(image_file_path)
    plt.imshow(img)
    ax = plt.gca()
    for r in result:
        ax.add_patch(Rectangle((r['box'][0], r['box'][1]),
                               r['box'][2]-r['box'][0],
                               r['box'][3]-r['box'][1],
                               fill=False,
                               edgecolor='red',
                               linewidth=3))
        ax.text(r['box'][0], r['box'][1], r['cap'], style='italic', bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})
    fig = plt.gcf()
    plt.tick_params(labelbottom='off', labelleft='off')
    s=image_file_path.split('/')[-1]
    plt.savefig('media/result/'+s)
    plt.close()
    return s

 

