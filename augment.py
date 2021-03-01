from PIL import Image, ImageDraw
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import json
import os

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_name, box_data,input_shape, random=True, max_boxes=20, jitter=.5, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
#     line = annotation_line.split()
    image = Image.open(annotation_name)
    iw, ih = image.size
    w,h= input_shape
    box = np.array(box_data)

    # 对图像进行缩放并且进行长和宽的扭曲
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.5,1.5)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # 将图像多余的部分加上灰条
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 翻转图像
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 色域扭曲
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # 将box进行调整
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box
    
    return image_data, box_data


def get_defect_name(i):
    if temps[i]["defect_name"]=="沾污":
        defect_name = 1
    elif temps[i]["defect_name"]=="错花":
        defect_name = 2
    elif temps[i]["defect_name"] == "水印":
        defect_name = 3
    elif temps[i]["defect_name"] == "花毛":
        defect_name = 4
    elif temps[i]["defect_name"] == "缝头":
        defect_name = 5
    elif temps[i]["defect_name"] == "缝头印":
        defect_name = 6
    elif temps[i]["defect_name"] == "虫粘":
        defect_name = 7
    elif temps[i]["defect_name"] == "破洞":
        defect_name = 8
    elif temps[i]["defect_name"] == "褶子":
        defect_name = 9
    elif temps[i]["defect_name"] == "织疵":
        defect_name = 10
    elif temps[i]["defect_name"] == "漏印":
        defect_name = 11
    elif temps[i]["defect_name"] == "蜡斑":
        defect_name = 12
    elif temps[i]["defect_name"] == "色差":
        defect_name = 13
    elif temps[i]["defect_name"] == "网折":
        defect_name = 14
    elif temps[i]["defect_name"] == "其他":
        defect_name = 15
        
    return defect_name

def reduce_defect_name(i):
    name = ["沾污", "错花", "水印", "花毛", "缝头", "缝头印", "虫粘", "破洞", "褶子", "织疵", "漏印", "蜡斑", "色差", "网折", "其他"]
    return name[i-1]



if __name__ == "__main__":
    josn_path = "./train_data/guangdong1_round2_train2_20191004_Annotations/Annotations/anno_train.json"
    with open(josn_path, 'r') as f:
        temps = json.loads(f.read()) # temps为读取的json文件（dic格式）
    new = temps#记录json变化
    
    #选择数据增强哪些label
    label = ["错花", "水印", "花毛", "缝头", "缝头印", "虫粘", "破洞", "褶子", "织疵", "漏印", "蜡斑", "色差", "网折", "其他"]
    passed = []
    for i in range(len(temps)):
        if (temps[i]['name'] in passed):#确保一张图片只增强一次
            continue
        if (temps[i]['defect_name'] not in label):
            continue
        
        raw_box = []
        
        x_l, y_l, x_r, y_r = temps[i]["bbox"]
        defect_name = get_defect_name(i)
        
        raw_box.append([x_l, y_l, x_r, y_r, defect_name])
        
        #往上找该图片是否有box跳过了
        i_ = i
        while temps[i_-1]['name'] == temps[i]['name']:
            x_l, y_l, x_r, y_r = temps[i_-1]["bbox"]
            defect_name = get_defect_name(i_-1)
            i_-=1
        #往下找把该图片的所有box都添加上
        i_ = i
        while i_+1 < len(temps) and temps[i_+1]['name'] == temps[i]['name']:
            x_l, y_l, x_r, y_r = temps[i_+1]["bbox"]
            defect_name = get_defect_name(i_+1)
            i_+=1

            raw_box.append([x_l, y_l, x_r, y_r, defect_name])
        
        passed.append(temps[i]['name'])
        
        img_name = temps[i]['name']
        img_path = './train_data/guangdong1_round2_train2_20191004_images/defect/' + img_name.split('.')[0]+'/'+temps[i]['name']
        
        if (temps[i]['defect_name'] in ["花毛", "缝头", "缝头印", "虫粘", "破洞", "褶子", "织疵", "漏印", "蜡斑", "色差", "网折", "其他"]):
            num = 4
        else:
            num = 1
        for t in range(num):
    #         print(raw_box)
            image_data, box_data = get_random_data(img_path,raw_box,[1024,1024],max_boxes=len(raw_box))
            os.makedirs('./process_data_new/all_defect/detect/'+img_name.split('.')[0] + '_'+ str(t)) 
            img = Image.fromarray((image_data*255).astype(np.uint8))
            img.save('./process_data_new/all_defect/detect/' +img_name.split('.')[0] + '_'+ str(t)+'/'+img_name.split('.')[0] + '_'+ str(t) + '.jpg')
            
            for k in box_data:
                tmp_k = list(k)
#                 print(type(k))
                append  = dict(name=img_name.split('.')[0] + '_'+ str(t) + '.jpg',defect_name=reduce_defect_name(int(tmp_k[4])),
                               bbox=[round(tmp_k[0],2),round(tmp_k[1],2),round(tmp_k[2],2),round(tmp_k[3],2)])
                new.append(append)
            
    with open("./train_data/guangdong1_round2_train2_20191004_Annotations/Annotations/new.json", 'w') as f:
        json.dump(new,f,indent=4)  