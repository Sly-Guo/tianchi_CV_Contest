import numpy as np # linear algebra
import os
import json
from tqdm.auto import tqdm  # Python进度条,对于任何可迭代的对象均可使用
import shutil as sh # shutil模块对文件和文件集合提供了许多高级操作，特别是提供了支持文件复制和删除的函数

import cv2

josn_path = "./train_data/guangdong1_round2_train1_20190924_Annotations/new.json"
image_path = "./train_data/guangdong1_round2_train1_20190924_images/"

# josn_path = "./train_data/guangdong1_round2_train2_20191004_Annotations/Annotations/new.json"
# image_path = "./process_data_new/all_defect/detect/"
# image_path = "./train_data/guangdong1_round2_train2_20191004_images/defect/"

name_list = []
image_h_list = []
image_w_list = []
c_list = []
w_list = []
h_list = []
x_center_list = []
y_center_list = []

with open(josn_path, 'r') as f:
    temps = tqdm(json.loads(f.read()))
    for temp in temps:
        # image_w = temp["image_width"]
        # image_h = temp["image_height"]
        name = temp["name"].split('.')[0]   # 去掉.jpg后缀 
        # temp["name"] = name + '.jpg'
        path = os.path.join(image_path, name, temp["name"]) # 图片数据的路径， imagename/imagename.jpg
        print('path: ',path)
        im = cv2.imread(path)
        sp = im.shape
        image_h, image_w = sp[0], sp[1] # h:1800 w:4096
        # print("image_h, image_w: ", image_h, image_w)
        # print("defect_name: ",temp["defect_name"])
        #bboxs
        x_l, y_l, x_r, y_r = temp["bbox"]
        # print(temp["name"], temp["bbox"])
        if temp["defect_name"]=="沾污":
            defect_name = '0'
        elif temp["defect_name"]=="错花":
            defect_name = '1'
        elif temp["defect_name"] == "水印":
            defect_name = '2'
        elif temp["defect_name"] == "花毛":
            defect_name = '3'
        elif temp["defect_name"] == "缝头":
            defect_name = '4'
        elif temp["defect_name"] == "缝头印":
            defect_name = '5'
        elif temp["defect_name"] == "虫粘":
            defect_name = '6'
        elif temp["defect_name"] == "破洞":
            defect_name = '7'
        elif temp["defect_name"] == "褶子":
            defect_name = '8'
        elif temp["defect_name"] == "织疵":
            defect_name = '9'
        elif temp["defect_name"] == "漏印":
            defect_name = '10'
        elif temp["defect_name"] == "蜡斑":
            defect_name = '11'
        elif temp["defect_name"] == "色差":
            defect_name = '12'
        elif temp["defect_name"] == "网折":
            defect_name = '13'
        elif temp["defect_name"] == "其他":
            defect_name = '14'
        else:
            defect_name = '15'
            print("----------------------------------error---------------------------")
            raise("erro")
        # print(image_w, image_h)
        # print(defect_name)
        x_center = (x_l + x_r)/(2*image_w)
        y_center = (y_l + y_r)/(2*image_h)
        w = (x_r - x_l)/(image_w)
        h = (y_r - y_l)/(image_h)
        # print(x_center, y_center, w, h)
        name_list.append(temp["name"])  # 图片名
        c_list.append(defect_name)  # 瑕疵种类的名称
        image_h_list.append(image_w) 
        image_w_list.append(image_h)
        ### 下面则是yolo的box格式
        x_center_list.append(x_center)
        y_center_list.append(y_center)
        w_list.append(w)
        h_list.append(h)
    print("Transforming done !!!") 


    # index = list(set(name_list))
    index = name_list
    print(len(index))
    
    for fold in [0]:
        # 这个地方是分1/5的数据为测试集，4/5的数据为训练集，并分别存储在val与train下
        val_index = index[len(index) * fold // 5:len(index) * (fold + 1) // 5]
        print(len(val_index))
        for num, name in enumerate(name_list):
            # print: defect_name, 坐标信息
            print(c_list[num], x_center_list[num], y_center_list[num], w_list[num], h_list[num])
            row = [c_list[num], x_center_list[num], y_center_list[num], w_list[num], h_list[num]]
            if name in val_index:
                path2save = 'val/'
            else:
                path2save = 'train/'
            # print('convertor\\fold{}\\labels\\'.format(fold) + path2save)
            # print('convertor\\fold{}/labels\\'.format(fold) + path2save + name.split('.')[0] + ".txt")
            # print("{}/{}".format(image_path, name))
            # print('convertor\\fold{}\\images\\{}\\{}'.format(fold, path2save, name))
            if not os.path.exists('convertor/fold{}/labels/'.format(fold) + path2save):
                os.makedirs('convertor/fold{}/labels/'.format(fold) + path2save)
            with open('convertor/fold{}/labels/'.format(fold) + path2save + name.split('.')[0] + ".txt", 'a+') as f:
                for data in row:
                    f.write('{} '.format(data))
                f.write('\n')
                if not os.path.exists('convertor/fold{}/images/{}'.format(fold, path2save)):
                    os.makedirs('convertor/fold{}/images/{}'.format(fold, path2save))
                sh.copy(os.path.join(image_path, name.split('.')[0], name),
                        'convertor/fold{}/images/{}/{}'.format(fold, path2save, name))


