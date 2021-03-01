# tianchi_CV_Contest

For more informaton please visit my CSDN BLOG:https://editor.csdn.net/md/?articleId=114086202
or just contact me here by Issues.

## FYI
weights文件需要自己yolov5的官方仓库去下载sh文件，然后在终端运行；

以及代码的路径问题要注意，其实理清了代码的执行思路之后，路径就很好设置了。

——————————————————---

训练：python -m torch.distributed.launch --nproc_per_node 2 train.py --weights weights/yolov5s.pt --epochs 200 --batch-size 16 --cfg models/yolov5s.yaml --device 0,1 

修改容器中的文件：可以选择Ctrl+P+Q退出且不关闭容器，然后copy至容器内，再commit（具体过程可参考博客https://blog.csdn.net/dechengtju/article/details/85009836）

>  docker run -it 镜像ID /bin/bash
>
>  ctrl+p+q 退出
>
>  从宿主机拷贝文件到容器:docker cp /opt/test/file.txt mycontainer:/opt/testnew/
>
>  进入在运行的容器中：docker exec -it 容器ID /bin/bash
>
>  ls -l查看时间戳是否改变来确定是否copy成功
>
>  
>
>  保存容器为镜像：docker commit container_id 镜像仓库:版本号
>
>  docker stop 容器ID （exit退出容器（容器会关闭））
>
>  

接下来push 镜像：

> 登录：docker login --username=孟寻carl registry.cn-guangzhou.aliyuncs.com
>
> tag： docker tag [ImageId] registry.cn-guangzhou.aliyuncs.com/carl_namespace/defect_detect_new:[镜像版本号]
>
> push: docker push registry.cn-guangzhou.aliyuncs.com/carl_namespace/defect_detect_new:[镜像版本号]

容器内安装vim（方便修改文件）：apt-get install -y vim

查看镜像信息：docker images

查看容器信息： docker ps

保存容器为镜像：docker commit container_id 镜像仓库:版本号



import cv2 仍报错：

apt install libgl1-mesa-glx --fix-missing



## 数据

每张图片对应的瑕疵不止一处；

通过convertTrainLabel 将数据集划分为1:4，存放在convertor/images/train,  val中（label文件夹中每个图片只存了一个defect label, 以yolo要求的txt格式存储）

目前来看batchsize越小越好（慢），以及imagesize修改之后640,640也变得更好了。

## 自己的代码运行次序

1. convertTrainLabel.py
2. process_data_yolo.py
3. train.py
4. pt_edit.py，运行后会生成new.pt，接下来预测要修改默认参数best.pt为new.pt
5. 接下来代码运行结束，直接将pt文件copy到容器中并保存为新版本镜像即可。
