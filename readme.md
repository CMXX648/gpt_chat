# 项目依赖安装
==
1. 使用conda create 一个新环境 python==3.10
2. 创建好后conda activate 新环境
3. 切换conda工作目录到当前文件 pip install -r requirements.txt 安装项目依赖库

4. https://console.picovoice.ai/   生成属于自己的唤醒词后，得到的AccessKey 

### 最重要

进入网页https://www.paddlepaddle.org.cn/ 选择版本2.6 操作系统windows 安装方式pip 芯片厂商根据你电脑配置选择 
然后复制下方生成的代码 在conda终端中执行

当前项目新建文件夹 PaddleSpeech
###### 国内 github 访问较慢，这里使用 gitee 的仓库
###### 逐行执行
git clone -b r1.2 https://gitee.com/paddlepaddle/PaddleSpeech

cd PaddleSpeech

pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install uvicorn==0.18.3

pip install typeguard==2.13.3

下载nltk数据包 并解压https://paddlespeech.bj.bcebos.com/Parakeet/tools/nltk_data.tar.gz


## resource/setting填写对应内容
![img](Gpt_Chat/8be9315d69af1dcfbd6b8398304c677.png)

# 最后
```shell
python main.py
```