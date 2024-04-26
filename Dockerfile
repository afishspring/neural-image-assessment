# 设置基础镜像
FROM python:3.11

# 将代码添加到app文件夹中，无需新建（Docker运行时会自动创建）
COPY . /app

# 将code文件夹设置为工作目录
WORKDIR /app

# 安装所需的Python库
RUN pip install -r requirements.txt

EXPOSE 80

CMD ["python", "main.py", "0.0.0.0", "80"]