from protection import Fawkes
import os
import glob

# 创建 Fawkes 实例
protector = Fawkes(feature_extractor='arcface_extractor_0', gpu='0', batch_size=1, mode='high')

# 设置其他参数
directory = 'imgs/'  # 更新图像路径
th = 0.01
sd = 1e6
lr = 2
max_step = 1000
batch_size = 1
format = 'png'
separate_target = False
debug = False
no_align = False

# 获取图像路径
image_paths = glob.glob(os.path.join(directory, "*"))
image_paths = [path for path in image_paths if "_cloaked" not in path.split("/")[-1]]

# 调用 run_protection 方法
protector.run_protection(image_paths, th=th, sd=sd, lr=lr, max_step=max_step,
                         batch_size=batch_size, format=format,
                         separate_target=separate_target, debug=debug, no_align=no_align)
