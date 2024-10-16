# 导入必要的库
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置 TensorBoard 日志文件的路径
log_path = '/home/majiancong/MambaCD/changedetection/logs/baseline_base_whu-ds-2'  # 替换为您的 TensorBoard 日志目录路径
log_path_nods = '/home/majiancong/MambaCD/changedetection/logs/baseline_base_whu-nods' 
# 加载 TensorBoard 日志数据
ea = event_accumulator.EventAccumulator(log_path)
ea.Reload()
ea_nods = event_accumulator.EventAccumulator(log_path_nods)
ea_nods.Reload()
# 提取 'IoU' 标量数据
scalars = ea.Scalars('final loss')  
scalars_nods = ea_nods.Scalars('final loss')
# 将 TensorBoard 数据转换为 Pandas DataFrame
df_ds = pd.DataFrame(scalars)
df_nods = pd.DataFrame(scalars_nods)

epoch_size = 90  # 每个 epoch 的迭代次数
df_ds['epoch'] = (df_ds['step'] - 1) // epoch_size + 1  # 计算每个迭代属于哪个 epoch
df_nods['epoch'] = (df_nods['step'] - 1) // epoch_size + 1 

# 计算每个 epoch 的 IoU 平均值
epoch_loss = df_ds.groupby('epoch')['value'].mean().reset_index()
epoch_loss_nods = df_nods.groupby('epoch')['value'].mean().reset_index()

# 计算平滑的 epoch loss（使用 rolling）
window_size = 70  # 窗口大小
epoch_loss['smoothed_epoch_loss'] = epoch_loss['value'].rolling(window=window_size).mean()
epoch_loss_nods['smoothed_epoch_loss'] = epoch_loss_nods['value'].rolling(window=window_size).mean()


# 绘图
palette = sns.color_palette("Paired")  # 设置调色盘
sns.set(style="whitegrid", context="talk")
plt.figure(figsize=(12, 6))

# 高亮特定区域
# plt.axvspan(0, 300, color='gray', alpha=0.2, label='Rise stage')
# plt.axvspan(300, 600, color='red', alpha=0.2, label='Stationary stage of with supervision')
# plt.axvspan(600, 1100, color='green', alpha=0.2, label='Stationary stage of both')
plt.xlim(0,700)
# 去掉 NaN 值
epoch_loss.dropna(subset=['smoothed_epoch_loss'], inplace=True)
epoch_loss_nods.dropna(subset=['smoothed_epoch_loss'], inplace=True)
# 平移 epoch 值
epoch_loss['epoch'] -= epoch_loss['epoch'].min()  # 将所有 epoch 平移到 0
epoch_loss_nods['epoch'] -= epoch_loss_nods['epoch'].min()  # 将所有 epoch 平移到 0
# 绘制平滑后的 epoch loss


sns.lineplot(x='epoch', y='smoothed_epoch_loss', data=epoch_loss, label='IoU with deep supervision', color='orange')
sns.lineplot(x='epoch', y='smoothed_epoch_loss', data=epoch_loss_nods, label='IoU without deep supervision', color='blue')

# 添加标题和标签
plt.title('IoU Curves: model w/o deep supervision', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('IoU', fontsize=12)
# plt.xticks(ticks=epoch_loss['epoch'])  # 设置 x 轴刻度为 epoch 数
plt.legend()
plt.savefig('epoch_IoU.png', dpi=300, bbox_inches='tight')
plt.show()