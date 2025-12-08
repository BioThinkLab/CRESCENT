import pandas as pd
import matplotlib.pyplot as plt

chr = 'ch2'
# 文件路径（请根据实际情况修改）
bins_file = f"/Users/sanjati/jangoTemp/temp2/pycharmD/preprocess/Version0209/output/BRCA/bins/processed_cnv_{chr}.txt"
ru_file= f'/Users/sanjati/jangoTemp/temp2/pycharmD/Data/RUBIC_output/BRCA/gains/focal_gains_{chr}.tsv'
gi_file = "/Data/GISTIC_output_extracted/BRCA/amp.tsv"

# 读取 bins 文件（例如之前生成的 bin 文件）
df_bins = pd.read_csv(bins_file, sep='\t')
print("Bins 数据预览：")
print(df_bins.head())

# 读取背景标注文件
df_ru = pd.read_csv(ru_file, sep='\t')
df_gi = pd.read_csv(gi_file, sep='\t')

# 如果染色体命名不一致（如 bins 为 "chr2"，背景为 "2"），则统一转换
df_ru['Chromosome'] = df_ru['Chromosome'].apply(
    lambda x: f"chr{x}" if not str(x).startswith("chr") else x
)
# 选择要绘制的染色体（本例以 "chr2" 为例）
chrom_to_plot = chr
df_bins = df_bins[df_bins['Chromosome'] == chrom_to_plot]
df_ru = df_ru[df_ru['Chromosome'] == chrom_to_plot]
df_gi = df_gi[df_gi['Chromosome'] == chrom_to_plot]

# 计算每个 bin 的宽度，用于绘制柱状图
df_bins['width'] = df_bins['End'] - df_bins['Start']

# 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制 Bin 柱状图：以基因组坐标 Start 为横坐标、宽度为 bin 的宽度，柱高为 n
ax.bar(df_bins['Start'], df_bins['n'], width=df_bins['width'], align='edge',
       color='skyblue', edgecolor='black', label='Bin counts (n)')

# 为使背景区域随图像缩放，这里以数据坐标绘制矩形
# 设定背景矩形的纵向范围：这里以柱状图的底部 0 到柱状图的最大高度 max_n 为例
max_n = df_bins['n'].max()
y_bottom = 0

# 遍历每个背景区间，用 Rectangle 绘制（注意：transform 默认就是 ax.transData）
for idx, row in df_ru.iterrows():
    x_left = row['Start']
    width_rect = row['End'] - row['Start']
    # 构造矩形：左下角在 (x_left, y_bottom)，高度覆盖整个柱状图区域
    rect = plt.Rectangle((x_left, y_bottom), width_rect, max_n,
                         color='orange', alpha=0.3, zorder=0)
    ax.add_patch(rect)

for idx, row in df_gi.iterrows():
    x_left = row['Start']
    width_rect = row['End'] - row['Start']
    # 构造矩形：左下角在 (x_left, y_bottom)，高度覆盖整个柱状图区域
    rect = plt.Rectangle((x_left, y_bottom), width_rect, max_n,
                         color='blue', alpha=0.3, zorder=0)
    ax.add_patch(rect)

# 设置横轴、纵轴标签及标题
ax.set_xlabel('Genomic Coordinate')
ax.set_ylabel('n')
ax.set_title(f'Bin counts with background regions on {chrom_to_plot}')

# 添加图例
ax.legend()

# 如果需要，可以设置合适的横轴范围，例如覆盖所有 bin 的区域
ax.set_xlim(df_bins['Start'].min(), df_bins['End'].max())
# 设置纵轴范围（例如从0到比最大值略大的数值）
ax.set_ylim(0, max_n * 1.1)

#  plt.tight_layout()
plt.show()