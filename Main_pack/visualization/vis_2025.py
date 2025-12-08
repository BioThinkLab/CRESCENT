import os
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots


def replace_zeros_with_mean(series):
    return series.replace(0, series.mean())


def merge_intervals(intervals):
    """
    Merge overlapping or contiguous intervals.

    Parameters:
        intervals (list of tuples): List of (start, end) tuples.

    Returns:
        merged (list of tuples): List of merged (start, end) tuples.
    """
    if not intervals:
        return []

    # Sort intervals based on start position
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        last = merged[-1]
        # Check if current interval overlaps or is contiguous with the last interval
        if current[0] <= last[1] + 1:
            # Merge the two intervals
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged


def draw(cancer_type, chr, is_gain=True):
    # 控制gain和loss
    type_ = 'gains' if is_gain else 'loss'
    g_type = 'amp' if is_gain else 'del'
    rubic = f"/Users/sanjati/jangoTemp/temp2/pycharmD/shin_Data/{cancer_type}/{g_type}/{cancer_type}_gistic_{g_type}.t2sv"
    gistic = f"/Users/sanjati/jangoTemp/temp2/pycharmD/shin_Data/{cancer_type}/{g_type}/{cancer_type}_rubic_{g_type}.tsv"
    # 2025-0307版
    filename = f"/Users/sanjati/jangoTemp/temp2/pycharmD/preprocess/Version0209/output/bin_with_case/{cancer_type}/cnv_{chr}.txt"
    # 第一版
    # filename = f"../Data/Matrix_cluster/{cancer_type}/{chr}_matrix_Focal.tsv"
    # 原始
    # filename = f"/Users/sanjati/jangoTemp/temp2/pycharmD/Data/Matrix/{cancer_type}/raw/{chr}_matrix.tsv"
    # 第二版
    # filename = f"/Users/sanjati/jangoTemp/temp2/pycharmD/Data/cluster_2025/{cancer_type}/focal/cnv_{chr}.txt"

    # 模型输出（用于测试检验，可选）
    test_filename = f"../Data/final/{cancer_type}/chr{chr}_matrix_.tsv"

    # --- 读取新格式的 GISTIC 数据 ---
    if os.path.exists(gistic):
        df_gistic = pd.read_csv(gistic, sep='\t')
        df_gistic = df_gistic[df_gistic["Chromosome"] == chr]
    else:
        df_gistic = pd.DataFrame()  # 如果文件不存在，初始化为空的DataFrame

    # 读取 RUBIC 数据
    if os.path.exists(rubic):
        df_rubic = pd.read_csv(rubic, sep='\t')
    else:
        df_rubic = pd.DataFrame()  # 如果文件不存在，初始化为空的DataFrame

    # 读取主数据文件
    df = pd.read_csv(filename, sep='\t')
    df.fillna(0, inplace=True)
    df.sort_values(by=['start', 'end'], inplace=True)

    # --- 着色粒范围数据---
    chrom_arm_file = "/Users/sanjati/jangoTemp/temp2/pycharmD/Data/ChromosomeData2025/GRCh38_Chromosome_Arm_Ranges.tsv"
    df_chrom_arm = pd.read_csv(chrom_arm_file, sep='\t')

    # 初始化子图：上部分用于数据曲线，下部分用于按钮或其他控制
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.9, 0.1],
        shared_xaxes=True,
        vertical_spacing=0.02
    )

    # --- 绘制 GISTIC 的绿色背景区域 ---
    if not df_gistic.empty:
        for idx, row in df_gistic.iterrows():
            start = int(row['Start'])
            end = int(row['End'])
            fig.add_shape(
                type="rect",
                x0=start,
                x1=end,
                yref="paper",  # 使用 paper 坐标系（相对于整个画布）
                y0=0,
                y1=1,
                fillcolor="green",
                opacity=0.3,
                line_width=0,
            )
            # 添加注释
            fig.add_annotation(
                x=(start + end) / 2,
                y=1.03,
                text=f"{start}-{end}",
                showarrow=False,
                yref="paper",
                align="center",
                font=dict(color="black"),
                bgcolor="green",
                bordercolor="black",
            )

    # --- 绘制 RUBIC 的红色背景区域 ---
    if not df_rubic.empty:
        for index, row in df_rubic.iterrows():
            fig.add_shape(
                type="rect",
                x0=row['Start'],
                x1=row['End'],
                yref="paper",
                y0=0,
                y1=1,
                fillcolor="red",
                opacity=0.3,
                line_width=0,
            )
            # 添加注释
            fig.add_annotation(
                x=(int(row['Start']) + int(row['End'])) / 2,
                y=1.03,
                text=f"{int(row['Start'])}-{int(row['End'])}",
                showarrow=False,
                yref="paper",
                align="center",
                font=dict(color="black"),
                bgcolor="red",
                bordercolor="black",
            )

    # --- 新逻辑：绘制染色体片段的灰色背景区域 ---
    # 只选择当前染色体的行
    df_chrom_arm_filtered = df_chrom_arm[df_chrom_arm["chromosome"] == chr]

    # 对于每一行，绘制 `end0` 和 `start1` 之间的灰色背景区域
    for index, row in df_chrom_arm_filtered.iterrows():
        end0 = int(row['end0'])
        start1 = int(row['start1'])
        fig.add_shape(
            type="rect",
            x0=end0,
            x1=start1,
            yref="paper",
            y0=0,
            y1=1,
            fillcolor="gray",
            opacity=0.3,
            line_width=0,
        )
        # 添加注释
        fig.add_annotation(
            x=(end0 + start1) / 2,
            y=1.06,  # 稍高于其他区域的注释
            text=f"{end0}-{start1}",
            showarrow=False,
            yref="paper",
            align="center",
            font=dict(color="black"),
            bgcolor="gray",
            bordercolor="black",
        )

    # --- 绘制主数据文件中的各个 series 曲线 ---
    for col in df.columns[4:22]:
        x = []
        y = []
        for _, row in df.iterrows():
            x.extend([row['start'], row['end']])
            y.extend([row[col], row[col]])
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=col,
            line_shape='hv',
            visible='legendonly'  # 默认设为隐藏，可在图例中开启
        ), row=1, col=1)

    # --- 新增逻辑：叠加所有 series 后除以2 ---
    summed_values = df[df.columns[4:22]].sum(axis=1)
    adjusted_values = summed_values

    x_avg = []
    y_avg = []
    for idx, row in df.iterrows():
        x_avg.extend([row['start'], row['end']])
        y_avg.extend([adjusted_values.iloc[idx], adjusted_values.iloc[idx]])

    fig.add_trace(go.Scatter(
        x=x_avg,
        y=y_avg,
        mode='lines',
        name='Summed',
        line_shape='hv',
        visible=True  # 默认显示叠加曲线
    ), row=1, col=1)

    # --- 新增逻辑：读取并绘制 Model2025 的数据 ---
    # 构造数据文件路径：目录结构为 pycharmD/Model2025/output/{cancer_type}/{chr}.txt
    model2025_path = f"/Users/sanjati/jangoTemp/temp2/pycharmD/Model2025/output/{cancer_type}/{chr}.txt"
    if os.path.exists(model2025_path):
        with open(model2025_path, 'r') as f:
            # 每行读取一个浮点数
            model_values = [float(line.strip()) for line in f if line.strip()]
        n_segments = len(model_values)
        if n_segments > 0:
            # 数值乘以200
            model_values = [val * 200 for val in model_values]
            # 使用主数据文件的横轴范围进行等分
            x_min = df['start'].min()
            x_max = df['end'].max()
            segment_width = (x_max - x_min) / n_segments
            # 计算每个区间的中点作为 x 位置
            x_positions = [x_min + (i + 0.5) * segment_width for i in range(n_segments)]
            fig.add_trace(go.Scatter(
                x=x_positions,
                y=model_values,
                mode='markers+lines',
                name='Model2025',
                line_shape='hv'
            ), row=1, col=1)

    # --- 可选：添加一个不可见的 trace 用于按钮（控制显示/隐藏） ---
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=1, color="rgba(0,0,0,0)"),
        showlegend=False
    ), row=2, col=1)

    # 更新布局，设置标题、坐标轴标签、图例及按钮等
    fig.update_layout(
        title=f"{cancer_type} -- {chr}",
        xaxis_title="Position",
        yaxis_title="Value",
        height=800,
        width=1600,
        showlegend=True,
        legend_title="Series",
        margin=dict(l=50, r=50, t=50, b=50),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.5,
            y=-0.15,
            buttons=list([
                dict(
                    args=[{"visible": [trace.visible for trace in fig.data]}],
                    label="Hide All",
                    method="update",
                ),
                dict(
                    args=[{"visible": [True for _ in fig.data]}],
                    label="Show All",
                    method="update",
                ),
            ]),
        )],
    )

    # 保存图片并显示图形
    pio.write_image(fig, './vis_plotly.png')
    fig.show()


if __name__ == '__main__':
    draw(cancer_type='BLCA', chr='chr', is_gain=True)