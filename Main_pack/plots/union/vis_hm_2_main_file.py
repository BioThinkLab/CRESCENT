# 用来画heatmap静态图的(figure4上半部分)
import os
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

def unify_column_case(df):
    rename_map = {}
    for col in df.columns:
        kl = col.lower()
        if kl == 'chromosome':
            rename_map[col] = 'Chromosome'
        elif kl == 'start':
            rename_map[col] = 'Start'
        elif kl == 'end':
            rename_map[col] = 'End'
    df.rename(columns=rename_map, inplace=True)
    return df

def load_tsv(path):
    if os.path.exists(path):
        df = pd.read_csv(path, sep='\t')
        return unify_column_case(df)
    else:
        print(f"[DEBUG] File not found: {path}")
        return pd.DataFrame()

def draw_static(cancer_type, amp_del_other, chrom):
    # 选择增幅或缺失
    if amp_del_other == "amp":
        g_type = f_type = 'amp'
    elif amp_del_other == "del":
        g_type = f_type = 'del'
    else:
        g_type, f_type = 'del', 'amp'

    base = "/Users/sanjati/jangoTemp/temp2/pycharmD"
    df_g = load_tsv(f"{base}/shin_Data/{cancer_type}/{g_type}/{cancer_type}_gistic_{g_type}.tsv")
    df_r = load_tsv(f"{base}/shin_Data/{cancer_type}/{g_type}/{cancer_type}_rubic_{g_type}.tsv")
    df_o = load_tsv(f"{base}/shin_Data/{cancer_type}/{g_type}/{cancer_type}_om_{g_type}.tsv")

    main_file = (
        f"{base}/"
        + ("bin_with_case_amp_compressed" if f_type == 'amp' else "bin_with_case_del_compressed")
        + f"/{cancer_type}/cnv_{chrom}"
        + (".txt" if f_type == 'amp' else ".tsv")
    )
    print(f"input:{main_file}")
    if not os.path.exists(main_file):
        print(f"[DEBUG] Missing data for {chrom}, skip.")
        return

    df = pd.read_csv(main_file, sep='\t')
    unify_column_case(df)
    df.fillna(0, inplace=True)
    df.sort_values(['Start', 'End'], inplace=True)

    # 计算中点及热图矩阵
    mid = ((df['Start'] + df['End']) / 2).tolist()
    sample_names = df.columns[4:38].tolist()
    sample_labels = [sn[:4] + '...' if len(sn) > 4 else sn for sn in sample_names]
    z = [df[c].tolist() for c in sample_names]

    # 构建两层子图：上面热力图 + confidence 曲线，下面基线和区段
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        row_heights=[0.85, 0.15],
        shared_xaxes=True,
        vertical_spacing=0.1
    )

    # 热力图
    fig.add_trace(
        go.Heatmap(
            x=mid, y=sample_labels, z=z,
            showscale=True,
            colorbar=dict(
                title=' ', titleside='right', ticks='outside', thickness=15,
                len=0.8, y=0.6, yanchor='middle', x=1.0, xanchor='left'
            )
        ),
        row=1, col=1, secondary_y=False
    )

    # confidence 曲线（不显示图例）
    if 'prob' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=mid, y=df['prob'], mode='lines',
                name='prob', showlegend=False,
                line=dict(color='green', width=4)
            ),
            row=1, col=1, secondary_y=True
        )

    # 下方：三条灰色基线，隐藏图例
    x_min, x_max = int(df['Start'].min()), int(df['End'].max())
    y_pos = {'GISTIC': 3, 'RUBIC': 2, 'OUR': 1}
    for pos in y_pos.values():
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max], y=[pos, pos], mode='lines',
                line=dict(color='lightgray', width=2),
                showlegend=False
            ),
            row=2, col=1
        )

    # 绘制各方法显著区段（增加最小长度保护）
    for df_bg, color, label in [(df_g, 'green', 'GISTIC'),
                                (df_r, 'red', 'RUBIC'),
                                (df_o, 'blue', 'OUR')]:
        if df_bg.empty:
            continue
        sub = df_bg[df_bg['Chromosome'] == chrom]
        for _, r in sub.iterrows():
            s, e = int(r['Start']), int(r['End'])
            if e <= s:
                e = s + 1   # 保证至少有 1 的长度
            fig.add_trace(
                go.Scatter(
                    x=[s, e], y=[y_pos[label], y_pos[label]],
                    mode='lines', line=dict(color=color, width=15),
                    showlegend=False
                ),
                row=2, col=1
            )

    fig.update_yaxes(row=2, col=1, range=[0.5, 3.5], showticklabels=False)

    # 隐藏上层 X 轴
    fig.update_xaxes(visible=False, row=1, col=1)

    # 将下层 X 轴标签放到下方
    fig.update_xaxes(
        row=2, col=1,
        side='bottom',
        title_text='Genomic Position',
        title_standoff=10,
        ticks='outside', ticklen=6,
        showline=True, mirror=True
    )

    # 添加贯穿子图的竖线网格
    tick_vals = fig.layout.xaxis.tickvals
    if tick_vals:
        for x_val in tick_vals:
            fig.add_shape(
                type='line',
                xref='x', yref='paper',
                x0=x_val, x1=x_val,
                y0=0, y1=1,
                line=dict(color='lightgrey', width=1),
                layer='below'
            )

    # 整体布局调整：设置全局字体为 Times
    fig.update_layout(
        font=dict(family="Times"),
        yaxis1=dict(title="Samples", automargin=True),
        margin=dict(l=150, r=150, t=80, b=120),
        height=600, width=1600,
        title_text=f"{cancer_type} — {chrom}"
    )

    # 输出
    out_path = '/Users/sanjati/jangoTemp/temp2/pycharmD/plots/union/main.png'
    pio.write_image(fig, out_path, scale=4)
    print(f"[INFO] Saved plot to {out_path}")

if __name__ == '__main__':
    draw_static(cancer_type='BLCA', amp_del_other='del', chrom='chr2')
