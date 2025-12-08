import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ------------------------- 公共工具 -------------------------
def unify_column_case(df):
    """
    将 df 中与 'chromosome', 'start', 'end' 等关键列同义的列统一重命名为
    'Chromosome', 'Start', 'End'。其余列名保持不变。
    """
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'chromosome':
            rename_map[col] = 'Chromosome'
        elif col_lower == 'start':
            rename_map[col] = 'Start'
        elif col_lower == 'end':
            rename_map[col] = 'End'
    df.rename(columns=rename_map, inplace=True)
    return df

def clip_by_percentile(mat, lower_q=2, upper_q=98):
    """
    将矩阵 mat 的值裁剪到 [第 lower_q 百分位, 第 upper_q 百分位] 之间，
    剔除最极端的 (lower_q)% 和 (100-upper_q)% 的值。
    """
    lo, hi = np.percentile(mat, [lower_q, upper_q])
    return np.clip(mat, lo, hi)

def format_mb(x):
    """将基因组坐标值格式化为 M 单位字符串，如 1234000 -> '1.23456M'"""
    try:
        return f"{float(x)/1e6:.5f}M"
    except Exception:
        return str(x)

def region_to_bin_span(df_bins, s, e):
    """
    把 [s, e] 基因组区间映射到 df_bins 的 bin 索引区间 [i0, i1]。
    假设 df_bins 已按 Start/End 排序。
    """
    starts = df_bins['Start'].to_numpy()
    ends   = df_bins['End'].to_numpy()
    i0 = np.searchsorted(ends, s, side='left')
    i1 = np.searchsorted(starts, e, side='right') - 1
    n = len(df_bins)
    i0 = max(0, min(i0, n-1))
    i1 = max(0, min(i1, n-1))
    if i1 < i0:
        i0, i1 = i1, i0
    return i0, i1

# ------------------------- 合并后的主函数 -------------------------
def draw_interactive(
    cancer_type,
    type,                      # 'amp' / 'del' / other(amp+del路径按你原逻辑)
    chrom_list=None,
    lower_q=2,
    upper_q=98,
    mode="rich",               # 'rich' 对应第一段；'lite' 对应第二段
    x_mode="bin"               # 仅在 mode='rich' 时生效：'bin' 或 'genomic'
):
    """
    合并版：
    - mode='rich'：对应你第一段脚本的“丰富版本”
        * 支持 x_mode：
            - 'bin'：每个片段固定宽度；泳道区域会映射到 bin 索引（更像矩阵）
            - 'genomic'：按真实坐标；泳道按真实坐标绘制
        * hover 中显示 bin / start / end（start/end 用 M 单位）
    - mode='lite'：对应你第二段脚本的“简化版本”
        * x 轴使用每个 bin 的中点（真实坐标），泳道直接用真实坐标
        * 注释显示原始数值（可自行改成 M 单位）

    其他一致点：
    - 第一行（row=1）：热力图 + （可选）prob 折线（secondary_y）
    - 第二行（row=2）：三条泳道（GISTIC / RUBIC / Another）+ 染色体臂灰底
    - 22 个按钮切换染色体
    """
    if chrom_list is None:
        chrom_list = [f"chr{i}" for i in range(1, 23)]

    # 根据 type 参数决定路径
    if type == "amp":
        g_type, f_type = 'amp', 'amp'
    elif type == "del":
        g_type, f_type = 'del', 'del'
    else:
        g_type, f_type = 'del', 'amp'

    base = "/Users/sanjati/jangoTemp/temp2/pycharmD"

    def load_tsv(path):
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t')
            return unify_column_case(df)
        else:
            print(f"[DEBUG] File not found: {path}")
            return pd.DataFrame()

    # 参考区域与染色体臂
    df_gistic    = load_tsv(f"{base}/shin_Data/{cancer_type}/{g_type}/{cancer_type}_gistic_{g_type}.tsv")
    df_rubic     = load_tsv(f"{base}/shin_Data/{cancer_type}/{g_type}/{cancer_type}_rubic_{g_type}.tsv")
    df_another   = load_tsv(f"{base}/shin_Data/{cancer_type}/{g_type}/{cancer_type}_om_{g_type}.tsv")
    df_chrom_arm = load_tsv(f"{base}/Data/ChromosomeData2025/GRCh38_Chromosome_Arm_Ranges.tsv")

    print("file source:\n"
          f"df_om:{base}/shin_Data/{cancer_type}/{g_type}/{cancer_type}_om_{g_type}.tsv")

    # 2x1 子图：第一行 secondary_y；第二行为泳道
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": False}]],
        row_heights=[0.85, 0.15],
        shared_xaxes=True,
        vertical_spacing=0.02
    )

    all_traces = []
    use_secondary = []
    chrom_info = {}

    # 泳道参数
    lane_height = 1/3
    lane_map = {'gistic': 2, 'rubic': 1, 'another': 0}

    # 逐染色体生成
    for chrom in chrom_list:
        trace_idxs = []
        shapes = []
        annots = []

        main_file = (
            f"{base}/"
            + ("bin_with_case_amp_compressed" if f_type=='amp' else "bin_with_case_del_compressed")
            + f"/{cancer_type}/cnv_{chrom}"
            + (".txt" if f_type=='amp' else ".tsv")
        )
        print(f"[INFO] Loading {main_file}")
        if not os.path.exists(main_file):
            print(f"[DEBUG] Missing data for {chrom}")
            continue

        df = pd.read_csv(main_file, sep='\t')
        unify_column_case(df)
        df.fillna(0, inplace=True)
        # 稳定排序，保证 bin 顺序一致
        df.sort_values(['Start','End'], inplace=True, kind='mergesort')

        # 统一样本列选择（与你原逻辑保持一致）
        samples = df.columns[4:38].tolist()
        mat = df[samples].values.T
        mat_clipped = clip_by_percentile(mat, lower_q, upper_q)

        # --- 不同模式下的 x 轴 / hover / swimlane 行为 ---
        if mode == "rich":
            # 支持 x_mode：'bin' / 'genomic'
            n_bins = len(df)
            x_bin  = np.arange(n_bins)                    # 0..n-1
            x_mid  = ((df['Start'] + df['End']) / 2).to_numpy()

            if x_mode == 'bin':
                # 固定宽度 bin
                x_for_plot = x_bin
                xref_name = "x1"
                # 自定义 hover：显示 bin / start(M) / end(M)
                start_mb = np.array([format_mb(v) for v in df['Start'].to_numpy()])
                end_mb   = np.array([format_mb(v) for v in df['End'].to_numpy()])
                bin_str  = x_bin.astype(str)
                cd_cols  = np.stack([start_mb, end_mb, bin_str], axis=1)     # (n_bins, 3)
                customdata_hm = np.tile(cd_cols[None, :, :], (mat_clipped.shape[0], 1, 1))
                hm_hover = (
                    "sample=%{y}<br>"
                    "bin=%{x}<br>"
                    "start=%{customdata[0]}<br>"
                    "end=%{customdata[1]}<br>"
                    "value=%{z}<extra></extra>"
                )
            else:
                # 真实坐标
                x_for_plot = x_mid
                xref_name = "x1"
                start_mb = np.array([format_mb(v) for v in df['Start'].to_numpy()])
                end_mb   = np.array([format_mb(v) for v in df['End'].to_numpy()])
                bin_str  = x_bin.astype(str)
                cd_cols  = np.stack([start_mb, end_mb, bin_str], axis=1)
                customdata_hm = np.tile(cd_cols[None, :, :], (mat_clipped.shape[0], 1, 1))
                hm_hover = (
                    "sample=%{y}<br>"
                    "genomic=%{x}<br>"
                    "bin=%{customdata[2]}<br>"
                    "start=%{customdata[0]}<br>"
                    "end=%{customdata[1]}<br>"
                    "value=%{z}<extra></extra>"
                )

            # 热力图（rich）
            hm = go.Heatmap(
                x=x_for_plot.tolist(), y=samples, z=mat_clipped.tolist(),
                visible=False, showscale=False,
                customdata=customdata_hm,
                hovertemplate=hm_hover
            )
            all_traces.append(hm); use_secondary.append(False)
            trace_idxs.append(len(all_traces)-1)

            # prob 折线（rich）
            if 'prob' in df.columns:
                customdata_line = cd_cols
                if x_mode == 'bin':
                    line_hover = (
                        "bin=%{x}<br>"
                        "start=%{customdata[0]}<br>"
                        "end=%{customdata[1]}<br>"
                        "prob=%{y}<extra></extra>"
                    )
                else:
                    line_hover = (
                        "genomic=%{x}<br>"
                        "bin=%{customdata[2]}<br>"
                        "start=%{customdata[0]}<br>"
                        "end=%{customdata[1]}<br>"
                        "prob=%{y}<extra></extra>"
                    )
                line = go.Scatter(
                    x=x_for_plot.tolist(), y=df['prob'].tolist(),
                    mode='lines', name='prob',
                    visible=False, line=dict(color="green", width=3),
                    customdata=customdata_line,
                    hovertemplate=line_hover
                )
                all_traces.append(line); use_secondary.append(True)
                trace_idxs.append(len(all_traces)-1)

            # 泳道与注释（rich）
            offset = 0.05
            spacing = 0.06
            for df_bg, color, name in [
                (df_gistic, 'green',   'gistic'),
                (df_rubic,  'red',     'rubic'),
                (df_another,'blue',    'another')
            ]:
                lane = lane_map[name]
                y0 = lane * lane_height
                y1 = (lane + 1) * lane_height
                if df_bg.empty:
                    continue
                sub = df_bg[df_bg['Chromosome'] == chrom]
                for _, r in sub.iterrows():
                    s, e = int(r['Start']), int(r['End'])
                    if x_mode == 'bin':
                        # 把 [s,e] 映射到 bin 索引并画等宽矩形
                        i0, i1 = region_to_bin_span(df, s, e)
                        x0 = i0 - 0.5
                        x1 = i1 + 0.5
                    else:
                        x0, x1 = s, e
                    shapes.append(dict(
                        type="rect",
                        xref=xref_name, yref="y3",
                        x0=x0, x1=x1, y0=y0, y1=y1,
                        fillcolor=color, opacity=0.4, line_width=0
                    ))
                    annots.append(dict(
                        x=(x0 + x1) / 2,
                        y=-(offset + lane*spacing),
                        xref=xref_name, yref="paper",
                        text=f"{format_mb(s)}-{format_mb(e)}",
                        showarrow=False,
                        xanchor="center", yanchor="top",
                        font=dict(size=10),
                        bgcolor=color, bordercolor="black"
                    ))

            # 染色体臂灰底（rich：若 x_mode='bin' 也做映射）
            if not df_chrom_arm.empty and 'Chromosome' in df_chrom_arm.columns:
                arms = df_chrom_arm[df_chrom_arm['Chromosome'] == chrom]
                for _, r in arms.iterrows():
                    e0, s1 = int(r['end0']), int(r['start1'])
                    if x_mode == 'bin':
                        i0, _ = region_to_bin_span(df, e0, e0)
                        _, i1 = region_to_bin_span(df, s1, s1)
                        x0 = i0 - 0.5
                        x1 = i1 + 0.5
                    else:
                        x0, x1 = e0, s1
                    shapes.append(dict(
                        type="rect",
                        xref=xref_name, yref="y3",
                        x0=x0, x1=x1, y0=0, y1=1,
                        fillcolor="gray", opacity=0.3, line_width=0
                    ))

            chrom_info[chrom] = dict(
                indices=trace_idxs,
                shapes=shapes,
                annotations=annots,
                title=f"{cancer_type} — {chrom} ({'Fixed-width bins' if x_mode=='bin' else 'Genomic'})"
            )

        else:
            # mode == 'lite'：对应第二段脚本
            mid = ((df['Start'] + df['End']) / 2).tolist()
            z = clip_by_percentile(mat, lower_q, upper_q).tolist()

            # 热力图（lite）
            hm = go.Heatmap(x=mid, y=samples, z=z, visible=False, showscale=False)
            all_traces.append(hm); use_secondary.append(False)
            trace_idxs.append(len(all_traces)-1)

            # prob 折线（lite）
            if 'prob' in df.columns:
                line = go.Scatter(
                    x=mid, y=df['prob'],
                    mode='lines', name='prob', visible=False,
                    line=dict(color="green", width=5)
                )
                all_traces.append(line); use_secondary.append(True)
                trace_idxs.append(len(all_traces)-1)

            # 泳道 + 底部段注释（lite：直接用真实坐标，注释用原值）
            offset = 0.05
            spacing = 0.06
            for df_bg, color, name in [
                (df_gistic, 'green',   'gistic'),
                (df_rubic,  'red',     'rubic'),
                (df_another,'blue',    'another')
            ]:
                lane = lane_map[name]
                y0 = lane * lane_height
                y1 = (lane + 1) * lane_height
                if not df_bg.empty:
                    sub = df_bg[df_bg['Chromosome'] == chrom]
                    for _, r in sub.iterrows():
                        s, e = int(r['Start']), int(r['End'])
                        shapes.append(dict(
                            type="rect",
                            xref="x1", yref="y3",
                            x0=s, x1=e, y0=y0, y1=y1,
                            fillcolor=color, opacity=0.4, line_width=0
                        ))
                        annots.append(dict(
                            x=(s+e)/2,
                            y=-(offset + lane*spacing),
                            xref="x1", yref="paper",
                            text=f"{s}-{e}",
                            showarrow=False,
                            xanchor="center", yanchor="top",
                            font=dict(size=10),
                            bgcolor=color, bordercolor="black"
                        ))

            # 染色体臂灰底（lite：真实坐标）
            arms = df_chrom_arm[df_chrom_arm['Chromosome'] == chrom]
            for _, r in arms.iterrows():
                e0, s1 = int(r['end0']), int(r['start1'])
                shapes.append(dict(
                    type="rect",
                    xref="x1", yref="y3",
                    x0=e0, x1=s1, y0=0, y1=1,
                    fillcolor="gray", opacity=0.3, line_width=0
                ))

            chrom_info[chrom] = dict(
                indices=trace_idxs,
                shapes=shapes,
                annotations=annots,
                title=f"{cancer_type} — {chrom}"
            )

    # 将所有 trace 加到图里
    for trace, sec in zip(all_traces, use_secondary):
        fig.add_trace(trace, row=1, col=1, secondary_y=sec)

    # 第二行放一个透明 dummy trace 来承载泳道（y3）
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=1, color="rgba(0,0,0,0)"),
        showlegend=False
    ), row=2, col=1)

    # 按钮：按染色体切换可见性 + shapes/annotations/title
    buttons = []
    N = len(all_traces)
    for chrom in chrom_list:
        if chrom not in chrom_info:
            continue
        idxs = chrom_info[chrom]['indices']
        vis = [j in idxs for j in range(N)] + [True]  # 最后一个 dummy 永远 True
        buttons.append(dict(
            method="update",
            label=chrom.replace('chr',''),
            args=[
                {"visible": vis},
                {"shapes": chrom_info[chrom]['shapes'],
                 "annotations": chrom_info[chrom]['annotations'],
                 "title.text": chrom_info[chrom]['title']}
            ]
        ))

    # 默认显示第一条染色体
    first = next(iter(chrom_info))
    for j in range(N):
        fig.data[j].visible = j in chrom_info[first]['indices']
    fig.data[-1].visible = True

    # 布局
    fig.update_layout(
        shapes=chrom_info[first]['shapes'],
        annotations=chrom_info[first]['annotations'],
        title_text=chrom_info[first]['title'],
        updatemenus=[dict(
            type="buttons", direction="right",
            buttons=buttons,
            x=0.5, y=-0.2, xanchor="center", yanchor="top",
            showactive=True
        )],
        xaxis1=dict(title="" if mode=="rich" and x_mode=="bin" else ""),
        yaxis1=dict(title="Samples"),
        yaxis2=dict(title="prob", overlaying="y", side="right"),
        yaxis3=dict(visible=False),
        height=1000,
        width=2000,
        margin=dict(l=80, r=80, t=100, b=380)
    )

    # 泳道 y 轴刻度
    fig.update_yaxes(
        row=2, col=1,
        title_text="Regions",
        tickmode='array',
        tickvals=[0.5*lane_height, 1.5*lane_height, 2.5*lane_height],
        ticktext=['GISTIC', 'RUBIC', 'Another'],
        showgrid=False,
        range=[0,1]
    )

    # 泳道右端标签
    for name, lane in lane_map.items():
        y0 = lane * lane_height
        y1 = (lane + 1) * lane_height
        fig.add_annotation(
            x=1.01, y=(y0 + y1) / 2,
            xref='paper', yref='y3',
            text=name.capitalize(),
            showarrow=False,
            xanchor='left', yanchor='middle',
            font=dict(size=12)
        )

    # 最下方横轴总标签
    fig.add_annotation(
        x=0.5, y=-0.04,
        xref='paper', yref='paper',
        text=('Bin Index' if (mode=="rich" and x_mode=='bin') else 'Genomic Position'),
        showarrow=False,
        font=dict(size=14)
    )

    # 导出并展示
    pio.write_image(fig, './vis_plotly_regions_swimlanes.png')
    fig.show()

# ------------------------- 示例 -------------------------
if __name__ == '__main__':
    # 对应第一段功能更完整的版本：
    draw_interactive(cancer_type='BLCA', type='amp', mode='rich')
    # 对应第二段更简洁的版本：
    # draw_interactive(cancer_type='BLCA', type='amp', mode='lite')
