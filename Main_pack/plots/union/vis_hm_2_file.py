# 用来画heatmap组图的(figure4下半部分)
import os
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from PIL import Image

# —— 用户配置区 ——
# CONFIG 值可以是 0（所有染色体）、单个数字或数字列表
CONFIG = [
    # {'UCEC': [8,10,17,19]},
    {'GBM': [7,12]},
    # {'BLCA': [1,6]},
]

AMP_DEL_TYPE = 'amp'  # 可为 'amp', 'del' 或 'both'
BASE = "/Users/sanjati/jangoTemp/temp2/pycharmD"
ALL_CHROMS = [f"chr{i}" for i in range(1, 23)]


def unify_column_case(df):
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc == 'chromosome': rename[col] = 'Chromosome'
        if lc == 'start':      rename[col] = 'Start'
        if lc == 'end':        rename[col] = 'End'
    df.rename(columns=rename, inplace=True)
    return df


def load_tsv(path):
    if os.path.exists(path):
        df = pd.read_csv(path, sep='\t')
        return unify_column_case(df)
    return pd.DataFrame()


def merge_intervals(segs):
    segs = sorted(segs, key=lambda x: x[0])
    merged = []
    for s, e in segs:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]


def compress_coords(df_all):
    samples = df_all.columns[4:38].tolist()
    zero_mask = (df_all[samples] == 0).all(axis=1)
    zero_segs = list(zip(df_all.loc[zero_mask, 'Start'], df_all.loc[zero_mask, 'End']))
    gaps = merge_intervals(zero_segs)

    def compress_x(x):
        shift = sum((e - s) for s, e in gaps if e <= x)
        return x - shift

    return compress_x, ~zero_mask


def merge_images_vertically(image_paths, output_path):
    imgs = [Image.open(p) for p in image_paths]
    widths, heights = zip(*(im.size for im in imgs))
    canvas = Image.new('RGB', (max(widths), sum(heights)), (255, 255, 255))
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(output_path)


def draw_and_save(cancer_type, amp_del_other, chrom_list=None):
    if chrom_list is None:
        chrom_list = ALL_CHROMS
    types = [amp_del_other] if amp_del_other in ['amp','del'] else ['amp','del']
    saved_paths = []

    for t in types:
        # --- 在这里先构造三条子文件的路径 ---
        g_path = f"{BASE}/shin_Data/{cancer_type}/{t}/{cancer_type}_gistic_{t}.tsv"
        r_path = f"{BASE}/shin_Data/{cancer_type}/{t}/{cancer_type}_rubic_{t}.tsv"
        o_path = f"{BASE}/shin_Data/{cancer_type}/{t}/{cancer_type}_om_{t}.tsv"

        # --- 只在文件存在时才 load，否则置为 None ---
        df_g = load_tsv(g_path) if os.path.exists(g_path) else None
        df_r = load_tsv(r_path) if os.path.exists(r_path) else None
        df_o = load_tsv(o_path) if os.path.exists(o_path) else None

        for chrom in chrom_list:
            main_file = (
                f"{BASE}/"
                + ("bin_with_case_amp_compressed" if t=='amp' else "bin_with_case_del")
                + f"/{cancer_type}/cnv_{chrom}"
                + (".txt" if t=='amp' else ".tsv")
            )
            if not os.path.exists(main_file):
                continue

            df_all = pd.read_csv(main_file, sep='\t')
            unify_column_case(df_all)
            df_all.fillna(0, inplace=True)
            df_all.sort_values(['Start','End'], inplace=True)

            compress_x, valid_mask = compress_coords(df_all)
            df = df_all.loc[valid_mask].reset_index(drop=True)
            mid = [(s+e)/2 for s,e in zip(df['Start'],df['End'])]
            mid = [compress_x(x) for x in mid]
            samples = df.columns[4:38].tolist()
            z = [df[s].tolist() for s in samples]

            dom_start = compress_x(int(df_all['Start'].min()))
            dom_end   = compress_x(int(df_all['End'].max()))
            def get_segs_from(df_bg):
                segs = [(int(r['Start']), int(r['End']))
                        for _,r in df_bg[df_bg['Chromosome']==chrom].iterrows()]
                return [(compress_x(s), compress_x(e)) for s,e in merge_intervals(segs)]
            def complement(segs):
                comps, cur = [], dom_start
                for s,e in sorted(segs):
                    if s>cur: comps.append((cur,s))
                    cur = max(cur,e)
                if cur<dom_end: comps.append((cur,dom_end))
                return comps

            # --- 开始画图 ---
            fig = make_subplots(
                rows=2, cols=1,
                specs=[[{"secondary_y":True}], [{}]],
                row_heights=[0.7,0.3], shared_xaxes=True, vertical_spacing=0.02
            )
            # 热图
            fig.add_trace(go.Heatmap(
                x=mid, y=samples, z=z,
                showscale=False, hoverinfo='none'),
                row=1, col=1
            )
            # prob 曲线（如果有）
            if 'prob' in df.columns:
                fig.add_trace(go.Scatter(
                    x=mid, y=df['prob'], mode='lines',
                    line=dict(color='green', width=2), hoverinfo='none'),
                    row=1, col=1, secondary_y=True
                )

            # --- 只有在 df_g 不为 None 时才画 GISTIC 轨道 ---
            if df_g is not None:
                g_segs = get_segs_from(df_g)
                g_gaps = complement(g_segs)
                for s,e in g_gaps:
                    fig.add_trace(go.Scatter(
                        x=[s,e], y=[3,3], mode='lines',
                        line=dict(color='lightgray', width=15),
                        hoverinfo='none', showlegend=False),
                        row=2, col=1)
                for s,e in g_segs:
                    fig.add_trace(go.Scatter(
                        x=[s,e], y=[3,3], mode='lines',
                        line=dict(color='green', width=15),
                        hoverinfo='none', showlegend=False),
                        row=2, col=1)

            # RUBIC
            if df_r is not None:
                r_segs = get_segs_from(df_r)
                r_gaps = complement(r_segs)
                for s,e in r_gaps:
                    fig.add_trace(go.Scatter(
                        x=[s,e], y=[2,2], mode='lines',
                        line=dict(color='lightgray', width=15),
                        hoverinfo='none', showlegend=False),
                        row=2, col=1)
                for s,e in r_segs:
                    fig.add_trace(go.Scatter(
                        x=[s,e], y=[2,2], mode='lines',
                        line=dict(color='red', width=15),
                        hoverinfo='none', showlegend=False),
                        row=2, col=1)

            # OM
            if df_o is not None:
                o_segs = get_segs_from(df_o)
                o_gaps = complement(o_segs)
                for s,e in o_gaps:
                    fig.add_trace(go.Scatter(
                        x=[s,e], y=[1,1], mode='lines',
                        line=dict(color='lightgray', width=15),
                        hoverinfo='none', showlegend=False),
                        row=2, col=1)
                for s,e in o_segs:
                    fig.add_trace(go.Scatter(
                        x=[s,e], y=[1,1], mode='lines',
                        line=dict(color='blue', width=15),
                        hoverinfo='none', showlegend=False),
                        row=2, col=1)

            fig.update_layout(
                showlegend=False,
                plot_bgcolor='white', paper_bgcolor='white',
                title_text=f"{cancer_type}-{chrom} ({t})",
                title_font_size=12,
                height=150, width=800,
                margin=dict(l=10, r=10, t=25, b=10)
            )
            fig.update_yaxes(showticklabels=False, row=1, col=1)
            fig.update_yaxes(showticklabels=False, row=2, col=1, range=[0.5,3.5])
            fig.update_xaxes(showgrid=False, row=2, col=1, showticklabels=False)

            out_png = os.path.join(BASE, f"{cancer_type}_{t}_{chrom}.png")
            pio.write_image(fig, out_png, scale=5)
            saved_paths.append(out_png)

    return saved_paths


if __name__ == '__main__':
    all_images = []
    for entry in CONFIG:
        ct = next(iter(entry))
        spec = entry[ct]
        if isinstance(spec, int):
            spec_list = [spec]
        else:
            spec_list = spec
        if 0 in spec_list:
            chroms = None
        else:
            chroms = [f"chr{s}" for s in spec_list]
        all_images.extend(draw_and_save(ct, AMP_DEL_TYPE, chroms))

    if all_images:
        merged_file = os.path.join('/Users/sanjati/jangoTemp/temp2/pycharmD/plots/union', f"{AMP_DEL_TYPE}_combined_all.png")
        merge_images_vertically(all_images, merged_file)
        print(f"[INFO] Saved final merged image → {merged_file}")
    else:
        print("[WARN] No images to merge.")

