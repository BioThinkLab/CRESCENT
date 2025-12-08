# app_centers_editor.py
import os
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from dash import Dash, dcc, html, Input, Output, State, no_update

BASE_DIR = "/Users/sanjati/jangoTemp/temp2/pycharmD"

# ---- 吸附阈值固定在代码中（不再在 UI 中配置）----
SNAP_TOL_BIN = 300        # bin 模式下的吸附阈值（单位：bin）
SNAP_TOL_BP  = 300_000    # genomic 模式下的吸附阈值（单位：bp）

SAMPLE_COL_START = 4
SAMPLE_COL_END = 38

def unify_column_case(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl == "chromosome":
            rename_map[col] = "Chromosome"
        elif cl == "start":
            rename_map[col] = "Start"
        elif cl == "end":
            rename_map[col] = "End"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def clip_by_percentile(mat, lower_q=2, upper_q=98):
    lo, hi = np.percentile(mat, [lower_q, upper_q])
    return np.clip(mat, lo, hi)

def format_mb(x):
    try:
        return f"{float(x)/1e6:.5f}M"
    except Exception:
        return str(x)

def region_to_bin_span(df_bins: pd.DataFrame, s: int, e: int):
    starts = df_bins["Start"].to_numpy()
    ends = df_bins["End"].to_numpy()
    i0 = np.searchsorted(ends, s, side="left")
    i1 = np.searchsorted(starts, e, side="right") - 1
    n = len(df_bins)
    i0 = max(0, min(i0, n - 1))
    i1 = max(0, min(i1, n - 1))
    if i1 < i0:
        i0, i1 = i1, i0
    return i0, i1

def load_tsv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, sep="\t")
        return unify_column_case(df)
    return pd.DataFrame()

def ensure_tc_dict(d):
    if d is None:
        d = {}
    d.setdefault("TYPE_CENTERS", {})
    return d

app = Dash(__name__)
# 允许 duplicate 回调在首帧执行（Dash 2.9+ 支持）
app.config.prevent_initial_callbacks = "initial_duplicate"

SERVER_TITLE = "Type Centers Visual Editor"
CHROMS = [f"chr{i}" for i in range(1, 23)]

app.layout = html.Div([
    html.H2(SERVER_TITLE),
    html.Div([
        html.Div([html.Label("Cancer Type"),
                  dcc.Input(id="inp-cancer", value="BLCA", debounce=True, style={"width": "140px"})],
                 style={"display": "inline-block", "margin-right": "16px"}),
        html.Div([html.Label("CNV 类型"),
                  dcc.Dropdown(id="dd-type", clearable=False,
                               options=[{"label": "AMP", "value": "amp"}, {"label": "DEL", "value": "del"}],
                               value="amp", style={"width": "120px"})],
                 style={"display": "inline-block", "margin-right": "16px"}),
        html.Div([html.Label("Chromosome"),
                  dcc.Dropdown(id="dd-chrom", clearable=False,
                               options=[{"label": c, "value": c} for c in CHROMS],
                               value="chr1", style={"width": "140px"})],
                 style={"display": "inline-block", "margin-right": "16px"}),
        html.Div([html.Label("X 轴模式"),
                  dcc.Dropdown(id="dd-xmode", clearable=False,
                               options=[{"label": "Bin Index（固定宽）", "value": "bin"},
                                        {"label": "Genomic（真实坐标）", "value": "genomic"}],
                               value="bin", style={"width": "200px"})],
                 style={"display": "inline-block", "margin-right": "16px"}),
        html.Div([html.Label("YAML 路径"),
                  dcc.Input(id="inp-yaml", value="../gen_dataset/type_centers_index.yaml", debounce=True,
                            style={"width": "320px"}),
                  html.Button("加载 YAML", id="btn-load-yaml", n_clicks=0, style={"margin-left": "8px"})],
                 style={"display": "inline-block", "margin-right": "16px", "verticalAlign": "top"}),
    ], style={"margin": "8px 0"}),

    html.Hr(),

    html.Div([
        dcc.Checklist(
            id="add-mode-toggle",
            options=[{"label": "进入添加模式 (Add Mode)", "value": "add"}],
            value=[],
            inputStyle={"margin-right": "8px"},
            labelStyle={"display": "inline-block", "margin-right": "16px"}
        ),
        dcc.RadioItems(
            id="add-to-label",
            options=[{"label": "添加到 pos", "value": "pos"},
                     {"label": "添加到 neg", "value": "neg"}],
            value="pos", inline=True, style={"margin-left": "16px", "margin-right": "16px"}
        ),

        html.Button("确认添加", id="confirm-add-btn", n_clicks=0, disabled=True, style={"margin-right": "8px"}),
        html.Button("删除所选", id="delete-center-btn", n_clicks=0, disabled=True, style={"margin-right": "8px"}),
        html.Button("复位选择", id="reset-select-btn", n_clicks=0, style={"margin-right": "16px"}),
        html.Button("保存到YAML", id="save-yaml-btn", n_clicks=0),
        html.Span(id="status-text", style={"margin-left": "12px", "color": "#555"}),
    ], style={"margin": "8px 0"}),

    dcc.Store(id="centers-store"),
    dcc.Store(id="selection-store"),
    dcc.Store(id="provisional-store"),

    dcc.Graph(id="chr-graph", figure=go.Figure(), clear_on_unhover=True)
], style={"maxWidth": "2000px", "margin": "10px auto"})


# === 1) 加载 YAML ===
@app.callback(
    Output("centers-store", "data"),
    Output("status-text", "children"),
    Input("btn-load-yaml", "n_clicks"),
    State("inp-yaml", "value"),
    prevent_initial_call=True,
)
def on_load_yaml(nc, yaml_path):
    if not nc:
        return no_update, no_update
    try:
        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        data = ensure_tc_dict(data)
        data["_centers_yaml_path"] = yaml_path
        return data, f"已加载: {yaml_path}"
    except Exception as e:
        return no_update, f"加载失败: {e}"


# === 2) 绘图（含 uirevision 保持缩放 & 预览线） ===
@app.callback(
    Output("chr-graph", "figure"),
    Output("centers-store", "data", allow_duplicate=True),
    Input("inp-cancer", "value"),
    Input("dd-type", "value"),
    Input("dd-chrom", "value"),
    Input("dd-xmode", "value"),
    Input("centers-store", "data"),
    Input("provisional-store", "data"),
    State("selection-store", "data"),
)
def update_figure(cancer_type, f_type, chrom, x_mode, centers_state, provisional, selection):
    centers_state = ensure_tc_dict(centers_state or {})
    centers_state["_current_type"] = cancer_type
    centers_state["_current_chrom"] = chrom
    centers_state["_x_mode"] = x_mode

    main_file = os.path.join(
        BASE_DIR,
        "bin_with_case_amp_compressed" if f_type == "amp" else "bin_with_case_del_compressed",
        cancer_type,
        f"cnv_{chrom}" + (".txt" if f_type == "amp" else ".tsv")
    )
    df = load_tsv(main_file)
    fig = go.Figure()

    subtitle = f"{cancer_type} / {chrom}  |  file: {os.path.relpath(main_file, BASE_DIR)}"

    if df.empty or "Start" not in df.columns or "End" not in df.columns:
        fig.update_layout(
            title=f"{cancer_type} — {chrom}（{f_type.upper()}）",
            annotations=[dict(text=f"数据缺失或列名不完整：{subtitle}", x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=14, color="crimson"))],
            # 关键：即便无数据，也设置 uirevision，保持交互状态一致
            uirevision=f"{cancer_type}-{chrom}-{x_mode}",
        )
        return fig, centers_state

    df = df.fillna(0).sort_values(["Start", "End"], kind="mergesort")
    n_bins = len(df)
    x_bin = np.arange(n_bins)
    x_mid = ((df["Start"] + df["End"]) / 2).to_numpy()

    x_mode = x_mode or "bin"
    x_for_plot = x_bin if x_mode == "bin" else x_mid
    xref_name = "x"

    sample_cols = df.columns[SAMPLE_COL_START:SAMPLE_COL_END].tolist()
    mat = df[sample_cols].to_numpy(dtype=float).T
    mat_clip = clip_by_percentile(mat, 2, 98)

    start_vec = df["Start"].to_numpy()
    end_vec = df["End"].to_numpy()
    start_mb = np.array([format_mb(v) for v in start_vec])
    end_mb = np.array([format_mb(v) for v in end_vec])
    bin_str = x_bin.astype(str)
    cd_cols = np.stack([start_mb, end_mb, bin_str], axis=1)
    customdata_hm = np.tile(cd_cols[None, :, :], (mat_clip.shape[0], 1, 1))

    if x_mode == "bin":
        hm_hover = "bin=%{x}<br>start=%{customdata[0]}<br>end=%{customdata[1]}<br>value=%{z}<extra></extra>"
    else:
        hm_hover = "genomic=%{x}<br>bin=%{customdata[2]}<br>start=%{customdata[0]}<br>end=%{customdata[1]}<br>value=%{z}<extra></extra>"

    fig.add_trace(go.Heatmap(
        x=x_for_plot, y=[f"s{i}" for i in range(mat_clip.shape[0])],
        z=mat_clip, customdata=customdata_hm, hovertemplate=hm_hover,
        showscale=False, name="matrix"
    ))

    if "prob" in df.columns:
        customdata_line = cd_cols
        if x_mode == "bin":
            line_hover = "bin=%{x}<br>start=%{customdata[0]}<br>end=%{customdata[1]}<br>prob=%{y:.3f}<extra></extra>"
        else:
            line_hover = "genomic=%{x}<br>bin=%{customdata[2]}<br>start=%{customdata[0]}<br>end=%{customdata[1]}<br>prob=%{y:.3f}<extra></extra>"
        fig.add_trace(go.Scatter(
            x=x_for_plot, y=df["prob"], mode="lines", name="prob",
            line=dict(color="red", width=2.5), yaxis="y2",
            customdata=customdata_line, hovertemplate=line_hover
        ))

    centers = (centers_state.get("TYPE_CENTERS") or {}).get(cancer_type, {}).get(chrom, {})
    shapes = []

    def _x_from_value(v):
        return v if x_mode == "bin" else float(v)

    # 既有 centers
    for lab, color, dash in (("pos", "white", "solid"), ("neg", "white", "dash")):
        for v in centers.get(lab, []) or []:
            x = _x_from_value(v)
            shapes.append(dict(
                type="line", xref=xref_name, yref="paper",
                x0=x, x1=x, y0=0, y1=1, line=dict(color=color, width=2, dash=dash)
            ))

    # 选中中心（黑实线）
    if selection and selection.get("type") == cancer_type and selection.get("chrom") == chrom:
        x = _x_from_value(selection["value"])
        shapes.append(dict(
            type="line", xref=xref_name, yref="paper",
            x0=x, x1=x, y0=0, y1=1, line=dict(color="black", width=3)
        ))

    # 预览中心（与目标标签同色的点线）
    if provisional and provisional.get("type") == cancer_type and provisional.get("chrom") == chrom:
        x = _x_from_value(provisional["value"])
        lab = provisional.get("label", "pos")
        prev_color = "green" if lab == "pos" else "red"
        shapes.append(dict(
            type="line", xref=xref_name, yref="paper",
            x0=x, x1=x, y0=0, y1=1,
            line=dict(color=prev_color, width=2, dash="dot")
        ))

    fig.update_layout(
        title=f"{cancer_type} — {chrom}（{f_type.upper()}）",
        shapes=shapes,
        xaxis=dict(title="Bin Index" if x_mode == "bin" else "Genomic Position"),
        yaxis=dict(title="", showticklabels=False),
        yaxis2=dict(title="prob", overlaying="y", side="right", rangemode="tozero"),
        height=520,
        margin=dict(l=70, r=80, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # ★ 关键：保持缩放/平移/选中状态
        uirevision=f"{cancer_type}-{chrom}-{x_mode}",
    )
    fig.add_annotation(
        x=1, y=-0.22, xref="paper", yref="paper", showarrow=False,
        text=f"{cancer_type} / {chrom}  |  file: {os.path.relpath(main_file, BASE_DIR)}",
        font=dict(size=11, color="#666"), xanchor="right"
    )
    return fig, centers_state


# === 3) 图上点击：添加模式 -> 产生预览；非添加模式 -> 吸附选择 ===
@app.callback(
    Output("selection-store", "data"),
    Output("provisional-store", "data"),
    Output("status-text", "children", allow_duplicate=True),
    Output("confirm-add-btn", "disabled"),
    Output("delete-center-btn", "disabled"),
    Input("chr-graph", "clickData"),
    State("add-mode-toggle", "value"),
    State("add-to-label", "value"),
    State("centers-store", "data"),
    prevent_initial_call=True
)
def on_click_graph(clickData, add_mode, add_to_label, centers_state):
    if not clickData or not centers_state:
        return no_update, no_update, no_update, True, True

    cancer_type = centers_state.get("_current_type")
    chrom = centers_state.get("_current_chrom")
    x_mode = centers_state.get("_x_mode", "bin")
    if not cancer_type or not chrom:
        return no_update, no_update, "请先选择癌种与染色体", True, True

    point = clickData["points"][0]
    x_clicked = float(point["x"])

    label_dict = (centers_state.get("TYPE_CENTERS", {})
                  .get(cancer_type, {})
                  .get(chrom, {})) or {}

    def nearest_center(ldict, x):
        best = None
        for lab in ("pos", "neg"):
            arr = np.array(ldict.get(lab, []) or [], dtype=float)
            if arr.size == 0:
                continue
            d = np.abs(arr - x)
            i = int(d.argmin())
            cand = (lab, i, float(arr[i]), float(d[i]))
            if best is None or cand[3] < best[3]:
                best = cand
        return best

    # 添加模式：只产生预览
    if "add" in (add_mode or []):
        prov = dict(type=cancer_type, chrom=chrom, label=add_to_label, value=x_clicked)
        status = f"预览: 准备添加 {add_to_label} center @ {x_clicked:.2f}（点『确认添加』才生效，保存才写入文件）"
        return None, prov, status, False, True

    # 选择模式：吸附最近的 center
    hit = nearest_center(label_dict, x_clicked)
    if not hit:
        return None, None, "该染色体暂无 center", True, True

    lab, idx, val, dist = hit
    tol = float(SNAP_TOL_BIN) if x_mode == "bin" else float(SNAP_TOL_BP)
    if dist <= tol:
        sel = dict(type=cancer_type, chrom=chrom, label=lab, index=idx, value=val)
        status = f"选中 {lab} center @ {val:.2f}（可『删除所选』，或『复位选择』取消）"
        return sel, None, status, True, False
    else:
        return None, None, f"附近未找到可吸附的 center（最近距离 {dist:.2f} 超出阈值 {tol:.0f}）", True, True


# === 4) 确认添加 ===
@app.callback(
    Output("centers-store", "data", allow_duplicate=True),
    Output("provisional-store", "data", allow_duplicate=True),
    Output("status-text", "children", allow_duplicate=True),
    Input("confirm-add-btn", "n_clicks"),
    State("provisional-store", "data"),
    State("centers-store", "data"),
    prevent_initial_call=True
)
def confirm_add(nc, prov, centers_state):
    if not nc:
        return no_update, no_update, no_update
    if not prov:
        return no_update, no_update, "没有待确认的预览中心"

    t, c, lab, val = prov["type"], prov["chrom"], prov["label"], float(prov["value"])
    centers_state = ensure_tc_dict(centers_state or {})
    centers_state["TYPE_CENTERS"].setdefault(t, {}).setdefault(c, {}).setdefault(lab, [])
    arr = centers_state["TYPE_CENTERS"][t][c][lab]
    arr.append(val)
    arr.sort()
    return centers_state, None, f"已添加 {lab} center @ {val:.2f}（尚未写入文件，点『保存到YAML』才会落盘）"


# === 5) 删除选中 ===
@app.callback(
    Output("centers-store", "data", allow_duplicate=True),
    Output("selection-store", "data", allow_duplicate=True),
    Output("status-text", "children", allow_duplicate=True),
    Input("delete-center-btn", "n_clicks"),
    State("selection-store", "data"),
    State("centers-store", "data"),
    prevent_initial_call=True
)
def delete_selected(nc, sel, centers_state):
    if not nc:
        return no_update, no_update, no_update
    if not sel:
        return no_update, no_update, "当前没有选中的 center 可删除"

    t, c, lab, idx, val = sel["type"], sel["chrom"], sel["label"], int(sel["index"]), float(sel["value"])
    arr = (centers_state.get("TYPE_CENTERS", {})
           .get(t, {})
           .get(c, {})
           .get(lab, []))
    if not arr:
        return no_update, None, "找不到要删除的 center"
    try:
        arr.pop(idx)
        msg = f"已删除 {lab} center @ {val:.2f}（尚未写入文件，点『保存到YAML』才会落盘）"
        return centers_state, None, msg
    except Exception:
        return no_update, None, "删除失败：索引无效"


# === 6) 复位选择/预览 ===
@app.callback(
    Output("selection-store", "data", allow_duplicate=True),
    Output("provisional-store", "data", allow_duplicate=True),
    Output("status-text", "children", allow_duplicate=True),
    Input("reset-select-btn", "n_clicks"),
    prevent_initial_call=True
)
def reset_selection(nc):
    if not nc:
        return no_update, no_update, no_update
    return None, None, "已复位当前选择/预览"


# === 7) 保存到 YAML ===
@app.callback(
    Output("status-text", "children", allow_duplicate=True),
    Input("save-yaml-btn", "n_clicks"),
    State("centers-store", "data"),
    prevent_initial_call=True
)
def save_yaml(nc, centers_state):
    if not nc:
        return no_update
    centers_state = centers_state or {}
    out_path = centers_state.get("_centers_yaml_path", "./center_position.yaml")
    try:
        data_to_dump = {"TYPE_CENTERS": centers_state.get("TYPE_CENTERS", {})}
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_to_dump, f, allow_unicode=True, sort_keys=True)
        return f"已保存到 {out_path}"
    except Exception as e:
        return f"保存失败: {e}"


if __name__ == "__main__":
    app.run_server(debug=True)
