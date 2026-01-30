import os
import yaml
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from dash import Dash, dcc, html, Input, Output, State, no_update

BASE_DIR = "../run"

SNAP_TOL_BIN = 300
SNAP_TOL_BP  = 300_000

SAMPLE_COL_START = 4
SAMPLE_COL_END = 38

UPLOAD_DIR = os.path.join(os.getcwd(), "uploaded_yamls")


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


def list_subdirs(path: str):
    if not os.path.isdir(path):
        return []
    out = []
    for name in os.listdir(path):
        if name.startswith("."):
            continue
        full = os.path.join(path, name)
        if os.path.isdir(full):
            out.append(name)
    return sorted(out)


def cancer_root_for_type(f_type: str) -> str:
    return os.path.join(
        BASE_DIR,
        "bin_with_case_amp" if f_type == "amp" else "bin_with_case_del",
    )


def default_cancer_options(f_type: str):
    root = cancer_root_for_type(f_type)
    cancers = list_subdirs(root)
    return [{"label": c, "value": c} for c in cancers], (cancers[0] if cancers else None)


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _timestamp_name():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _decode_upload_contents(contents: str) -> bytes:
    # contents = "data:...;base64,XXXX"
    if not contents or "," not in contents:
        return b""
    b64 = contents.split(",", 1)[1]
    return base64.b64decode(b64)


app = Dash(__name__)
app.config.prevent_initial_callbacks = "initial_duplicate"

SERVER_TITLE = "Visual Editor"
CHROMS = [f"chr{i}" for i in range(1, 23)]

init_opts, init_val = default_cancer_options("amp")
if init_val is None:
    init_val = "BLCA"

app.layout = html.Div(
    className="app-shell",
    children=[
        html.Div(
            className="header",
            children=[
                html.Div(children=[
                    html.H2(SERVER_TITLE, className="title"),
                ]),
            ],
        ),

        # Panel: Controls
        html.Div(
            className="panel",
            children=[
                html.Div(
                    className="panel-header",
                    children=[
                        html.Div("Dataset / View Controls", className="panel-title"),
                        html.Div("Tip: Zoom state is preserved via uirevision.", className="badge"),
                    ],
                ),

                html.Div(
                    className="grid",
                    children=[
                        html.Div(
                            className="field",
                            style={"gridColumn": "span 3"},
                            children=[
                                html.Div("Dataset", className="label"),
                                dcc.Dropdown(
                                    id="inp-cancer",
                                    clearable=False,
                                    options=init_opts,
                                    value=init_val,
                                ),
                            ],
                        ),

                        html.Div(
                            className="field",
                            style={"gridColumn": "span 2"},
                            children=[
                                html.Div("Mutation Type", className="label"),
                                dcc.Dropdown(
                                    id="dd-type",
                                    clearable=False,
                                    options=[{"label": "AMP", "value": "amp"},
                                             {"label": "DEL", "value": "del"}],
                                    value="amp",
                                ),
                            ],
                        ),

                        html.Div(
                            className="field",
                            style={"gridColumn": "span 2"},
                            children=[
                                html.Div("Chromosome", className="label"),
                                dcc.Dropdown(
                                    id="dd-chrom",
                                    clearable=False,
                                    options=[{"label": c, "value": c} for c in CHROMS],
                                    value="chr1",
                                ),
                            ],
                        ),

                        html.Div(
                            className="field",
                            style={"gridColumn": "span 2"},
                            children=[
                                html.Div("X-axis Mode", className="label"),
                                dcc.Dropdown(
                                    id="dd-xmode",
                                    clearable=False,
                                    options=[{"label": "Bin Index (fixed width)", "value": "bin"},
                                             {"label": "Genomic (real coordinates)", "value": "genomic"}],
                                    value="bin",
                                ),
                            ],
                        ),

                        # YAML controls
                        html.Div(
                            className="field",
                            style={"gridColumn": "span 3"},
                            children=[
                                html.Div("YAML File", className="label"),
                                html.Div(
                                    className="inline-row",
                                    children=[
                                        dcc.Input(
                                            id="inp-yaml",
                                            value="",
                                            placeholder="(no file loaded)",
                                            debounce=False,
                                            className="dcc-input",
                                            style={"flex": "1 1 520px"},
                                            readOnly=True,
                                        ),
                                        html.Button(
                                            "Create new file",
                                            id="btn-create-yaml",
                                            n_clicks=0,
                                            className="btn btn-primary",
                                        ),
                                        dcc.Upload(
                                            id="upload-yaml",
                                            accept=".yaml,.yml",
                                            multiple=False,
                                            children=html.Button(
                                                "Load file",
                                                id="btn-load-file",
                                                n_clicks=0,
                                                className="btn",
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        html.Div(className="hr"),

        # Panel: Actions
        html.Div(
            className="panel",
            children=[
                html.Div(
                    className="panel-header",
                    children=[
                        html.Div("Edit Actions", className="panel-title"),
                        html.Div("Add / Select / Delete centers", className="badge"),
                    ],
                ),

                html.Div(
                    className="inline-row",
                    children=[
                        # Add mode switch + help
                        html.Div(
                            className="inline-row",
                            style={"gap": "8px"},
                            children=[
                                dcc.Checklist(
                                    id="add-mode-toggle",
                                    options=[{"label": "Add mode", "value": "add"}],
                                    value=[],
                                    className="switch",
                                ),
                                html.Button("?", id="help-btn", n_clicks=0, className="btn btn-icon"),
                                html.Div(
                                    id="help-popover",
                                    className="popover",
                                    children=[
                                        html.Div(
                                            className="popover-head",
                                            children=[
                                                html.Div("Add mode help", className="popover-title"),
                                                html.Button("×", id="help-close", n_clicks=0, className="popover-close"),
                                            ],
                                        ),
                                        html.Div(
                                            className="popover-body",
                                            children=[
                                                html.Div("• Click in Chromosome View to preview (dotted)."),
                                                html.Div("• Red = pos, Green = neg."),
                                                html.Div("• Click Confirm Add to apply."),
                                            ],
                                        ),
                                    ],
                                    style={"display": "none"},
                                ),
                            ],
                        ),

                        # Segmented toggle (pos/neg) with visible selected feedback
                        dcc.Tabs(
                            id="add-to-label",
                            value="pos",
                            className="seg-tabs",
                            parent_className="seg-tabs-parent",
                            children=[
                                dcc.Tab(label="Add \n pos", value="pos",
                                        className="seg-tab", selected_className="seg-tab--selected"),
                                dcc.Tab(label="Add \n neg", value="neg",
                                        className="seg-tab", selected_className="seg-tab--selected"),
                            ],
                        ),

                        html.Div(style={"flex": "1 1 auto"}),

                        html.Button(
                            "Confirm Add",
                            id="confirm-add-btn",
                            n_clicks=0,
                            disabled=True,
                            className="btn btn-primary",
                        ),
                        html.Button(
                            "Delete Selected",
                            id="delete-center-btn",
                            n_clicks=0,
                            disabled=True,
                            className="btn btn-danger",
                        ),
                        html.Button(
                            "Reset Selection",
                            id="reset-select-btn",
                            n_clicks=0,
                            className="btn",
                        ),
                        html.Button(
                            "Save to YAML",
                            id="save-yaml-btn",
                            n_clicks=0,
                            className="btn btn-primary",
                        ),
                    ],
                ),
            ],
        ),

        html.Div(className="hr"),

        dcc.Store(id="centers-store"),
        dcc.Store(id="selection-store"),
        dcc.Store(id="provisional-store"),
        dcc.Store(id="help-open-store", data=False),

        # Graph
        html.Div(
            className="panel",
            children=[
                html.Div(
                    className="panel-header",
                    children=[
                        html.Div("Chromosome View", className="panel-title"),
                        html.Div(id="status-text", className="badge"),
                    ],
                ),
                html.Div(
                    className="graph-wrap",
                    children=[
                        dcc.Graph(
                            id="chr-graph",
                            figure=go.Figure(),
                            clear_on_unhover=True,
                            config={"displaylogo": False, "scrollZoom": True},
                        )
                    ],
                ),
            ],
        ),
    ],
)


# --- Help popover toggle ---
@app.callback(
    Output("help-open-store", "data"),
    Input("help-btn", "n_clicks"),
    Input("help-close", "n_clicks"),
    State("help-open-store", "data"),
    prevent_initial_call=True
)
def toggle_help_popover(n_help, n_close, is_open):
    ctx = getattr(__import__("dash"), "callback_context")
    if not ctx.triggered:
        return is_open
    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    if trig == "help-close":
        return False
    # help-btn toggles
    return not bool(is_open)


@app.callback(
    Output("help-popover", "style"),
    Input("help-open-store", "data"),
)
def render_help_popover(is_open):
    return {"display": "block"} if is_open else {"display": "none"}


# --- Refresh Cancer Type list when CNV Type changes ---
@app.callback(
    Output("inp-cancer", "options"),
    Output("inp-cancer", "value"),
    Output("status-text", "children", allow_duplicate=True),
    Input("dd-type", "value"),
    State("inp-cancer", "value"),
    prevent_initial_call=True
)
def refresh_cancer_dropdown(f_type, current_value):
    opts, default_val = default_cancer_options(f_type)
    if not opts:
        msg = f"Warning: no cancer type folders found under {os.path.relpath(cancer_root_for_type(f_type), BASE_DIR)}"
        return [], current_value, msg

    allowed = {o["value"] for o in opts}
    new_val = current_value if current_value in allowed else default_val
    msg = f"Cancer Type list updated ({len(opts)} found) for {f_type.upper()}"
    return opts, new_val, msg


# --- Create new YAML file (server-side) ---
@app.callback(
    Output("centers-store", "data"),
    Output("inp-yaml", "value"),
    Output("status-text", "children", allow_duplicate=True),
    Input("btn-create-yaml", "n_clicks"),
    prevent_initial_call=True
)
def create_new_yaml(nc):
    if not nc:
        return no_update, no_update, no_update

    fname = f"{_timestamp_name()}.yaml"
    out_path = os.path.join(os.getcwd(), fname)

    data = {"TYPE_CENTERS": {}}
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=True)

    st = ensure_tc_dict({})
    st["_centers_yaml_path"] = out_path
    return st, out_path, f"Created: {out_path}"


# --- Load YAML via file picker (Upload) ---
@app.callback(
    Output("centers-store", "data", allow_duplicate=True),
    Output("inp-yaml", "value", allow_duplicate=True),
    Output("status-text", "children", allow_duplicate=True),
    Input("upload-yaml", "contents"),
    State("upload-yaml", "filename"),
    prevent_initial_call=True
)
def load_yaml_from_upload(contents, filename):
    if not contents or not filename:
        return no_update, no_update, no_update

    try:
        raw = _decode_upload_contents(contents)
        text = raw.decode("utf-8", errors="replace")
        parsed = yaml.safe_load(text) or {}
        parsed = ensure_tc_dict(parsed)

        _safe_mkdir(UPLOAD_DIR)
        safe_name = os.path.basename(filename)
        out_path = os.path.join(UPLOAD_DIR, safe_name)

        # persist exact uploaded content
        with open(out_path, "wb") as f:
            f.write(raw)

        parsed["_centers_yaml_path"] = out_path
        return parsed, out_path, f"Loaded: {out_path}"
    except Exception as e:
        return no_update, no_update, f"Load failed: {e}"


# === Plotting ===
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
        "bin_with_case_amp" if f_type == "amp" else "bin_with_case_del",
        cancer_type,
        f"cnv_{chrom}" + (".txt" if f_type == "amp" else ".tsv")
    )
    df = load_tsv(main_file)
    fig = go.Figure()

    subtitle = f"{cancer_type} / {chrom}  |  file: {os.path.relpath(main_file, BASE_DIR)}"

    if df.empty or "Start" not in df.columns or "End" not in df.columns:
        fig.update_layout(
            title=f"{cancer_type} — {chrom} ({f_type.upper()})",
            annotations=[dict(
                text=f"Missing data or incomplete columns: {subtitle}",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14, color="crimson")
            )],
            uirevision=f"{cancer_type}-{chrom}-{x_mode}",
            height=520,
            margin=dict(l=70, r=80, t=60, b=60),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
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
        hm_hover = ("genomic=%{x}<br>bin=%{customdata[2]}<br>start=%{customdata[0]}"
                    "<br>end=%{customdata[1]}<br>value=%{z}<extra></extra>")

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
            line_hover = ("genomic=%{x}<br>bin=%{customdata[2]}<br>start=%{customdata[0]}"
                          "<br>end=%{customdata[1]}<br>prob=%{y:.3f}<extra></extra>")
        fig.add_trace(go.Scatter(
            x=x_for_plot, y=df["prob"], mode="lines", name="prob",
            line=dict(color="red", width=2.5), yaxis="y2",
            customdata=customdata_line, hovertemplate=line_hover
        ))

    centers = (centers_state.get("TYPE_CENTERS") or {}).get(cancer_type, {}).get(chrom, {})
    shapes = []

    def _x_from_value(v):
        return v if x_mode == "bin" else float(v)

    for lab, color, dash in (("pos", "white", "solid"), ("neg", "white", "dash")):
        for v in centers.get(lab, []) or []:
            x = _x_from_value(v)
            shapes.append(dict(
                type="line", xref=xref_name, yref="paper",
                x0=x, x1=x, y0=0, y1=1, line=dict(color=color, width=2, dash=dash)
            ))

    if selection and selection.get("type") == cancer_type and selection.get("chrom") == chrom:
        x = _x_from_value(selection["value"])
        shapes.append(dict(
            type="line", xref=xref_name, yref="paper",
            x0=x, x1=x, y0=0, y1=1, line=dict(color="black", width=3)
        ))

    # Preview line: red=pos, green=neg
    if provisional and provisional.get("type") == cancer_type and provisional.get("chrom") == chrom:
        x = _x_from_value(provisional["value"])
        lab = provisional.get("label", "pos")
        prev_color = "red" if lab == "pos" else "green"
        shapes.append(dict(
            type="line", xref=xref_name, yref="paper",
            x0=x, x1=x, y0=0, y1=1,
            line=dict(color=prev_color, width=2, dash="dot")
        ))

    fig.update_layout(
        title=f"{cancer_type} — {chrom} ({f_type.upper()})",
        shapes=shapes,
        xaxis=dict(title="Bin Index" if x_mode == "bin" else "Genomic Position"),
        yaxis=dict(title="", showticklabels=False),
        yaxis2=dict(title="prob", overlaying="y", side="right", rangemode="tozero"),
        height=520,
        margin=dict(l=70, r=80, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uirevision=f"{cancer_type}-{chrom}-{x_mode}",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.88)"),
    )
    fig.add_annotation(
        x=1, y=-0.22, xref="paper", yref="paper", showarrow=False,
        text=f"{cancer_type} / {chrom}  |  file: {os.path.relpath(main_file, BASE_DIR)}",
        font=dict(size=11, color="rgba(255,255,255,0.65)"), xanchor="right"
    )
    return fig, centers_state


# === Click handling ===
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
        return no_update, no_update, "Please select a cancer type and chromosome first", True, True

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

    if "add" in (add_mode or []):
        prov = dict(type=cancer_type, chrom=chrom, label=add_to_label, value=x_clicked)
        return None, prov, f"Preview: {add_to_label} @ {x_clicked:.2f}", False, True

    hit = nearest_center(label_dict, x_clicked)
    if not hit:
        return None, None, "No centers on this chromosome yet", True, True

    lab, idx, val, dist = hit
    tol = float(SNAP_TOL_BIN) if x_mode == "bin" else float(SNAP_TOL_BP)
    if dist <= tol:
        sel = dict(type=cancer_type, chrom=chrom, label=lab, index=idx, value=val)
        return sel, None, f"Selected {lab} @ {val:.2f}", True, False

    return None, None, f"No center within tolerance (nearest {dist:.2f} > {tol:.0f})", True, True


# === Confirm Add ===
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
        return no_update, no_update, "No preview to confirm"

    t, c, lab, val = prov["type"], prov["chrom"], prov["label"], float(prov["value"])
    centers_state = ensure_tc_dict(centers_state or {})
    centers_state["TYPE_CENTERS"].setdefault(t, {}).setdefault(c, {}).setdefault(lab, [])
    arr = centers_state["TYPE_CENTERS"][t][c][lab]
    arr.append(val)
    arr.sort()
    return centers_state, None, f"Added {lab} @ {val:.2f} (not saved yet)"


# === Delete Selected ===
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
        return no_update, no_update, "No selected center"

    t, c, lab, idx, val = sel["type"], sel["chrom"], sel["label"], int(sel["index"]), float(sel["value"])
    arr = (centers_state.get("TYPE_CENTERS", {})
           .get(t, {})
           .get(c, {})
           .get(lab, []))
    if not arr:
        return no_update, None, "Could not find center"
    try:
        arr.pop(idx)
        return centers_state, None, f"Deleted {lab} @ {val:.2f} (not saved yet)"
    except Exception:
        return no_update, None, "Delete failed: invalid index"


# === Reset selection/provisional ===
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
    return None, None, "Selection and preview reset"


# === Save to YAML ===
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
    out_path = centers_state.get("_centers_yaml_path", "")

    if not out_path:
        return "Save failed: no YAML file set (Create new file or Load file first)"

    try:
        data_to_dump = {"TYPE_CENTERS": centers_state.get("TYPE_CENTERS", {})}
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_to_dump, f, allow_unicode=True, sort_keys=True)
        return f"Saved to {out_path}"
    except Exception as e:
        return f"Save failed: {e}"


if __name__ == "__main__":
    app.run_server(debug=True)