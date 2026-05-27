import streamlit as st
import pandas as pd
import io
import plotly.graph_objects as go
from scipy import stats
import numpy as np
from statsmodels.stats.multitest import multipletests

st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("Data Explorer")

# ─── Initialize Session State ────────────────────────────────────────────────
if "datasets" not in st.session_state:
    st.session_state.datasets = {}

# ─── Helper Functions ────────────────────────────────────────────────
@st.cache_data
def load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    try:
        return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(file_bytes), encoding="cp949")
    except Exception as e:
        st.error(f"Error loading file {filename}: {e}")
        return None

@st.cache_data
def compute_correlation(df_values: np.ndarray, feature_names: list, method: str):
    n_features = len(feature_names)

    corr_vals = np.zeros((n_features, n_features))
    p_vals = np.ones((n_features, n_features))

    df_subset = pd.DataFrame(df_values, columns=feature_names)

    for i, col_i in enumerate(feature_names):
        for j, col_j in enumerate(feature_names):

            if method == "pearson":    
                r, p = stats.pearsonr(df_subset[col_i], df_subset[col_j])
            elif method == "spearman":
                r, p = stats.spearmanr(df_subset[col_i], df_subset[col_j])
            elif method == "kendall":
                r, p = stats.kendalltau(df_subset[col_i], df_subset[col_j])
            else:
                raise ValueError(f"Invalid correlation method: {method}")

            corr_vals[i, j] = r
            p_vals[i, j] = p

    return corr_vals, p_vals
    
# ─── Sidebar: Dataset Catalog ────────────────────────────────────────────────
with st.sidebar:
    st.header("Files")

    # Load Sample Data
    if st.button("Load Sample Data"):
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df_sample = iris.frame
        st.session_state.datasets["sample_iris.csv"] = df_sample

    uploaded_files = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    for f in uploaded_files:
        if f.name not in st.session_state.datasets:
            try:
                st.session_state.datasets[f.name] = load_file(f.getvalue(), f.name)
            except Exception as e:
                st.error(f"Error loading file {f.name}: {e}")

    if not st.session_state.datasets:
        st.stop()

    selected_name = st.selectbox("Dataset", list(st.session_state.datasets.keys()))

    # Remove selected dataset from session state
    if st.button("Remove", use_container_width=True):
        del st.session_state.datasets[selected_name]
        st.rerun()

# ─── Prepare Dataset ────────────────────────────────────────────────
df = st.session_state.datasets[selected_name]
numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

# ─── Data Preview ────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Mising Values", df.isna().sum().sum())

# ─── Tabs ────────────────────────────────────────────────
tab_data, tab_chart, tab_config, tab_corr = st.tabs(["Data", "Chart", "Config", "Correlation"])

# ─── Data Tab ────────────────────────────────────────────────
with tab_data:
    st.dataframe(df, use_container_width=True)

    if numeric_columns:
        with st.expander("Numeric Columns"):
            st.dataframe(df[numeric_columns].describe().T, use_container_width=True)
# ─── Chart Tab ────────────────────────────────────────────────
with tab_chart:
    if not numeric_columns:
        st.warning("No numeric columns")
    else:
        selected = st.multiselect("Select columns for chart", numeric_columns, default=numeric_columns[:3])

        if selected:
            cleaned_subset = df[selected]
            fig = go.Figure()

            for col in selected:
                fig.add_trace(
                    go.Scattergl(
                        x=cleaned_subset.index,
                        y=cleaned_subset[col],
                        name=col, 
                        mode="lines", 
                        connectgaps=False
                    )
                )
            fig.update_layout(
                    height=500,
                    hovermode="x unified",
                    xaxis=dict(rangeslider=dict(visible=True)),
                    legend=dict(orientation="h", y=1.08),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
            st.plotly_chart(fig, use_container_width=True)
# ─── Config Tab ────────────────────────────────────────────────
with tab_config:
    st.subheader("Step 1. Data Configuration")

    row_meaning = st.radio(
        "Select the meaning of rows",
        ["Samples", "Features"],
        index=0,
        horizontal=True
    )

    st.subheader("Step 2. ID Column")
    non_numeric_columns = df.select_dtypes(exclude=["number"]).columns.tolist()
    if not non_numeric_columns:
        st.caption("No non-numeric columns found. Skipping.")
        id_col = None
    else:
        id_col = st.selectbox(
            "Select the ID column (gene name, protein ID, sample ID, etc.)",
            options=[None] + non_numeric_columns,
            format_func=lambda x: "None" if x is None else x
        )
    
    st.subheader("Step 3. Label Column")

    # Auto-detect label column if not selected
    label_candidates = [c for c in df.columns if c != id_col]

    label_col = st.selectbox(
        "Select the label column (group, condition, diagnosis, etc.)",
        options=[None] + label_candidates,
        format_func=lambda x: "None" if x is None else x
    )

    if label_col:
        unique_labels = df[label_col].unique()
        st.caption(f"Unique Labels: {unique_labels}")
    
    # ── Step 4: Summary + Build Analysis DataFrame ─────────────
    st.subheader("Step 4. Confirm & Build Analysis DataFrame")
    st.table({
        "Setting": ["Row meaning", "ID Column", "Label Column"],
        "Value": [
            row_meaning,
            id_col if id_col else "None",
            label_col if label_col else "None"
        ]
    })

    if st.button("Build Analysis DataFrame", type="primary"):
        work = df.copy()

        # Set ID column as Index
        if id_col:
            work = work.set_index(id_col)
        
        # Leave only numeric columns
        work = work.select_dtypes(include=["number"])

        # Transpose if rows are features
        if row_meaning == "Features":
            work = work.T
        
        # Separate labels
        if label_col and label_col in work.columns:
            labels = work[label_col]
            work = work.drop(columns=[label_col])
        else:
            labels = None
        
        
        st.session_state["analysis_df"] = work
        st.session_state["analysis_labels"] = labels
        st.success(f"Analysis DataFrame built successfully with {work.shape[0]} samples and {work.shape[1]} features")

# ─── Correlation Tab ────────────────────────────────────────────────
with tab_corr:
    if "analysis_df" not in st.session_state:
        st.info("No analysis DataFrame built. Please build one in the Config tab first.")
        st.stop()
    
    adf = st.session_state["analysis_df"]

    # Make user select columns for correlation
    all_features = adf.columns.tolist()

    selected_features = st.multiselect(
        "Select features for correlation analysis",
        all_features,
        default=all_features[:20]
    )

    # Select stat method
    corr_method = st.radio(
        "Correlation method",
        ["pearson", "spearman", "kendall"],
        index=0,
        horizontal=True
    )

    if len(selected_features) < 2:
        st.warning("Select at least 2 features for correlation analysis")
        st.stop()
    
    # Calculate correlation matrix
    numeric_features = adf[selected_features].select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_features) < len(selected_features):
        st.caption(f"Non-numeric features excluded: {set(selected_features) - set(numeric_features)}")
    
    missing_count = adf[numeric_features].isna().sum().sum()

    if missing_count > 0:
        st.caption(f"Missing values: {missing_count}")
        drop_na = st.radio(
            "How to handle missing values?",
            ["Drop rows with any NaN (listwise)", "Keep (pairwise, scipy will return NaN for affected pairs)"],
            horizontal=True
        )
        cleaned_subset = adf[numeric_features].dropna() if "Drop" in drop_na else adf[numeric_features].dropna(axis=1)
    else:
        st.caption("No missing values detected.")
        cleaned_subset = adf[numeric_features]

    corr_vals, p_vals = compute_correlation(cleaned_subset.values, numeric_features, corr_method)
    corr_matrix = pd.DataFrame(corr_vals, index=numeric_features, columns=numeric_features)
    pval_matrix = pd.DataFrame(p_vals, index=numeric_features, columns=numeric_features)


    # Heatmap
    map = go.Figure( go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        colorbar=dict(title="r")
    ))

    map.update_layout(
        height=600,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(map, use_container_width=False)

    # Value Table
    with st.expander("Correlation Matrix(Numeric)"):
        st.dataframe(
            corr_matrix.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.2f}"),
            use_container_width=False
        )
        
        st.caption("P-values")
        st.dataframe(
            pval_matrix.style.background_gradient(cmap="Reds_r", vmin=0, vmax=0.05).format("{:.3f}"),
            use_container_width=False
        )

        # FDR (False Discovery Rate) 
        n_features = len(numeric_features)
        mask = np.triu(np.ones((n_features, n_features), dtype=bool), k=1)
        raw_pvals = p_vals[mask]

        _, fdr_corrected, _, _ = multipletests(raw_pvals, method="fdr_bh")

        fdr_matrix = np.zeros((n_features, n_features))
        fdr_matrix[mask] = fdr_corrected
        fdr_matrix = fdr_matrix + fdr_matrix.T
        np.fill_diagonal(fdr_matrix, 0.0)

        fdr_df = pd.DataFrame(fdr_matrix, index=numeric_features, columns=numeric_features)

        st.caption("FDR-corrected P-values (Benjamini-Hochberg)")
        st.dataframe(
            fdr_df.style.background_gradient(cmap="Reds_r", vmin=0, vmax=0.05).format("{:.3f}"),
            use_container_width=False
        )
