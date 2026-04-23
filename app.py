import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

st.set_page_config(page_title="QUANT ANALYZER | METODO F", layout="wide")

# =========================
# ⚙️ CONFIGURACIÓN GENERAL
# =========================
st.sidebar.markdown("## ⚙️ Configuración General")

INITIAL_CAPITAL = st.sidebar.number_input(
    "💰 Capital inicial de la cuenta ($)",
    min_value=100.0,
    value=10000.0,
    step=500.0,
    help="Capital inicial usado para Equity Curve, DrawDown y Monte Carlo (StrategyQuant style)"
)

st.markdown("---")

# =========================
# ESTILO
# =========================
st.markdown("""
<style>
.main { background-color: #0e1117; color: #ffffff; }
.stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 4px; }
div[data-testid="stMetricValue"] { color: #58a6ff; font-family: 'Courier New', monospace; }
</style>
""", unsafe_allow_html=True)

# =========================
# ✅ CSV AUTO-REGIÓN (solo lectura)
# =========================
def _detect_csv_dialect_from_sample(sample_text: str):
    """
    Detecta separador y decimal para casos típicos:
      - AR/ES: sep=';' decimal=','
      - US/CO: sep=',' decimal='.'
    """
    candidate_seps = [';', ',', '\t', '|']
    lines = [ln for ln in sample_text.splitlines() if ln.strip()][:50]

    if not lines:
        return (";", ",")

    sep_scores = {}
    for sep in candidate_seps:
        counts = [ln.count(sep) for ln in lines[:30]]
        counts_sorted = sorted(counts)
        median = counts_sorted[len(counts_sorted) // 2] if counts_sorted else 0
        sep_scores[sep] = median

    sep = max(sep_scores, key=sep_scores.get)
    if sep_scores.get(sep, 0) == 0:
        sep = ";"

    joined = "\n".join(lines)
    comma_dec = len(re.findall(r"\d+,\d{1,6}\b", joined))
    dot_dec = len(re.findall(r"\d+\.\d{1,6}\b", joined))

    if comma_dec > dot_dec:
        decimal = ","
    elif dot_dec > comma_dec:
        decimal = "."
    else:
        decimal = "," if sep == ";" else "."

    return (sep, decimal)

def read_csv_any_locale(file_obj):
    """
    Lee un CSV detectando separador/decimal + encoding (utf-8-sig / latin1),
    sin modificar el resto del pipeline.
    Funciona con st.file_uploader (file-like).
    """
    for enc in ("utf-8-sig", "latin1"):
        try:
            file_obj.seek(0)
            head_bytes = file_obj.read(200_000)
            file_obj.seek(0)

            sample_text = head_bytes.decode(enc, errors="replace")
            sep, dec = _detect_csv_dialect_from_sample(sample_text)

            df = pd.read_csv(
                file_obj,
                sep=sep,
                encoding=enc,
                decimal=dec,
                engine="python"
            )
            return df
        except Exception:
            continue

    file_obj.seek(0)
    return pd.read_csv(file_obj, engine="python")

# =========================
# HELPERS
# =========================
def clean_numeric_value(val):
    if isinstance(val, str):
        val = (
            val.replace('$', '')
               .replace(' ', '')
               .replace('\xa0', '')
        )
        if ',' in val and '.' in val:
            val = val.replace(',', '')
        elif ',' in val:
            val = val.replace(',', '.')
    return pd.to_numeric(val, errors="coerce")

def find_column(cols, keywords):
    for c in cols:
        if any(k in c.lower() for k in keywords):
            return c
    return None

def parse_datetime_mixed(series):
    s = series.astype(str).str.strip()
    s = s.str.replace(".", "/", regex=False)

    dt = pd.to_datetime(
        s,
        errors="coerce",
        dayfirst=True
    )

    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(
            s[mask],
            errors="coerce",
            dayfirst=False
        )

    return dt

def safe_metric_number(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return value
    except Exception:
        return default

# =========================
# PARSER
# =========================
def procesar_archivo(file):
    try:
        if file.name.lower().endswith(".csv"):
            df = read_csv_any_locale(file)
        else:
            df = pd.read_excel(file)

        df.columns = [str(c).strip() for c in df.columns]
        cols = list(df.columns)

        col_entry_time = find_column(cols, ["tiempo de entrada", "entry time"])
        col_exit_time = find_column(cols, ["tiempo de salida", "exit time"])
        col_pnl = (
            find_column(cols, ["con ganancia neto", "cum. net profit"]) or
            find_column(cols, ["ganancia neto", "profit"]) or
            find_column(cols, ["ganancias"])
        )
        col_strategy = find_column(cols, ["estrategia", "strategy"])

        if not col_entry_time or not col_exit_time or not col_pnl:
            return None, "❌ No se detectó Entry Time, Exit Time o PnL"

        df[col_pnl] = df[col_pnl].apply(clean_numeric_value)
        df[col_entry_time] = parse_datetime_mixed(df[col_entry_time])
        df[col_exit_time] = parse_datetime_mixed(df[col_exit_time])

        df = df.dropna(subset=[col_entry_time, col_exit_time, col_pnl]).copy()
        if df.empty:
            return None, "⚠️ El archivo no contiene filas válidas luego del parseo."

        df = df.sort_values(col_exit_time)

        pnl_acumulado = df[col_pnl].astype(float)
        pnl_trade = pnl_acumulado.diff()
        if not pnl_trade.empty:
            pnl_trade.iloc[0] = pnl_acumulado.iloc[0]

        strategy_name = df[col_strategy] if col_strategy else file.name

        out = pd.DataFrame({
            "EntryTime": df[col_entry_time],
            "Timestamp": df[col_exit_time],
            "PnL": pnl_trade,
            "Strategy": strategy_name
        })

        out = out.dropna(subset=["EntryTime", "Timestamp", "PnL"])
        if out.empty:
            return None, "⚠️ No quedaron trades válidos para procesar."

        return out.sort_values("Timestamp"), None

    except Exception as e:
        return None, f"❌ Error procesando archivo: {e}"

# =========================
# INGESTA
# =========================
files = st.sidebar.file_uploader(
    "Subir reportes NinjaTrader (Trade List)",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

if not files:
    st.stop()

dfs = []

for f in files:
    df, error = procesar_archivo(f)
    if error:
        st.warning(f"{f.name}: {error}")
        continue

    st.success(f"{f.name}: {len(df)} trades cargados")
    dfs.append(df)

if not dfs:
    st.error("❌ No se pudo cargar ningún archivo válido.")
    st.stop()

# =========================
# FILTRO POR DÍA
# =========================
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

st.sidebar.markdown("### 📅 Días activos")

active_days = st.sidebar.multiselect(
    "Seleccionar días",
    options=DAYS,
    default=DAYS
)

all_trades = pd.concat(dfs, ignore_index=True)
all_trades["DOW"] = all_trades["Timestamp"].dt.day_name()

filtered_trades = all_trades[
    all_trades["DOW"].isin(active_days)
].copy()

if filtered_trades.empty:
    st.warning("⚠️ No hay trades para los días seleccionados.")
    st.stop()

# =========================
# SERIES DIARIAS
# =========================
series = {}

for name, df in filtered_trades.groupby("Strategy"):
    daily = (
        df.set_index("Timestamp")["PnL"]
          .resample("D")
          .sum()
          .fillna(0)
    )
    series[name] = daily

df_master = pd.DataFrame(series).fillna(0)

if df_master.empty:
    st.warning("⚠️ No se pudo construir la serie diaria del portafolio.")
    st.stop()

df_master["PORTFOLIO"] = df_master.sum(axis=1)

# =========================
# TABS
# =========================
tab_main, tab_corr = st.tabs([
    "📊 Análisis Completo",
    "🔗 Correlación de Portafolios"
])

with tab_main:

    # =========================
    # EQUITY
    # =========================
    st.subheader("📈 Equity Curve")

    fig = go.Figure()

    equity_curve = INITIAL_CAPITAL + df_master["PORTFOLIO"].cumsum()
    fig.add_trace(go.Scatter(
        x=df_master.index,
        y=equity_curve,
        name="PORTFOLIO TOTAL",
        line=dict(width=3, color="#58a6ff")
    ))

    for c in df_master.columns:
        if c == "PORTFOLIO":
            continue
        fig.add_trace(go.Scatter(
            x=df_master.index,
            y=INITIAL_CAPITAL + df_master[c].cumsum(),
            name=f"{c} (estrategia)",
            line=dict(dash="dot")
        ))

    fig.update_layout(
        height=500,
        xaxis_title="Fecha",
        yaxis_title="Equity ($)",
        legend_title="Curvas",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # DRAWDOWN
    # =========================
    st.subheader("📉 Drawdown del Portafolio")

    equity = df_master["PORTFOLIO"].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak

    st.metric("Max Drawdown ($)", f"{safe_metric_number(drawdown.min()):,.2f}")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        fill="tozeroy"
    ))
    fig_dd.update_layout(height=300)
    st.plotly_chart(fig_dd, use_container_width=True)

    # =========================
    # WINRATE
    # =========================
    st.subheader("🎯 Winrate")

    winrate = (filtered_trades["PnL"] > 0).mean() * 100 if len(filtered_trades) > 0 else 0
    st.metric("Winrate total (%)", f"{safe_metric_number(winrate):.2f}%")

    # =========================
    # RESULTADOS POR DÍA
    # =========================
    st.subheader("📅 Resultados por día de la semana")

    profit_dow = (
        filtered_trades
        .groupby("DOW")["PnL"]
        .sum()
        .reindex(DAYS)
    )

    winrate_dow = (
        filtered_trades
        .assign(Win=filtered_trades["PnL"] > 0)
        .groupby("DOW")["Win"]
        .mean()
        .reindex(DAYS) * 100
    )

    dow_stats = pd.DataFrame({
        "Profit": profit_dow,
        "Winrate": winrate_dow
    }).fillna(0)

    fig_dow = go.Figure()
    fig_dow.add_trace(go.Bar(
        x=dow_stats.index,
        y=dow_stats["Profit"],
        customdata=dow_stats["Winrate"],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Profit: $%{y:,.2f}<br>"
            "Winrate: %{customdata:.2f}%"
            "<extra></extra>"
        )
    ))

    fig_dow.update_layout(height=400)
    st.plotly_chart(fig_dow, use_container_width=True)

    # =====================================================
    # 📆 MEJORES Y PEORES DÍAS DE TRADING
    # =====================================================
    st.subheader("📆 Mejores y Peores Días de Trading")

    daily_stats = (
        filtered_trades
        .groupby(filtered_trades["Timestamp"].dt.date)
        .agg(
            Trades=("PnL", "count"),
            Profit=("PnL", "sum"),
            Winrate=("PnL", lambda x: (x > 0).mean() * 100)
        )
        .reset_index()
    )

    if not daily_stats.empty:
        first_col = daily_stats.columns[0]
        daily_stats = daily_stats.rename(columns={first_col: "Fecha"})
        daily_stats["Fecha"] = pd.to_datetime(daily_stats["Fecha"])

        top_winners = daily_stats.sort_values("Profit", ascending=False).head(5)
        top_losers = daily_stats.sort_values("Profit", ascending=True).head(5)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 🟢 Mejores 5 días")
            st.dataframe(
                top_winners[["Fecha", "Trades", "Profit", "Winrate"]].rename(columns={
                    "Trades": "Cantidad de Trades",
                    "Profit": "PnL ($)",
                    "Winrate": "Winrate (%)"
                }),
                use_container_width=True
            )

        with c2:
            st.markdown("#### 🔴 Peores 5 días")
            st.dataframe(
                top_losers[["Fecha", "Trades", "Profit", "Winrate"]].rename(columns={
                    "Trades": "Cantidad de Trades",
                    "Profit": "PnL ($)",
                    "Winrate": "Winrate (%)"
                }),
                use_container_width=True
            )

        # =====================================================
        # 📋 TABLA COMPLETA DIARIA
        # =====================================================
        st.subheader("📋 Detalle diario completo")

        dias_es = {
            "Monday": "Lunes",
            "Tuesday": "Martes",
            "Wednesday": "Miércoles",
            "Thursday": "Jueves",
            "Friday": "Viernes",
            "Saturday": "Sábado",
            "Sunday": "Domingo"
        }

        daily_stats["Día"] = (
            daily_stats["Fecha"]
            .dt.day_name()
            .map(dias_es)
        )

        st.dataframe(
            daily_stats
            .sort_values("Fecha")[[
                "Fecha",
                "Día",
                "Trades",
                "Profit",
                "Winrate"
            ]]
            .rename(columns={
                "Trades": "Cantidad de Trades",
                "Profit": "PnL ($)",
                "Winrate": "Winrate (%)"
            }),
            use_container_width=True
        )
    else:
        st.info("No hay datos diarios para mostrar.")

    # =========================
    # RESULTADOS MENSUALES
    # =========================
    st.subheader("📆 Profit mensual")
    st.subheader("📅 Performance mensual – Winrate, Ganancia y Score")

    monthly_trades = filtered_trades.copy()
    monthly_trades["Month"] = monthly_trades["Timestamp"].dt.month
    monthly_trades["MonthName"] = monthly_trades["Timestamp"].dt.month_name()

    monthly_stats = (
        monthly_trades
        .groupby(["Month", "MonthName"])
        .agg(
            Trades=("PnL", "count"),
            Profit=("PnL", "sum"),
            Winrate=("PnL", lambda x: (x > 0).mean() * 100)
        )
        .reset_index()
        .sort_values("Month")
    )

    if not monthly_stats.empty:
        pnl_n = monthly_stats["Profit"] - monthly_stats["Profit"].min()
        pnl_n = pnl_n / pnl_n.max() if pnl_n.max() != 0 else pd.Series(0, index=monthly_stats.index)

        trades_n = monthly_stats["Trades"] / monthly_stats["Trades"].max() if monthly_stats["Trades"].max() != 0 else pd.Series(0, index=monthly_stats.index)
        winrate_n = monthly_stats["Winrate"] / 100

        monthly_stats["Score"] = (
            pnl_n * 0.5 +
            winrate_n * 0.3 +
            trades_n * 0.2
        )

        fig_month_perf = go.Figure()
        fig_month_perf.add_trace(go.Bar(
            x=monthly_stats["MonthName"],
            y=monthly_stats["Profit"],
            customdata=np.stack(
                [monthly_stats["Winrate"], monthly_stats["Score"]],
                axis=-1
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Profit: $%{y:,.2f}<br>"
                "Winrate: %{customdata[0]:.2f}%<br>"
                "Score: %{customdata[1]:.2f}"
                "<extra></extra>"
            )
        ))

        fig_month_perf.update_layout(
            height=400,
            xaxis_title="Mes",
            yaxis_title="Ganancia neta ($)"
        )

        st.plotly_chart(fig_month_perf, use_container_width=True)

        st.subheader("📋 Detalle mensual")

        st.dataframe(
            monthly_stats[[
                "MonthName",
                "Trades",
                "Winrate",
                "Profit",
                "Score"
            ]].rename(columns={
                "MonthName": "Mes",
                "Trades": "Cantidad de Trades",
                "Winrate": "Winrate (%)",
                "Profit": "Ganancia Neta ($)"
            }),
            use_container_width=True
        )

        best_month = monthly_stats.loc[monthly_stats["Score"].idxmax()]
        worst_month = monthly_stats.loc[monthly_stats["Score"].idxmin()]

        c1, c2 = st.columns(2)
        c1.success(
            f"🥇 Mejor mes: {best_month['MonthName']} | "
            f"Score: {best_month['Score']:.2f} | "
            f"PnL: ${best_month['Profit']:,.2f}"
        )

        c2.error(
            f"⚠️ Peor mes: {worst_month['MonthName']} | "
            f"Score: {worst_month['Score']:.2f} | "
            f"PnL: ${worst_month['Profit']:,.2f}"
        )
    else:
        st.info("No hay datos mensuales para mostrar.")

    monthly = (
        df_master["PORTFOLIO"]
        .resample("ME")
        .sum()
    )

    fig_m = go.Figure()
    fig_m.add_trace(go.Bar(
        x=monthly.index,
        y=monthly.values
    ))
    fig_m.update_layout(height=400)
    st.plotly_chart(fig_m, use_container_width=True)

    st.dataframe(monthly.to_frame("PnL Mensual"), use_container_width=True)

    # =====================================================
    # 🔬 ANALISIS AVANZADO DE ROBUSTEZ
    # =====================================================
    st.subheader("🧠 Análisis Avanzado de Robustez")

    pnl_trades = filtered_trades["PnL"].dropna()
    daily_returns = df_master["PORTFOLIO"]

    wins = pnl_trades[pnl_trades > 0]
    losses = pnl_trades[pnl_trades < 0]

    avg_win = wins.mean() if not wins.empty else 0
    avg_loss = abs(losses.mean()) if not losses.empty else 0

    profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty and abs(losses.sum()) != 0 else np.nan
    expectancy = pnl_trades.mean() if not pnl_trades.empty else np.nan
    risk_reward = avg_win / avg_loss if avg_loss != 0 else np.nan

    sharpe = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() not in [0, np.nan] and daily_returns.std() != 0 else np.nan
    )

    downside = daily_returns[daily_returns < 0]
    sortino = (
        daily_returns.mean() / downside.std() * np.sqrt(252)
        if len(downside) > 1 and downside.std() != 0 else np.nan
    )

    equity_adv = daily_returns.cumsum()
    peak_adv = equity_adv.cummax()
    drawdown_adv = equity_adv - peak_adv

    time_under_water = (
        (drawdown_adv < 0)
        .astype(int)
        .groupby((drawdown_adv == 0).cumsum())
        .sum()
        .max()
    )
    time_under_water = 0 if pd.isna(time_under_water) else int(time_under_water)

    losing = (pnl_trades < 0).astype(int)
    max_losing_streak = losing.groupby((losing == 0).cumsum()).sum().max()
    max_losing_streak = 0 if pd.isna(max_losing_streak) else int(max_losing_streak)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profit Factor", f"{safe_metric_number(profit_factor, np.nan):.2f}" if not pd.isna(profit_factor) else "N/A")
    c2.metric("Expectancy / Trade", f"${safe_metric_number(expectancy):,.2f}" if not pd.isna(expectancy) else "N/A")
    c3.metric("Risk / Reward", f"{safe_metric_number(risk_reward, np.nan):.2f}" if not pd.isna(risk_reward) else "N/A")
    c4.metric("Sharpe Ratio", f"{safe_metric_number(sharpe, np.nan):.2f}" if not pd.isna(sharpe) else "N/A")

    c1, c2, c3 = st.columns(3)
    c1.metric("Sortino Ratio", f"{safe_metric_number(sortino, np.nan):.2f}" if not pd.isna(sortino) else "N/A")
    c2.metric("Max Losing Streak", max_losing_streak)
    c3.metric("Time Under Water (días)", time_under_water)

    # =====================================================
    # 🎲 MONTE CARLO – StrategyQuant
    # =====================================================
    st.subheader("🎲 Monte Carlo")

    mc_runs = 500
    SKIP_PROB = 0.05

    pnl = filtered_trades["PnL"].dropna().values

    if len(pnl) > 0:
        orig_equity = INITIAL_CAPITAL + np.cumsum(pnl)
        orig_peak = np.maximum.accumulate(orig_equity)
        orig_drawdown = orig_equity - orig_peak

        orig_max_dd = orig_drawdown.min() if len(orig_drawdown) > 0 else np.nan
        orig_net = orig_equity[-1] - INITIAL_CAPITAL if len(orig_equity) > 0 else np.nan
        orig_ret_dd = (
            orig_net / abs(orig_max_dd)
            if pd.notna(orig_max_dd) and orig_max_dd < 0 else np.nan
        )

        mc_equities = []
        mc_stats = []

        for _ in range(mc_runs):
            shuffled = np.random.permutation(pnl)
            mask = np.random.rand(len(shuffled)) > SKIP_PROB
            sampled = shuffled[mask]

            if len(sampled) == 0:
                continue

            equity = INITIAL_CAPITAL + np.cumsum(sampled)
            peak = np.maximum.accumulate(equity)
            drawdown = equity - peak

            max_dd = drawdown.min()
            net_profit = equity[-1] - INITIAL_CAPITAL
            ret_dd = net_profit / abs(max_dd) if max_dd < 0 else np.nan
            exp = net_profit / len(sampled)

            mc_equities.append(equity)
            mc_stats.append({
                "Net Profit": net_profit,
                "Max DD $": max_dd,
                "Ret/DD": ret_dd,
                "Exp": exp
            })

        if mc_stats:
            mc_equity_df = pd.DataFrame(mc_equities).T
            mc_stats_df = pd.DataFrame(mc_stats)

            percentiles = [0, 5, 10, 30, 50, 70, 90, 95, 100]

            sq_table = pd.DataFrame({
                col: np.percentile(mc_stats_df[col].dropna(), percentiles)
                for col in mc_stats_df.columns
            }, index=[f"{p}%" for p in percentiles])

            fig_mc = go.Figure()

            for i in range(min(40, mc_equity_df.shape[1])):
                fig_mc.add_trace(go.Scatter(
                    y=mc_equity_df.iloc[:, i],
                    line=dict(width=1),
                    opacity=0.25,
                    showlegend=False
                ))

            fig_mc.update_layout(
                height=450,
                title="Monte Carlo – Equity Runs",
                xaxis_title="Trades",
                yaxis_title="Equity"
            )

            st.plotly_chart(fig_mc, use_container_width=True)

            st.subheader("📊 Confidence Levels")
            st.dataframe(
                sq_table.style.format({
                    "Net Profit": "${:,.2f}",
                    "Max DD $": "${:,.2f}",
                    "Ret/DD": "{:.2f}",
                    "Exp": "${:,.2f}"
                })
            )

            st.subheader("🧠 Análisis de Robustez")

            p95 = sq_table.loc["95%"]

            robust = (
                pd.notna(orig_net) and orig_net != 0 and
                pd.notna(orig_ret_dd) and orig_ret_dd != 0 and
                p95["Net Profit"] > 0 and
                p95["Net Profit"] / orig_net > 0.3 and
                p95["Ret/DD"] / orig_ret_dd > 0.3
            )

            if robust:
                st.success(
                    "✅ **La estrategia es ROBUSTA**\n\n"
                    "- Mantiene Net Profit positivo incluso al 95%\n"
                    "- El Ret/DD sigue siendo saludable\n"
                    "- El comportamiento es consistente bajo perturbaciones aleatorias\n\n"
                    "La estrategia tolera cambios en el orden y omisión de trades."
                )
            else:
                st.warning(
                    "⚠️ **La estrategia NO es robusta**\n\n"
                    "- En escenarios adversos (95%) pierde rentabilidad\n"
                    "- El drawdown se deteriora significativamente\n"
                    "- El retorno ajustado por riesgo no es estable\n\n"
                    "La estrategia depende fuertemente del orden exacto de los trades."
                )
        else:
            st.info("No se pudieron generar simulaciones de Monte Carlo.")
    else:
        st.info("No hay trades suficientes para ejecutar Monte Carlo.")

    # =====================================================
    # 🔥 STRESS TEST POR AÑO
    # =====================================================
    st.subheader("🔥 Stress Test – Rendimiento por Año")

    yearly = (
        df_master["PORTFOLIO"]
        .groupby(df_master.index.year)
        .sum()
    )

    fig_year = go.Figure()
    fig_year.add_trace(go.Bar(
        x=yearly.index.astype(str),
        y=yearly.values
    ))
    fig_year.update_layout(height=300)
    st.plotly_chart(fig_year, use_container_width=True)

    st.dataframe(yearly.to_frame("PnL Anual"), use_container_width=True)

    # =====================================================
    # ⏳ TIME UNDER WATER – ANÁLISIS DETALLADO
    # =====================================================
    st.subheader("⏳ Time Under Water – Análisis Detallado")

    equity_tuw = df_master["PORTFOLIO"].cumsum()
    peak_tuw = equity_tuw.cummax()
    drawdown_tuw = equity_tuw - peak_tuw

    underwater = drawdown_tuw < 0
    groups = (underwater != underwater.shift()).cumsum()

    tuw_periods = []

    for _, data in drawdown_tuw.groupby(groups):
        if len(data) > 0 and data.iloc[0] < 0:
            start = data.index[0]
            end = data.index[-1]
            duration = (end - start).days + 1
            max_dd = data.min()

            tuw_periods.append({
                "Inicio": start,
                "Fin": end,
                "Duración (días)": duration,
                "Max Drawdown ($)": max_dd
            })

    df_tuw = pd.DataFrame(tuw_periods)

    if not df_tuw.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Cantidad de períodos TUW", len(df_tuw))
        c2.metric("Duración promedio (días)", f"{df_tuw['Duración (días)'].mean():.1f}")
        c3.metric("Duración máxima (días)", int(df_tuw["Duración (días)"].max()))
    else:
        st.info("No se detectaron períodos de Time Under Water.")

    st.subheader("📊 Duración de cada Time Under Water")

    if not df_tuw.empty:
        fig_tuw = go.Figure()
        fig_tuw.add_trace(go.Bar(
            x=list(range(1, len(df_tuw) + 1)),
            y=df_tuw["Duración (días)"],
            customdata=df_tuw["Max Drawdown ($)"],
            hovertemplate=(
                "<b>Período %{x}</b><br>"
                "Duración: %{y} días<br>"
                "Max DD: $%{customdata:,.2f}"
                "<extra></extra>"
            )
        ))

        fig_tuw.update_layout(
            height=400,
            xaxis_title="Episodio de Drawdown",
            yaxis_title="Días bajo el último máximo"
        )

        st.plotly_chart(fig_tuw, use_container_width=True)

    st.subheader("📋 Detalle completo de Time Under Water")

    if not df_tuw.empty:
        st.dataframe(df_tuw, use_container_width=True)

    # =====================================================
    # ⏰ ANALISIS POR HORARIO DE ENTRADA (30 MIN)
    # =====================================================
    time_stats = None

    st.subheader("⏰ Rendimiento por horario de EJECUCIÓN (Entry) – bloques de 30 minutos")

    entry_col = "EntryTime" if "EntryTime" in filtered_trades.columns else None

    if entry_col is None:
        st.warning("⚠️ No se encontró la columna de horario de entrada (Entry Time / Tiempo de Entrada).")
    else:
        trades_time = filtered_trades.copy()

        trades_time[entry_col] = parse_datetime_mixed(trades_time[entry_col])
        trades_time = trades_time.dropna(subset=[entry_col])

        if not trades_time.empty:
            trades_time["TimeBucket"] = (
                trades_time[entry_col]
                .dt.floor("30min")
                .dt.strftime("%H:%M")
            )

            time_stats = trades_time.groupby("TimeBucket").agg(
                Trades=("PnL", "count"),
                Profit=("PnL", "sum"),
                Winrate=("PnL", lambda x: (x > 0).mean() * 100)
            ).reset_index()

            time_stats["SortKey"] = pd.to_datetime(
                time_stats["TimeBucket"],
                format="%H:%M"
            )
            time_stats = time_stats.sort_values("SortKey")

            colors = np.where(
                time_stats["Profit"] >= 0,
                "rgba(0,200,0,0.65)",
                "rgba(200,0,0,0.65)"
            )

            fig_time = go.Figure()
            fig_time.add_trace(go.Bar(
                x=time_stats["TimeBucket"],
                y=time_stats["Profit"],
                marker_color=colors,
                customdata=np.stack(
                    [time_stats["Trades"], time_stats["Winrate"]],
                    axis=-1
                ),
                hovertemplate=(
                    "<b>Horario de entrada %{x}</b><br>"
                    "Trades: %{customdata[0]}<br>"
                    "Winrate: %{customdata[1]:.2f}%<br>"
                    "PnL: $%{y:,.2f}"
                    "<extra></extra>"
                )
            ))

            fig_time.update_layout(
                height=450,
                xaxis_title="Horario de EJECUCIÓN (30 minutos)",
                yaxis_title="Ganancia / Pérdida ($)",
                bargap=0.15
            )

            st.plotly_chart(fig_time, use_container_width=True)

            st.subheader("📋 Resumen por horario de entrada")

            st.dataframe(
                time_stats[["TimeBucket", "Trades", "Winrate", "Profit"]]
                .rename(columns={
                    "TimeBucket": "Horario Entrada",
                    "Trades": "Cantidad de Trades",
                    "Winrate": "Winrate (%)",
                    "Profit": "PnL ($)"
                }),
                use_container_width=True
            )
        else:
            st.info("No hay horarios de entrada válidos para analizar.")

# =====================================================
# 🔗 TAB CORRELACIÓN DE PORTAFOLIOS
# =====================================================
with tab_corr:

    st.subheader("🔗 Correlación entre Portafolios (Cuentas Independientes)")

    all_strategies = [c for c in df_master.columns if c != "PORTFOLIO"]

    st.markdown("### 📦 Construcción de Portafolios")

    portfolio_returns = {}

    for i in range(1, 6):
        st.markdown(f"#### 🔹 Portafolio {i}")

        selected = st.multiselect(
            f"Estrategias del Portafolio {i}",
            options=all_strategies,
            default=[],
            key=f"portfolio_{i}"
        )

        if selected:
            portfolio_returns[f"Portafolio {i}"] = (
                df_master[selected].sum(axis=1)
            )

    if len(portfolio_returns) < 2:
        st.warning("⚠️ Debés construir al menos 2 portafolios para medir correlación.")
        st.stop()

    df_portfolios = pd.DataFrame(portfolio_returns)

    st.subheader("📈 Equity Curve por Portafolio")

    fig_eq = go.Figure()
    for c in df_portfolios.columns:
        fig_eq.add_trace(go.Scatter(
            x=df_portfolios.index,
            y=df_portfolios[c].cumsum(),
            name=c
        ))

    fig_eq.update_layout(height=450)
    st.plotly_chart(fig_eq, use_container_width=True, key="equity_portfolios")

    # =====================================================
    # 📉 CORRELACIÓN POR DRAWDOWN
    # =====================================================
    st.subheader("📉 Correlación por DrawDown entre Portafolios")

    equity_df = df_portfolios.cumsum()

    def compute_drawdown(equity_series):
        peak = equity_series.cummax()
        peak_safe = peak.replace(0, np.nan)
        dd = (equity_series - peak_safe) / peak_safe
        return dd.fillna(0)

    dd_df = equity_df.apply(compute_drawdown)
    dd_corr = dd_df.corr()

    fig_dd_corr = go.Figure(
        data=go.Heatmap(
            z=dd_corr.values,
            x=dd_corr.columns,
            y=dd_corr.index,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Correlación DD")
        )
    )

    fig_dd_corr.update_layout(height=450)

    st.plotly_chart(
        fig_dd_corr,
        use_container_width=True,
        key="portfolio_dd_corr_heatmap"
    )

    st.subheader("📋 Matriz de correlación por DrawDown")
    st.dataframe(dd_corr.round(2), use_container_width=True)

    upper_triangle = dd_corr.values[np.triu_indices_from(dd_corr.values, k=1)]
    dd_avg_corr = upper_triangle.mean() if len(upper_triangle) > 0 else 0

    st.metric(
        "Correlación promedio por DrawDown",
        f"{dd_avg_corr:.2f}",
        help="Mide si los portafolios entran en drawdown al mismo tiempo. Cuanto más bajo o negativo, mejor."
    )

    dd_pairs = []

    for i in range(len(dd_corr.columns)):
        for j in range(i + 1, len(dd_corr.columns)):
            a = dd_corr.columns[i]
            b = dd_corr.columns[j]
            val = dd_corr.iloc[i, j]

            if val < 0:
                dd_pairs.append({
                    "Portafolio A": a,
                    "Portafolio B": b,
                    "Correlación DD": val
                })

    if dd_pairs:
        st.subheader("🟢 Portafolios con DrawDown descorrelado (IDEAL)")
        st.dataframe(
            pd.DataFrame(dd_pairs)
            .sort_values("Correlación DD")
            .round(2),
            use_container_width=True
        )
    else:
        st.info("ℹ️ No se detectaron correlaciones negativas por DrawDown.")

    # =====================================================
    # 🧭 RECOMENDACIONES DE DIVERSIFICACIÓN
    # =====================================================
    st.subheader("🧭 Recomendación de Diversificación por DrawDown")

    st.markdown("""
    **📊 Benchmark profesional (Correlación por DrawDown):**

    - ❌ **> 0.50** → Riesgo conjunto alto (mala diversificación)  
    - ⚠️ **0.30 – 0.50** → Diversificación débil  
    - ✅ **0.15 – 0.30** → Buena diversificación  
    - 🟢 **0.00 – 0.15** → Muy buena  
    - 🟢🟢 **< 0.00** → Excelente (drawdowns se compensan)
    """)

    if dd_avg_corr > 0.50:
        st.error(
            f"❌ **Diversificación deficiente**\n\n"
            f"Correlación DD promedio = **{dd_avg_corr:.2f}**\n\n"
            "Los portafolios entran en drawdown al mismo tiempo. "
            "No es recomendable escalar capital."
        )

    elif dd_avg_corr > 0.30:
        st.warning(
            f"⚠️ **Diversificación débil**\n\n"
            f"Correlación DD promedio = **{dd_avg_corr:.2f}**\n\n"
            "Existe solapamiento de drawdowns. "
            "Se recomienda agregar estrategias no correlacionadas "
            "o ajustar pesos."
        )

    elif dd_avg_corr > 0.15:
        st.success(
            f"✅ **Buena diversificación**\n\n"
            f"Correlación DD promedio = **{dd_avg_corr:.2f}**\n\n"
            "El portafolio está razonablemente diversificado. "
            "Puede escalarse con control de riesgo."
        )

    elif dd_avg_corr >= 0.0:
        st.success(
            f"🟢 **Muy buena diversificación**\n\n"
            f"Correlación DD promedio = **{dd_avg_corr:.2f}**\n\n"
            "Los drawdowns se solapan poco. "
            "Estructura sólida para crecimiento estable."
        )

    else:
        st.success(
            f"🟢🟢 **Diversificación excelente**\n\n"
            f"Correlación DD promedio = **{dd_avg_corr:.2f}**\n\n"
            "Los drawdowns tienden a compensarse entre portafolios. "
            "Configuración ideal para minimizar drawdown agregado."
        )

    st.info(
        "🎯 **Objetivo recomendado:**\n\n"
        "Mantener la correlación promedio por DrawDown "
        "**≤ 0.20** para una diversificación robusta y escalable."
    )
