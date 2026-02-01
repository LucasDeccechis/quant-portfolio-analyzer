import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="QUANT ANALYZER | METODO F", layout="wide")

# =========================
# 🖼️ HEADER / LOGO
# =========================
st.image(
    "assets\header.png",  # 👈 ruta de la imagen
    use_container_width=True
)

st.markdown("---")

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
        dayfirst=True,
        infer_datetime_format=True
    )

    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(
            s[mask],
            errors="coerce",
            dayfirst=False,
            infer_datetime_format=True
        )
    return dt

# =========================
# PARSER
# =========================
def procesar_archivo(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file, sep=";", encoding="latin1")
    else:
        df = pd.read_excel(file)

    df.columns = [c.strip() for c in df.columns]
    cols = list(df.columns)

    # Detectar columnas por sinónimos español / inglés
    col_entry_time = find_column(cols, ["tiempo de entrada", "entry time"])
    col_exit_time  = find_column(cols, ["tiempo de salida", "exit time"])
    # PRIORIDAD: "Con Ganancia Neto" / "Cum. net profit" > otras
    col_pnl = (
        find_column(cols, ["con ganancia neto", "cum. net profit"]) or
        find_column(cols, ["ganancia neto", "profit"]) or
        find_column(cols, ["ganancias"])
    )
    col_strategy = find_column(cols, ["estrategia", "strategy"])

    if not col_entry_time or not col_exit_time or not col_pnl:
        return None, "❌ No se detectó Entry Time, Exit Time o PnL"

    # Limpiar valores numéricos
    df[col_pnl] = df[col_pnl].apply(clean_numeric_value)
    # Parsear fechas
    df[col_entry_time] = parse_datetime_mixed(df[col_entry_time])
    df[col_exit_time]  = parse_datetime_mixed(df[col_exit_time])

    df = df.dropna(subset=[col_entry_time, col_exit_time, col_pnl])
    df = df.sort_values(col_exit_time)

    # Calcular PnL por trade
    pnl_acumulado = df[col_pnl].astype(float)
    pnl_trade = pnl_acumulado.diff()
    pnl_trade.iloc[0] = pnl_acumulado.iloc[0]

    # Usar columna de estrategia si existe, sino usar nombre del archivo
    strategy_name = df[col_strategy] if col_strategy else file.name

    out = pd.DataFrame({
        "EntryTime": df[col_entry_time],
        "Timestamp": df[col_exit_time],
        "PnL": pnl_trade,  # <-- Aquí usamos exclusivamente "Con Ganancia Neto / Cum. net profit"
        "Strategy": strategy_name
    })

    return out.sort_values("Timestamp"), None


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

all_trades = pd.concat(dfs)
all_trades["DOW"] = all_trades["Timestamp"].dt.day_name()

filtered_trades = all_trades[
    all_trades["DOW"].isin(active_days)
].copy()

# =========================
# SERIES DIARIAS
# =========================
series = {}

for name, df in filtered_trades.groupby("Strategy"):
    # Tomamos SOLO la columna PnL que representa "Con Ganancia Neto / Cum. net profit"
    daily = (
        df.set_index("Timestamp")["PnL"]
          .resample("D")
          .sum()
          .fillna(0)
    )
    series[name] = daily

df_master = pd.DataFrame(series).fillna(0)
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

    # Equity del portfolio total (PORTFOLIO) con capital inicial
    equity_curve = INITIAL_CAPITAL + df_master["PORTFOLIO"].cumsum()
    fig.add_trace(go.Scatter(
        x=df_master.index,
        y=equity_curve,
        name="PORTFOLIO TOTAL",
        line=dict(width=3, color="#58a6ff")
    ))

    # Equity individual de cada estrategia
    for c in df_master.columns:
        if c == "PORTFOLIO":
            continue
        fig.add_trace(go.Scatter(
            x=df_master.index,
            y=INITIAL_CAPITAL + df_master[c].cumsum(),  # cada estrategia acumulada
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

    st.metric("Max Drawdown ($)", f"{drawdown.min():,.2f}")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        fill="tozeroy"
    ))
    fig_dd.update_layout(height=300)
    st.plotly_chart(fig_dd, width="stretch")

    # =========================
    # WINRATE
    # =========================
    st.subheader("🎯 Winrate")

    winrate = (filtered_trades["PnL"] > 0).mean() * 100
    st.metric("Winrate total (%)", f"{winrate:.2f}%")

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
    st.plotly_chart(fig_dow, width="stretch")

    # =====================================================
    # 📆 MEJORES Y PEORES DÍAS DE TRADING
    # =====================================================

    st.subheader("📆 Mejores y Peores Días de Trading")

    # -------------------------
    # Agregación diaria
    # -------------------------
    daily_stats = (
        filtered_trades
        .groupby(filtered_trades["Timestamp"].dt.date)
        .agg(
            Trades=("PnL", "count"),
            Profit=("PnL", "sum"),
            Winrate=("PnL", lambda x: (x > 0).mean() * 100)
        )
        .reset_index()
        .rename(columns={"Timestamp": "Fecha"})
    )

    daily_stats["Fecha"] = pd.to_datetime(daily_stats["Timestamp"] if "Timestamp" in daily_stats else daily_stats.iloc[:,0])

    # -------------------------
    # Top Ganadores / Perdedores
    # -------------------------
    top_winners = daily_stats.sort_values("Profit", ascending=False).head(5)
    top_losers  = daily_stats.sort_values("Profit", ascending=True).head(5)

    c1, c2 = st.columns(2)



    # =====================================================
    # 📋 TABLA COMPLETA DIARIA
    # =====================================================

    st.subheader("📋 Detalle diario completo")

    # Asegurar datetime
    daily_stats["Fecha"] = pd.to_datetime(daily_stats["Fecha"])

    # Día de la semana en español
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

    # Mostrar tabla final
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

    
    # =========================
    # RESULTADOS MENSUALES
    # =========================
    st.subheader("📆 Profit mensual")

    # =====================================================
    # 📅 PERFORMANCE POR MES (ENERO–DICIEMBRE)
    # =====================================================
    st.subheader("📅 Performance mensual – Winrate, Ganancia y Score")

    monthly_trades = filtered_trades.copy()
    monthly_trades["Month"] = monthly_trades["Timestamp"].dt.month
    monthly_trades["MonthName"] = monthly_trades["Timestamp"].dt.month_name()

    # -------------------------
    # Métricas mensuales
    # -------------------------
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

    # -------------------------
    # SCORE MENSUAL PONDERADO
    # -------------------------
    if not monthly_stats.empty:

        # Normalizaciones
        pnl_n = monthly_stats["Profit"] - monthly_stats["Profit"].min()
        pnl_n = pnl_n / pnl_n.max() if pnl_n.max() != 0 else 0

        trades_n = monthly_stats["Trades"] / monthly_stats["Trades"].max()
        winrate_n = monthly_stats["Winrate"] / 100

        monthly_stats["Score"] = (
            pnl_n * 0.5 +
            winrate_n * 0.3 +
            trades_n * 0.2
        )

    # -------------------------
    # 📊 Gráfico: Profit mensual
    # -------------------------
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

    st.plotly_chart(fig_month_perf, width="stretch")

    # -------------------------
    # 📋 Tabla resumen mensual
    # -------------------------
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

    # -------------------------
    # 🥇 Mejor y peor mes
    # -------------------------
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


    monthly = (
        df_master["PORTFOLIO"]
        .resample("M")
        .sum()
    )

    fig_m = go.Figure()
    fig_m.add_trace(go.Bar(
        x=monthly.index,
        y=monthly.values
    ))
    fig_m.update_layout(height=400)
    st.plotly_chart(fig_m, width="stretch")

    st.dataframe(monthly.to_frame("PnL Mensual"))

    # =====================================================
    # 🔬 ANALISIS AVANZADO DE ROBUSTEZ (AGREGADO)
    # =====================================================

    st.subheader("🧠 Análisis Avanzado de Robustez")

    # -------------------------
    # Datos base
    # -------------------------
    pnl_trades = filtered_trades["PnL"].dropna()
    daily_returns = df_master["PORTFOLIO"]

    # -------------------------
    # Wins / Losses
    # -------------------------
    wins = pnl_trades[pnl_trades > 0]
    losses = pnl_trades[pnl_trades < 0]

    avg_win = wins.mean() if not wins.empty else 0
    avg_loss = abs(losses.mean()) if not losses.empty else 0

    # -------------------------
    # Métricas clave
    # -------------------------
    profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty else np.nan
    expectancy = pnl_trades.mean()
    risk_reward = avg_win / avg_loss if avg_loss != 0 else np.nan

    sharpe = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() != 0 else np.nan
    )

    downside = daily_returns[daily_returns < 0]
    sortino = (
        daily_returns.mean() / downside.std() * np.sqrt(252)
        if downside.std() != 0 else np.nan
    )

    # -------------------------
    # Drawdown extendido
    # -------------------------
    equity_adv = daily_returns.cumsum()
    peak_adv = equity_adv.cummax()
    drawdown_adv = equity_adv - peak_adv

    # Time Under Water
    time_under_water = (
        (drawdown_adv < 0)
        .astype(int)
        .groupby((drawdown_adv == 0).cumsum())
        .sum()
        .max()
    )

    # Max Losing Streak
    losing = (pnl_trades < 0).astype(int)
    max_losing_streak = losing.groupby((losing == 0).cumsum()).sum().max()

    # -------------------------
    # Mostrar métricas
    # -------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profit Factor", f"{profit_factor:.2f}")
    c2.metric("Expectancy / Trade", f"${expectancy:,.2f}")
    c3.metric("Risk / Reward", f"{risk_reward:.2f}")
    c4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Sortino Ratio", f"{sortino:.2f}")
    c2.metric("Max Losing Streak", int(max_losing_streak))
    c3.metric("Time Under Water (días)", int(time_under_water))

    # =====================================================
    # 🎲 MONTE CARLO – StrategyQuant Style
    # =====================================================
    st.subheader("🎲 Monte Carlo – Robustez (StrategyQuant)")

    mc_runs = 500
    TRADE_REMOVAL_PROB = 0.05
    NOISE_STD_FACTOR = 0.05

    mc_equities = []
    mc_max_dd = []
    mc_final_equity = []

    # Tomamos la columna de PnL que usamos para Equity
    pnl_array = filtered_trades["PnL"].dropna().values

    for _ in range(mc_runs):
        # Eliminación aleatoria de trades
        mask = np.random.rand(len(pnl_array)) > TRADE_REMOVAL_PROB
        sampled = pnl_array[mask]

        if len(sampled) == 0:
            continue

        # Mezclar trades y agregar ruido
        shuffled = np.random.permutation(sampled)
        noise_std = np.abs(np.mean(shuffled)) * NOISE_STD_FACTOR
        shuffled += np.random.normal(0, noise_std, size=len(shuffled))

        # Equity simulado
        equity = INITIAL_CAPITAL + np.cumsum(shuffled)

        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak

        mc_equities.append(equity)
        mc_max_dd.append(drawdown.min())
        mc_final_equity.append(equity[-1])

    # -------------------------------------
    # Resultados Monte Carlo
    # -------------------------------------
    mc_equity_df = pd.DataFrame(mc_equities).T  # columnas = simulaciones

    mc_stats = pd.DataFrame({
        "Final Equity": mc_final_equity,
        "Max Drawdown": mc_max_dd
    })

    # Añadimos columna de Profit neto
    mc_stats["Net Profit"] = mc_stats["Final Equity"] - INITIAL_CAPITAL
    # Número de simulación
    mc_stats.index = np.arange(1, len(mc_stats) + 1)
    mc_stats.index.name = "Simulación"

    # -------------------------------------
    # Plot Monte Carlo
    # -------------------------------------
    fig_mc = go.Figure()

    for i in range(min(40, mc_equity_df.shape[1])):
        fig_mc.add_trace(go.Scatter(
            y=mc_equity_df[i],
            line=dict(width=1),
            opacity=0.25,
            showlegend=False
        ))

    fig_mc.update_layout(
        height=450,
        title="Monte Carlo – StrategyQuant Robustness"
    )

    st.plotly_chart(fig_mc, width="stretch", key="montecarlo_equity")

    # -------------------------------------
    # Tabla de resultados individuales
    # -------------------------------------
    st.subheader("📋 Resultados de cada simulación")
    st.dataframe(
        mc_stats[["Net Profit", "Max Drawdown"]].style.format({
            "Net Profit": "${:,.2f}",
            "Max Drawdown": "${:,.2f}"
        })
    )



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
    st.plotly_chart(fig_year, width="stretch")

    st.dataframe(yearly.to_frame("PnL Anual"))
 

    # =====================================================
    # ⏳ TIME UNDER WATER – ANÁLISIS DETALLADO
    # =====================================================

    st.subheader("⏳ Time Under Water – Análisis Detallado")

    # Equity y drawdown base
    equity_tuw = df_master["PORTFOLIO"].cumsum()
    peak_tuw = equity_tuw.cummax()
    drawdown_tuw = equity_tuw - peak_tuw

    # Detectar períodos bajo agua
    underwater = drawdown_tuw < 0
    groups = (underwater != underwater.shift()).cumsum()

    tuw_periods = []

    for g, data in drawdown_tuw.groupby(groups):
        if data.iloc[0] < 0:
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

    # =========================
    # Métricas resumen
    # =========================
    if not df_tuw.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Cantidad de períodos TUW", len(df_tuw))
        c2.metric("Duración promedio (días)", f"{df_tuw['Duración (días)'].mean():.1f}")
        c3.metric("Duración máxima (días)", int(df_tuw["Duración (días)"].max()))
    else:
        st.info("No se detectaron períodos de Time Under Water.")

    # =========================
    # 📊 Gráfico: Duración de cada período TUW
    # =========================
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

        st.plotly_chart(fig_tuw, width="stretch")

 

    # =========================
    # 📋 Tabla detallada
    # =========================
    st.subheader("📋 Detalle completo de Time Under Water")

    if not df_tuw.empty:
        st.dataframe(df_tuw)

    # =====================================================
    # ⏰ ANALISIS POR HORARIO DE ENTRADA (30 MIN)
    # =====================================================

    time_stats = None

    st.subheader("⏰ Rendimiento por horario de EJECUCIÓN (Entry) – bloques de 30 minutos")

    # =========================
    # Detectar columna de ENTRADA (robusto)
    # =========================
    entry_col = "EntryTime" if "EntryTime" in filtered_trades.columns else None


    if entry_col is None:
        st.warning("⚠️ No se encontró la columna de horario de entrada (Entry Time / Tiempo de Entrada).")
    else:
        trades_time = filtered_trades.copy()

        # Asegurar datetime
        trades_time[entry_col] = parse_datetime_mixed(trades_time[entry_col])
        trades_time = trades_time.dropna(subset=[entry_col])

        # =========================
        # Bucket de 30 minutos (ENTRY)
        # =========================
        trades_time["TimeBucket"] = (
            trades_time[entry_col]
            .dt.floor("30T")
            .dt.strftime("%H:%M")
        )

        # =========================
        # Métricas por horario
        # =========================
        time_stats = trades_time.groupby("TimeBucket").agg(
            Trades=("PnL", "count"),
            Profit=("PnL", "sum"),
            Winrate=("PnL", lambda x: (x > 0).mean() * 100)
        ).reset_index()


        # Orden cronológico real
        time_stats["SortKey"] = pd.to_datetime(
            time_stats["TimeBucket"],
            format="%H:%M"
        )
        time_stats = time_stats.sort_values("SortKey")

        # Color según PnL
        colors = np.where(
            time_stats["Profit"] >= 0,
            "rgba(0,200,0,0.65)",
            "rgba(200,0,0,0.65)"
        )

        # =========================
        # 📊 Gráfico
        # =========================
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

        st.plotly_chart(fig_time, width="stretch")

        # =========================
        # 📋 Tabla resumen
        # =========================
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






# =====================================================
# 🔗 TAB CORRELACIÓN DE PORTAFOLIOS
# =====================================================
with tab_corr:

    st.subheader("🔗 Correlación entre Portafolios (Cuentas Independientes)")

    # -------------------------------------------------
    # Estrategias disponibles
    # -------------------------------------------------
    all_strategies = [c for c in df_master.columns if c != "PORTFOLIO"]

    st.markdown("### 📦 Construcción de Portafolios")

    portfolio_returns = {}

    # -------------------------------------------------
    # Portafolios 1 a 5
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Validación
    # -------------------------------------------------
    if len(portfolio_returns) < 2:
        st.warning("⚠️ Debés construir al menos 2 portafolios para medir correlación.")
        st.stop()

    # -------------------------------------------------
    # DataFrame de portafolios
    # -------------------------------------------------
    df_portfolios = pd.DataFrame(portfolio_returns)

    # =====================================================
    # 📈 Equity Curve por Portafolio
    # =====================================================
    st.subheader("📈 Equity Curve por Portafolio")

    fig_eq = go.Figure()
    for c in df_portfolios.columns:
        fig_eq.add_trace(go.Scatter(
            x=df_portfolios.index,
            y=df_portfolios[c].cumsum(),
            name=c
        ))

    fig_eq.update_layout(height=450)
    st.plotly_chart(fig_eq, width="stretch", key="equity_portfolios")




    # =====================================================
    # 📉 CORRELACIÓN POR DRAWDOWN (LA MÁS IMPORTANTE)
    # =====================================================
    st.subheader("📉 Correlación por DrawDown entre Portafolios")

    # -------------------------------------
    # Equity curves por portafolio
    # -------------------------------------
    equity_df = df_portfolios.cumsum()

    # -------------------------------------
    # Función de drawdown
    # -------------------------------------
    def compute_drawdown(equity_series):
        peak = equity_series.cummax()
        dd = (equity_series - peak) / peak
        return dd

    # -------------------------------------
    # Drawdown por portafolio
    # -------------------------------------
    dd_df = equity_df.apply(compute_drawdown)

    # -------------------------------------
    # Matriz de correlación por DrawDown
    # -------------------------------------
    dd_corr = dd_df.corr()

    # -------------------------------------
    # Heatmap DrawDown Correlation
    # -------------------------------------
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
        width="stretch",
        key="portfolio_dd_corr_heatmap"
    )

    # -------------------------------------
    # Tabla DrawDown Correlation
    # -------------------------------------
    st.subheader("📋 Matriz de correlación por DrawDown")
    st.dataframe(dd_corr.round(2), use_container_width=True)

    # -------------------------------------
    # Correlación promedio por DrawDown
    # -------------------------------------
    dd_avg_corr = dd_corr.values[
        np.triu_indices_from(dd_corr.values, k=1)
    ].mean()

    st.metric(
        "Correlación promedio por DrawDown",
        f"{dd_avg_corr:.2f}",
        help="Mide si los portafolios entran en drawdown al mismo tiempo. Cuanto más bajo o negativo, mejor."
    )

    # -------------------------------------
    # Pares con DrawDown descorrelado
    # -------------------------------------
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
    # 🧭 RECOMENDACIONES DE DIVERSIFICACIÓN (DRAW DOWN)
    # =====================================================
    st.subheader("🧭 Recomendación de Diversificación por DrawDown")

    # Texto benchmark visible
    st.markdown("""
    **📊 Benchmark profesional (Correlación por DrawDown):**

    - ❌ **> 0.50** → Riesgo conjunto alto (mala diversificación)  
    - ⚠️ **0.30 – 0.50** → Diversificación débil  
    - ✅ **0.15 – 0.30** → Buena diversificación  
    - 🟢 **0.00 – 0.15** → Muy buena  
    - 🟢🟢 **< 0.00** → Excelente (drawdowns se compensan)
    """)

    # Evaluación automática
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

    # Objetivo recomendado
    st.info(
        "🎯 **Objetivo recomendado:**\n\n"
        "Mantener la correlación promedio por DrawDown "
        "**≤ 0.20** para una diversificación robusta y escalable."
    )
