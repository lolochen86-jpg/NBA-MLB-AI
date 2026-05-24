from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


SIMULATIONS = 10_000
RNG = np.random.default_rng()


@dataclass(frozen=True)
class NBAProfile:
    team_name: str
    pace: float
    off_rating: float
    def_rating: float


@dataclass(frozen=True)
class MLBProfile:
    team_name: str
    fg_abbr: str
    team_ops: float
    league_ops: float
    pitcher_name: str
    pitcher_era: float
    pitcher_xfip: float | None
    league_runs_per_team_game: float


def current_nba_season(today: dt.date | None = None) -> str:
    today = today or dt.date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def current_mlb_season(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 3 else today.year - 1


def decimal_odds(probability: float) -> float:
    return float("inf") if probability <= 0 else 1 / probability


def require_columns(frame: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise RuntimeError(f"{source} 回傳資料缺少欄位：{', '.join(missing)}")


@st.cache_data(ttl=60 * 60)
def nba_team_options() -> pd.DataFrame:
    from nba_api.stats.static import teams

    return pd.DataFrame(teams.get_teams()).sort_values("full_name")


@st.cache_data(ttl=60 * 30)
def fetch_nba_advanced_stats(season: str, last_n_games: int) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashteamstats

    endpoint = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="Per100Possessions",
        last_n_games=last_n_games,
        timeout=45,
    )
    frame = endpoint.get_data_frames()[0]
    require_columns(frame, ["TEAM_ID", "TEAM_NAME", "PACE", "OFF_RATING", "DEF_RATING"], "nba_api")
    return frame


def make_nba_profile(stats: pd.DataFrame, team_id: int) -> NBAProfile:
    row = stats.loc[stats["TEAM_ID"].astype(int) == int(team_id)]
    if row.empty:
        raise RuntimeError("找不到此 NBA 球隊的進階數據，請確認 season / last N games 設定。")
    row = row.iloc[0]
    return NBAProfile(
        team_name=str(row["TEAM_NAME"]),
        pace=float(row["PACE"]),
        off_rating=float(row["OFF_RATING"]),
        def_rating=float(row["DEF_RATING"]),
    )


def simulate_nba(away: NBAProfile, home: NBAProfile, n: int = SIMULATIONS) -> pd.DataFrame:
    possessions = (away.pace + home.pace) / 2
    away_points_mean = possessions * ((away.off_rating + home.def_rating) / 2) / 100
    home_points_mean = possessions * ((home.off_rating + away.def_rating) / 2) / 100

    score_sd = max(9.0, possessions * 0.105)
    home_court_edge = 2.2
    away_scores = RNG.normal(away_points_mean, score_sd, n)
    home_scores = RNG.normal(home_points_mean + home_court_edge, score_sd, n)
    ties = away_scores == home_scores
    home_scores[ties] += RNG.normal(1.2, 0.8, ties.sum())

    return pd.DataFrame(
        {
            "away_score": np.maximum(0, away_scores),
            "home_score": np.maximum(0, home_scores),
            "margin_home": home_scores - away_scores,
            "winner": np.where(home_scores > away_scores, home.team_name, away.team_name),
        }
    )


@st.cache_data(ttl=60 * 60 * 6)
def mlb_team_options() -> pd.DataFrame:
    import statsapi

    teams = statsapi.get("teams", {"sportId": 1, "activeStatus": "Y"})["teams"]
    frame = pd.DataFrame(
        {
            "id": int(team["id"]),
            "name": team["name"],
            "teamName": team.get("teamName", team["name"]),
            "abbreviation": team.get("abbreviation"),
            "fileCode": team.get("fileCode"),
        }
        for team in teams
    )
    return frame.sort_values("name")


@st.cache_data(ttl=60 * 60 * 8)
def fetch_fangraphs_batting(season: int) -> pd.DataFrame:
    from pybaseball import batting_stats

    frame = batting_stats(season, qual=0)
    require_columns(frame, ["Team", "OPS"], "pybaseball batting_stats")
    return frame


@st.cache_data(ttl=60 * 60 * 8)
def fetch_fangraphs_pitching(season: int) -> pd.DataFrame:
    from pybaseball import pitching_stats

    frame = pitching_stats(season, qual=0)
    require_columns(frame, ["Name", "Team", "ERA"], "pybaseball pitching_stats")
    return frame


def normalize_abbr(value: Any) -> str:
    return str(value).upper().replace(" ", "").replace(".", "")


def fg_team_abbr(mlb_row: pd.Series) -> str:
    overrides = {
        "AZ": "ARI",
        "ATH": "OAK",
        "CWS": "CHW",
        "KC": "KCR",
        "LA": "LAD",
        "SD": "SDP",
        "SF": "SFG",
        "TB": "TBR",
        "WSH": "WSN",
    }
    abbr = normalize_abbr(mlb_row.get("abbreviation") or mlb_row.get("fileCode"))
    return overrides.get(abbr, abbr)


def team_batting_profile(batting: pd.DataFrame, fg_abbr: str) -> tuple[float, float]:
    team_rows = batting.loc[batting["Team"].map(normalize_abbr) == normalize_abbr(fg_abbr)]
    if team_rows.empty:
        raise RuntimeError(f"FanGraphs batting_stats 找不到 {fg_abbr} 的打擊數據。")
    plate_appearances = team_rows.get("PA", pd.Series(np.ones(len(team_rows)), index=team_rows.index))
    team_ops = float(np.average(team_rows["OPS"].astype(float), weights=plate_appearances.astype(float)))
    league_ops = float(np.average(batting["OPS"].astype(float), weights=batting.get("PA", pd.Series(np.ones(len(batting)))).astype(float)))
    return team_ops, league_ops


def pitcher_options(pitching: pd.DataFrame, fg_abbr: str) -> pd.DataFrame:
    rows = pitching.loc[pitching["Team"].map(normalize_abbr) == normalize_abbr(fg_abbr)].copy()
    if rows.empty:
        return pd.DataFrame(columns=pitching.columns)
    innings_col = "IP" if "IP" in rows.columns else None
    if innings_col:
        rows = rows.sort_values(innings_col, ascending=False)
    return rows


def pitcher_metric(row: pd.Series) -> tuple[float, float | None]:
    era = float(row["ERA"])
    xfip = None
    for col in ("xFIP", "SIERA", "FIP"):
        if col in row.index and pd.notna(row[col]):
            xfip = float(row[col])
            break
    return era, xfip


@st.cache_data(ttl=60 * 60)
def fetch_mlb_league_runs_per_team_game(season: int) -> float:
    import statsapi

    today = dt.date.today()
    end_date = min(today, dt.date(int(season), 10, 31))
    start_date = dt.date(int(season), 3, 1)
    if end_date < start_date:
        raise RuntimeError("此 MLB season 尚未有可用例行賽比分，請選擇已有完賽資料的球季。")

    games = statsapi.schedule(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        sportId=1,
        gameType="R",
    )
    total_runs = 0
    team_games = 0
    for game in games:
        if "Final" not in str(game.get("status", "")):
            continue
        if game.get("home_score") is None or game.get("away_score") is None:
            continue
        total_runs += int(game["home_score"]) + int(game["away_score"])
        team_games += 2

    if team_games == 0:
        raise RuntimeError("statsapi 尚無此條件下的 MLB 完賽比分，無法估算聯盟平均得分。")
    return total_runs / team_games


@st.cache_data(ttl=60 * 15)
def probable_pitchers(team_id_a: int, team_id_b: int, game_date: dt.date) -> dict[str, str]:
    import statsapi

    schedule = statsapi.schedule(
        date=game_date.strftime("%Y-%m-%d"),
        sportId=1,
        team=team_id_a,
    )
    for game in schedule:
        if {int(game.get("home_id", 0)), int(game.get("away_id", 0))} == {int(team_id_a), int(team_id_b)}:
            return {
                str(game.get("away_id")): game.get("away_probable_pitcher", ""),
                str(game.get("home_id")): game.get("home_probable_pitcher", ""),
            }
    return {}


def make_mlb_profile(
    team_row: pd.Series,
    batting: pd.DataFrame,
    pitcher_row: pd.Series,
    league_runs_per_team_game: float,
) -> MLBProfile:
    fg_abbr = fg_team_abbr(team_row)
    team_ops, league_ops = team_batting_profile(batting, fg_abbr)
    era, xfip = pitcher_metric(pitcher_row)
    return MLBProfile(
        team_name=str(team_row["name"]),
        fg_abbr=fg_abbr,
        team_ops=team_ops,
        league_ops=league_ops,
        pitcher_name=str(pitcher_row["Name"]),
        pitcher_era=era,
        pitcher_xfip=xfip,
        league_runs_per_team_game=league_runs_per_team_game,
    )


def expected_mlb_runs(offense: MLBProfile, opponent_pitcher: MLBProfile, home: bool) -> float:
    pitcher_run_index = (opponent_pitcher.pitcher_xfip or opponent_pitcher.pitcher_era) / 4.20
    offense_index = offense.team_ops / offense.league_ops
    home_edge = 1.035 if home else 0.985
    expected = offense.league_runs_per_team_game * offense_index * pitcher_run_index * home_edge
    return float(np.clip(expected, 1.5, 8.5))


def simulate_mlb(away: MLBProfile, home: MLBProfile, n: int = SIMULATIONS) -> pd.DataFrame:
    away_lambda = expected_mlb_runs(away, home, home=False)
    home_lambda = expected_mlb_runs(home, away, home=True)
    away_runs = RNG.poisson(away_lambda, n)
    home_runs = RNG.poisson(home_lambda, n)

    tied = away_runs == home_runs
    while tied.any():
        away_extra = RNG.poisson(0.46, tied.sum())
        home_extra = RNG.poisson(0.49, tied.sum())
        away_runs[tied] += away_extra
        home_runs[tied] += home_extra
        tied = away_runs == home_runs

    return pd.DataFrame(
        {
            "away_score": away_runs,
            "home_score": home_runs,
            "margin_home": home_runs - away_runs,
            "winner": np.where(home_runs > away_runs, home.team_name, away.team_name),
        }
    )


def summarize_results(results: pd.DataFrame, away_name: str, home_name: str) -> pd.DataFrame:
    away_win_prob = float((results["winner"] == away_name).mean())
    home_win_prob = float((results["winner"] == home_name).mean())
    return pd.DataFrame(
        [
            {"Team": away_name, "Win Probability": away_win_prob, "Fair Decimal Odds": decimal_odds(away_win_prob)},
            {"Team": home_name, "Win Probability": home_win_prob, "Fair Decimal Odds": decimal_odds(home_win_prob)},
        ]
    )


def render_result_charts(results: pd.DataFrame, summary: pd.DataFrame, away_name: str, home_name: str) -> None:
    margin_distribution = (
        results.assign(home_margin=results["margin_home"].round().astype(int))
        .groupby("home_margin", as_index=False)
        .size()
        .rename(columns={"size": "simulations"})
    )
    margin_distribution["probability"] = margin_distribution["simulations"] / len(results)
    margin_distribution["home_margin_label"] = margin_distribution["home_margin"].map(lambda x: f"{x:+d}")

    margin_bar = px.bar(
        margin_distribution,
        x="home_margin",
        y="probability",
        title=f"主場得失分分布：{home_name} 分差（贏為正、輸為負）",
        labels={
            "home_margin": "主隊分差",
            "probability": "模擬機率",
            "home_margin_label": "主隊分差",
            "simulations": "模擬次數",
        },
        hover_data={"home_margin": False, "home_margin_label": True, "probability": ":.2%", "simulations": True},
    )
    margin_bar.add_vline(x=0, line_dash="dash", line_color="red")
    margin_bar.update_yaxes(tickformat=".0%")
    margin_bar.update_traces(marker_color="#2563eb")
    st.plotly_chart(margin_bar, use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        pie = px.pie(summary, names="Team", values="Win Probability", hole=0.42, title="模擬勝率")
        pie.update_traces(textinfo="label+percent")
        st.plotly_chart(pie, use_container_width=True)
    with right:
        spread = px.histogram(
            results,
            x="margin_home",
            nbins=50,
            title=f"主隊分差分佈：{home_name} - {away_name}",
        )
        spread.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(spread, use_container_width=True)

    display = summary.copy()
    display["Win Probability"] = display["Win Probability"].map(lambda x: f"{x:.2%}")
    display["Fair Decimal Odds"] = display["Fair Decimal Odds"].map(lambda x: f"{x:.2f}")
    st.dataframe(display, hide_index=True, use_container_width=True)


def render_saved_results(state_key: str) -> None:
    st.subheader("模擬結果")
    payload = st.session_state.get(state_key)
    if not payload:
        empty_distribution = pd.DataFrame(
            {
                "home_margin": list(range(-10, 11)),
                "probability": [0.0] * 21,
                "home_margin_label": [f"{margin:+d}" for margin in range(-10, 11)],
                "simulations": [0] * 21,
            }
        )
        empty_chart = px.bar(
            empty_distribution,
            x="home_margin",
            y="probability",
            title="主場得失分分布：等待模擬結果",
            labels={
                "home_margin": "主隊分差",
                "probability": "模擬機率",
                "home_margin_label": "主隊分差",
                "simulations": "模擬次數",
            },
            hover_data={"home_margin": False, "home_margin_label": True, "probability": ":.2%", "simulations": True},
        )
        empty_chart.add_vline(x=0, line_dash="dash", line_color="red")
        empty_chart.update_yaxes(tickformat=".0%", range=[0, 1])
        empty_chart.update_traces(marker_color="#cbd5e1")
        st.plotly_chart(empty_chart, use_container_width=True)
        st.info("按下「執行模擬」後，首頁這裡會顯示主場得失分分布圖、勝率圖與合理賠率。")
        return

    st.metric(payload["metric_label"], payload["metric_value"])
    st.caption(payload["caption"])
    render_result_charts(payload["results"], payload["summary"], payload["away_name"], payload["home_name"])


def render_nba() -> None:
    st.subheader("NBA Monte Carlo")
    teams = nba_team_options()
    season = st.text_input("NBA Season", value=current_nba_season())
    last_n_games = st.slider("近期場數", min_value=5, max_value=82, value=15, step=5)

    away_name = st.selectbox("客隊", teams["full_name"], index=0, key="nba_away")
    home_name = st.selectbox("主隊", teams["full_name"], index=1, key="nba_home")
    away_id = int(teams.loc[teams["full_name"] == away_name, "id"].iloc[0])
    home_id = int(teams.loc[teams["full_name"] == home_name, "id"].iloc[0])

    if st.button("執行 NBA 10,000 次模擬", type="primary", use_container_width=True):
        if away_id == home_id:
            st.error("請選擇兩支不同球隊。")
            return
        with st.spinner("抓取 nba_api 進階數據並執行模擬..."):
            stats = fetch_nba_advanced_stats(season, last_n_games)
            away = make_nba_profile(stats, away_id)
            home = make_nba_profile(stats, home_id)
            results = simulate_nba(away, home)
            summary = summarize_results(results, away.team_name, home.team_name)

        st.session_state["nba_results"] = {
            "results": results,
            "summary": summary,
            "away_name": away.team_name,
            "home_name": home.team_name,
            "metric_label": "平均主隊讓分",
            "metric_value": f"{results['margin_home'].mean():+.2f}",
            "caption": (
                f"Pace/OffRtg/DefRtg：{away.team_name} {away.pace:.1f}/{away.off_rating:.1f}/{away.def_rating:.1f}，"
                f"{home.team_name} {home.pace:.1f}/{home.off_rating:.1f}/{home.def_rating:.1f}"
            ),
        }

def select_pitcher(label: str, pitching: pd.DataFrame, fg_abbr: str, probable_name: str = "") -> pd.Series:
    rows = pitcher_options(pitching, fg_abbr)
    if rows.empty:
        raise RuntimeError(f"找不到 {fg_abbr} 的投手資料。")

    names = rows["Name"].dropna().astype(str).tolist()
    default_index = 0
    if probable_name:
        matches = [i for i, name in enumerate(names) if probable_name.lower() in name.lower() or name.lower() in probable_name.lower()]
        if matches:
            default_index = matches[0]
    selected = st.selectbox(label, names, index=default_index, key=label)
    return rows.loc[rows["Name"].astype(str) == selected].iloc[0]


def render_mlb() -> None:
    st.subheader("MLB Monte Carlo")
    season = st.number_input("MLB Season", min_value=2018, max_value=dt.date.today().year, value=current_mlb_season())
    game_date = st.date_input("比賽日期（用於抓取 probable pitchers）", value=dt.date.today())

    teams = mlb_team_options()
    away_name = st.selectbox("客隊", teams["name"], index=0, key="mlb_away")
    home_name = st.selectbox("主隊", teams["name"], index=1, key="mlb_home")
    away_row = teams.loc[teams["name"] == away_name].iloc[0]
    home_row = teams.loc[teams["name"] == home_name].iloc[0]

    with st.spinner("載入 FanGraphs 打擊/投手資料..."):
        batting = fetch_fangraphs_batting(int(season))
        pitching = fetch_fangraphs_pitching(int(season))

    probable = probable_pitchers(int(away_row["id"]), int(home_row["id"]), game_date)
    if probable:
        st.caption(
            "statsapi probable pitchers："
            f"{away_name} {probable.get(str(away_row['id']), 'N/A')}，"
            f"{home_name} {probable.get(str(home_row['id']), 'N/A')}"
        )

    away_pitcher = select_pitcher(
        f"{away_name} 先發投手",
        pitching,
        fg_team_abbr(away_row),
        probable.get(str(away_row["id"]), ""),
    )
    home_pitcher = select_pitcher(
        f"{home_name} 先發投手",
        pitching,
        fg_team_abbr(home_row),
        probable.get(str(home_row["id"]), ""),
    )

    league_runs = fetch_mlb_league_runs_per_team_game(int(season))
    st.caption(f"statsapi 估算本季聯盟平均每隊每場得分：{league_runs:.2f}")
    if st.button("執行 MLB 10,000 次模擬", type="primary", use_container_width=True):
        if int(away_row["id"]) == int(home_row["id"]):
            st.error("請選擇兩支不同球隊。")
            return
        with st.spinner("參數化投手壓制力與打線火力，執行泊松 9 局模擬..."):
            away = make_mlb_profile(away_row, batting, away_pitcher, league_runs)
            home = make_mlb_profile(home_row, batting, home_pitcher, league_runs)
            results = simulate_mlb(away, home)
            summary = summarize_results(results, away.team_name, home.team_name)

        st.session_state["mlb_results"] = {
            "results": results,
            "summary": summary,
            "away_name": away.team_name,
            "home_name": home.team_name,
            "metric_label": "平均主隊分差",
            "metric_value": f"{results['margin_home'].mean():+.2f}",
            "caption": (
                f"{away.team_name} OPS {away.team_ops:.3f}，SP {away.pitcher_name} ERA {away.pitcher_era:.2f}"
                f"{'' if away.pitcher_xfip is None else f' / xFIP/FIP {away.pitcher_xfip:.2f}'}；"
                f"{home.team_name} OPS {home.team_ops:.3f}，SP {home.pitcher_name} ERA {home.pitcher_era:.2f}"
                f"{'' if home.pitcher_xfip is None else f' / xFIP/FIP {home.pitcher_xfip:.2f}'}"
            ),
        }

def main() -> None:
    st.set_page_config(
        page_title="MLB / NBA Monte Carlo Simulator",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("賽事預測蒙地卡羅模擬器")
    st.caption("即時串接 nba_api、statsapi 與 pybaseball；不使用寫死假數據。")

    st.info(
        "模型輸出是量化估計，不是投注建議。若資料 API 暫時限流或無當日先發，請調整 season、日期或手動選擇投手。"
    )

    sport = st.segmented_control("運動類型", ["NBA", "MLB"], default="NBA")
    render_saved_results("nba_results" if sport == "NBA" else "mlb_results")
    st.divider()

    try:
        if sport == "NBA":
            render_nba()
        else:
            render_mlb()
    except Exception as exc:
        st.error(str(exc))
        st.exception(exc)


if __name__ == "__main__":
    main()
