from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


SIMULATIONS = 10_000
RNG = np.random.default_rng()
MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"


NBA_TEAM_ZH = {
    "Atlanta Hawks": "亞特蘭大老鷹",
    "Boston Celtics": "波士頓塞爾提克",
    "Brooklyn Nets": "布魯克林籃網",
    "Charlotte Hornets": "夏洛特黃蜂",
    "Chicago Bulls": "芝加哥公牛",
    "Cleveland Cavaliers": "克里夫蘭騎士",
    "Dallas Mavericks": "達拉斯獨行俠",
    "Denver Nuggets": "丹佛金塊",
    "Detroit Pistons": "底特律活塞",
    "Golden State Warriors": "金州勇士",
    "Houston Rockets": "休士頓火箭",
    "Indiana Pacers": "印第安那溜馬",
    "LA Clippers": "洛杉磯快艇",
    "Los Angeles Clippers": "洛杉磯快艇",
    "Los Angeles Lakers": "洛杉磯湖人",
    "Memphis Grizzlies": "曼菲斯灰熊",
    "Miami Heat": "邁阿密熱火",
    "Milwaukee Bucks": "密爾瓦基公鹿",
    "Minnesota Timberwolves": "明尼蘇達灰狼",
    "New Orleans Pelicans": "紐奧良鵜鶘",
    "New York Knicks": "紐約尼克",
    "Oklahoma City Thunder": "奧克拉荷馬雷霆",
    "Orlando Magic": "奧蘭多魔術",
    "Philadelphia 76ers": "費城76人",
    "Phoenix Suns": "鳳凰城太陽",
    "Portland Trail Blazers": "波特蘭拓荒者",
    "Sacramento Kings": "沙加緬度國王",
    "San Antonio Spurs": "聖安東尼奧馬刺",
    "Toronto Raptors": "多倫多暴龍",
    "Utah Jazz": "猶他爵士",
    "Washington Wizards": "華盛頓巫師",
    "76ers": "費城76人",
    "Bucks": "密爾瓦基公鹿",
    "Bulls": "芝加哥公牛",
    "Cavaliers": "克里夫蘭騎士",
    "Celtics": "波士頓塞爾提克",
    "Clippers": "洛杉磯快艇",
    "Grizzlies": "曼菲斯灰熊",
    "Hawks": "亞特蘭大老鷹",
    "Heat": "邁阿密熱火",
    "Hornets": "夏洛特黃蜂",
    "Jazz": "猶他爵士",
    "Kings": "沙加緬度國王",
    "Knicks": "紐約尼克",
    "Lakers": "洛杉磯湖人",
    "Magic": "奧蘭多魔術",
    "Mavericks": "達拉斯獨行俠",
    "Nets": "布魯克林籃網",
    "Nuggets": "丹佛金塊",
    "Pacers": "印第安那溜馬",
    "Pelicans": "紐奧良鵜鶘",
    "Pistons": "底特律活塞",
    "Raptors": "多倫多暴龍",
    "Rockets": "休士頓火箭",
    "Spurs": "聖安東尼奧馬刺",
    "Suns": "鳳凰城太陽",
    "Thunder": "奧克拉荷馬雷霆",
    "Timberwolves": "明尼蘇達灰狼",
    "Trail Blazers": "波特蘭拓荒者",
    "Warriors": "金州勇士",
    "Wizards": "華盛頓巫師",
}


MLB_TEAM_ZH = {
    "Arizona Diamondbacks": "亞利桑那響尾蛇",
    "Athletics": "運動家",
    "Atlanta Braves": "亞特蘭大勇士",
    "Baltimore Orioles": "巴爾的摩金鶯",
    "Boston Red Sox": "波士頓紅襪",
    "Chicago Cubs": "芝加哥小熊",
    "Chicago White Sox": "芝加哥白襪",
    "Cincinnati Reds": "辛辛那提紅人",
    "Cleveland Guardians": "克里夫蘭守護者",
    "Colorado Rockies": "科羅拉多洛磯",
    "Detroit Tigers": "底特律老虎",
    "Houston Astros": "休士頓太空人",
    "Kansas City Royals": "堪薩斯市皇家",
    "Los Angeles Angels": "洛杉磯天使",
    "Los Angeles Dodgers": "洛杉磯道奇",
    "Miami Marlins": "邁阿密馬林魚",
    "Milwaukee Brewers": "密爾瓦基釀酒人",
    "Minnesota Twins": "明尼蘇達雙城",
    "New York Mets": "紐約大都會",
    "New York Yankees": "紐約洋基",
    "Oakland Athletics": "奧克蘭運動家",
    "Philadelphia Phillies": "費城費城人",
    "Pittsburgh Pirates": "匹茲堡海盜",
    "Sacramento Athletics": "沙加緬度運動家",
    "San Diego Padres": "聖地牙哥教士",
    "San Francisco Giants": "舊金山巨人",
    "Seattle Mariners": "西雅圖水手",
    "St. Louis Cardinals": "聖路易紅雀",
    "Tampa Bay Rays": "坦帕灣光芒",
    "Texas Rangers": "德州遊騎兵",
    "Toronto Blue Jays": "多倫多藍鳥",
    "Washington Nationals": "華盛頓國民",
}


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


def zh_name(name: str, mapping: dict[str, str]) -> str:
    return mapping.get(str(name), str(name))


def display_team_name(name: str, mapping: dict[str, str]) -> str:
    translated = zh_name(name, mapping)
    return translated if translated == name else f"{translated} ({name})"


def require_columns(frame: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise RuntimeError(f"{source} 回傳資料缺少欄位：{', '.join(missing)}")


@st.cache_data(ttl=60 * 60)
def nba_team_options() -> pd.DataFrame:
    from nba_api.stats.static import teams

    frame = pd.DataFrame(teams.get_teams()).sort_values("full_name")
    frame["display_name"] = frame["full_name"].map(lambda name: display_team_name(name, NBA_TEAM_ZH))
    return frame


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
        team_name=zh_name(str(row["TEAM_NAME"]), NBA_TEAM_ZH),
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
    frame["display_name"] = frame["name"].map(lambda name: display_team_name(name, MLB_TEAM_ZH))
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


def numeric_stat(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", "-.--"):
        return default
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def mlb_api_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(f"{MLB_STATS_API_BASE}/{path.lstrip('/')}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def team_stat_splits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    splits: list[dict[str, Any]] = []
    for stat_block in payload.get("stats", []):
        splits.extend(stat_block.get("splits", []))
    return splits


@st.cache_data(ttl=60 * 60)
def fetch_mlb_official_team_stats(season: int, team_refs: tuple[tuple[int, str], ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_ids = ",".join(str(team_id) for team_id, _ in team_refs)
    id_to_abbr = {int(team_id): abbr for team_id, abbr in team_refs}

    def fetch_group(group: str) -> list[dict[str, Any]]:
        params = {
            "teamIds": team_ids,
            "stats": "season",
            "group": group,
            "season": int(season),
            "sportIds": 1,
            "gameType": "R",
        }
        payload = mlb_api_get("teams/stats", params)
        splits = team_stat_splits(payload)
        if splits:
            return splits

        fallback_splits: list[dict[str, Any]] = []
        for team_id, _ in team_refs:
            team_payload = mlb_api_get(
                f"teams/{team_id}/stats",
                {
                    "stats": "season",
                    "group": group,
                    "season": int(season),
                    "sportId": 1,
                    "gameType": "R",
                },
            )
            fallback_splits.extend(team_stat_splits(team_payload))
        return fallback_splits

    batting_rows = []
    for split in fetch_group("hitting"):
        team_id = int(split.get("team", {}).get("id", 0))
        if team_id not in id_to_abbr:
            continue
        stat = split.get("stat", {})
        ops = numeric_stat(stat.get("ops"))
        if ops is None:
            continue
        batting_rows.append(
            {
                "Team": id_to_abbr[team_id],
                "OPS": ops,
                "PA": numeric_stat(stat.get("plateAppearances"), 1.0) or 1.0,
            }
        )

    pitching_rows = []
    for split in fetch_group("pitching"):
        team_id = int(split.get("team", {}).get("id", 0))
        if team_id not in id_to_abbr:
            continue
        stat = split.get("stat", {})
        era = numeric_stat(stat.get("era"))
        if era is None:
            continue
        pitching_rows.append(
            {
                "Team": id_to_abbr[team_id],
                "Name": "(使用球隊 ERA)",
                "ERA": era,
                "IP": numeric_stat(stat.get("inningsPitched"), 0.0) or 0.0,
                "Source": "MLB 官方團隊ERA",
            }
        )

    batting = pd.DataFrame(batting_rows)
    pitching = pd.DataFrame(pitching_rows)
    require_columns(batting, ["Team", "OPS"], "MLB 官方 hitting stats")
    require_columns(pitching, ["Team", "Name", "ERA"], "MLB 官方 pitching stats")
    return batting, pitching


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


def mlb_abbr_aliases(fg_abbr: str) -> set[str]:
    aliases = {normalize_abbr(fg_abbr)}
    reverse_aliases = {
        "ARI": "AZ",
        "CHW": "CWS",
        "KCR": "KC",
        "LAD": "LA",
        "OAK": "ATH",
        "SDP": "SD",
        "SFG": "SF",
        "TBR": "TB",
        "WSN": "WSH",
    }
    if normalize_abbr(fg_abbr) in reverse_aliases:
        aliases.add(reverse_aliases[normalize_abbr(fg_abbr)])
    return aliases


def team_batting_profile(batting: pd.DataFrame, fg_abbr: str) -> tuple[float, float]:
    team_rows = batting.loc[batting["Team"].map(normalize_abbr).isin(mlb_abbr_aliases(fg_abbr))]
    if team_rows.empty:
        raise RuntimeError(f"找不到 {fg_abbr} 的打擊 OPS 數據。")
    plate_appearances = team_rows.get("PA", pd.Series(np.ones(len(team_rows)), index=team_rows.index))
    team_ops = float(np.average(team_rows["OPS"].astype(float), weights=plate_appearances.astype(float)))
    league_ops = float(np.average(batting["OPS"].astype(float), weights=batting.get("PA", pd.Series(np.ones(len(batting)))).astype(float)))
    return team_ops, league_ops


def pitcher_options(pitching: pd.DataFrame, fg_abbr: str) -> pd.DataFrame:
    rows = pitching.loc[pitching["Team"].map(normalize_abbr).isin(mlb_abbr_aliases(fg_abbr))].copy()
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

    schedule_payload = statsapi.get(
        "schedule",
        {
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "sportId": 1,
            "gameType": "R",
        },
    )
    games = [game for date_block in schedule_payload.get("dates", []) for game in date_block.get("games", [])]
    total_runs = 0
    team_games = 0
    for game in games:
        status = game.get("status", {})
        detailed_state = status.get("detailedState") if isinstance(status, dict) else str(status)
        if "Final" not in str(detailed_state):
            continue
        teams = game.get("teams", {})
        home_score = teams.get("home", {}).get("score", game.get("home_score"))
        away_score = teams.get("away", {}).get("score", game.get("away_score"))
        if home_score is None or away_score is None:
            continue
        total_runs += int(home_score) + int(away_score)
        team_games += 2

    if team_games == 0:
        raise RuntimeError("statsapi 尚無此條件下的 MLB 完賽比分，無法估算聯盟平均得分。")
    return total_runs / team_games


@st.cache_data(ttl=60 * 10)
def fetch_mlb_schedule_games(game_date: dt.date) -> pd.DataFrame:
    import statsapi

    payload = statsapi.get(
        "schedule",
        {
            "date": game_date.strftime("%Y-%m-%d"),
            "sportId": 1,
            "hydrate": "probablePitcher",
        },
    )
    rows = []
    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            teams = game.get("teams", {})
            away = teams.get("away", {})
            home = teams.get("home", {})
            away_team = away.get("team", {})
            home_team = home.get("team", {})
            away_name = away_team.get("name", "")
            home_name = home_team.get("name", "")
            away_probable = away.get("probablePitcher", {}).get("fullName", "")
            home_probable = home.get("probablePitcher", {}).get("fullName", "")
            status = game.get("status", {}).get("detailedState", "")
            game_number = int(game.get("gameNumber", 1) or 1)
            suffix = f" G{game_number}" if game_number > 1 else ""
            rows.append(
                {
                    "game_pk": int(game.get("gamePk", 0)),
                    "away_id": int(away_team.get("id", 0)),
                    "home_id": int(home_team.get("id", 0)),
                    "away_name": away_name,
                    "home_name": home_name,
                    "away_probable": away_probable,
                    "home_probable": home_probable,
                    "status": status,
                    "display_name": (
                        f"{zh_name(away_name, MLB_TEAM_ZH)} @ {zh_name(home_name, MLB_TEAM_ZH)}"
                        f"{suffix} - {status or 'Scheduled'}"
                    ),
                }
            )
    return pd.DataFrame(rows)


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
        team_name=zh_name(str(team_row["name"]), MLB_TEAM_ZH),
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
        st.info("尚未執行模擬。請先選擇主客隊與參數，再按下方的「執行模擬」。")
        return

    st.metric(payload["metric_label"], payload["metric_value"])
    st.caption(payload["caption"])
    render_result_charts(payload["results"], payload["summary"], payload["away_name"], payload["home_name"])


def render_nba() -> None:
    st.subheader("NBA Monte Carlo")
    teams = nba_team_options()
    season = st.text_input("NBA Season", value=current_nba_season())
    last_n_games = st.slider("近期場數", min_value=5, max_value=82, value=15, step=5)

    away_display = st.selectbox("客隊", teams["display_name"], index=0, key="nba_away")
    home_display = st.selectbox("主隊", teams["display_name"], index=1, key="nba_home")
    away_id = int(teams.loc[teams["display_name"] == away_display, "id"].iloc[0])
    home_id = int(teams.loc[teams["display_name"] == home_display, "id"].iloc[0])

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


def pitcher_choice_options(probable_name: str = "") -> list[str]:
    options = ["(使用球隊 ERA)"]
    if probable_name:
        options.append(f"{probable_name}（使用球隊 ERA）")
    return options


def official_pitcher_row(pitching: pd.DataFrame, fg_abbr: str, selected_choice: str) -> pd.Series:
    rows = pitcher_options(pitching, fg_abbr)
    if rows.empty:
        raise RuntimeError(f"找不到 {fg_abbr} 的官方投手/團隊 ERA。")
    row = rows.iloc[0].copy()
    if selected_choice != "(使用球隊 ERA)":
        row["Name"] = selected_choice.replace("（使用球隊 ERA）", "")
    return row


def render_mlb() -> None:
    st.subheader("MLB Monte Carlo")
    game_date = st.date_input("比賽日期", value=dt.date.today())
    season = current_mlb_season(game_date)
    st.caption(f"MLB Season：{season}")

    teams = mlb_team_options()
    schedule_games = fetch_mlb_schedule_games(game_date)
    if schedule_games.empty:
        st.warning("這個日期沒有 MLB 官方賽程，請改選其他日期。")
        return

    selected_game = st.selectbox("選擇賽程對戰", schedule_games["display_name"], key="mlb_game")
    game_row = schedule_games.loc[schedule_games["display_name"] == selected_game].iloc[0]
    away_row = teams.loc[teams["id"].astype(int) == int(game_row["away_id"])]
    home_row = teams.loc[teams["id"].astype(int) == int(game_row["home_id"])]
    if away_row.empty or home_row.empty:
        st.error("找不到此賽程對應的 MLB 球隊資料。")
        return
    away_row = away_row.iloc[0]
    home_row = home_row.iloc[0]
    away_probable = str(game_row.get("away_probable", "") or "")
    home_probable = str(game_row.get("home_probable", "") or "")
    away_pitcher_choice = f"{away_probable}（使用球隊 ERA）" if away_probable else "(使用球隊 ERA)"
    home_pitcher_choice = f"{home_probable}（使用球隊 ERA）" if home_probable else "(使用球隊 ERA)"

    st.caption(
        "MLB 官方賽程先發："
        f"{zh_name(str(game_row['away_name']), MLB_TEAM_ZH)} {away_probable or '未公布，使用球隊 ERA'}；"
        f"{zh_name(str(game_row['home_name']), MLB_TEAM_ZH)} {home_probable or '未公布，使用球隊 ERA'}"
    )

    use_fangraphs = st.checkbox(
        "嘗試 FanGraphs/pybaseball 取得 xFIP（若被 403 擋下，會自動改用 MLB 官方 OPS/ERA）",
        value=False,
    )

    if st.button("執行 MLB 10,000 次模擬", type="primary", use_container_width=True):
        if int(away_row["id"]) == int(home_row["id"]):
            st.error("請選擇兩支不同球隊。")
            return
        with st.spinner("抓取 MLB 官方 OPS/ERA，參數化投手壓制力與打線火力..."):
            team_refs = tuple((int(row["id"]), fg_team_abbr(row)) for _, row in teams.iterrows())
            batting, pitching = fetch_mlb_official_team_stats(int(season), team_refs)
            league_runs = fetch_mlb_league_runs_per_team_game(int(season))
            data_source_note = "資料源：MLB 官方 Stats API OPS / 團隊 ERA"

            if use_fangraphs:
                try:
                    fangraphs_batting = fetch_fangraphs_batting(int(season))
                    fangraphs_pitching = fetch_fangraphs_pitching(int(season))
                    if not fangraphs_batting.empty and not fangraphs_pitching.empty:
                        batting = fangraphs_batting
                        pitching = fangraphs_pitching
                        away_pitcher_choice = "(使用球隊 ERA)"
                        home_pitcher_choice = "(使用球隊 ERA)"
                        data_source_note = "資料源：FanGraphs/pybaseball OPS + ERA/xFIP"
                except Exception as exc:
                    st.warning(f"FanGraphs/pybaseball 無法使用，已改用 MLB 官方資料：{exc}")

            away_pitcher = official_pitcher_row(pitching, fg_team_abbr(away_row), away_pitcher_choice)
            home_pitcher = official_pitcher_row(pitching, fg_team_abbr(home_row), home_pitcher_choice)
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
                f"{'' if home.pitcher_xfip is None else f' / xFIP/FIP {home.pitcher_xfip:.2f}'}。"
                f"{data_source_note}，本季聯盟平均每隊每場得分 {league_runs:.2f}"
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
    try:
        if sport == "NBA":
            render_nba()
        else:
            render_mlb()

        st.divider()
        render_saved_results("nba_results" if sport == "NBA" else "mlb_results")
    except Exception as exc:
        st.error(str(exc))
        st.exception(exc)


if __name__ == "__main__":
    main()
