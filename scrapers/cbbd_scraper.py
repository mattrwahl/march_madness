"""
CBBD (College Basketball Data) API client.
Wraps the cbbd Python package for tournament games, ratings, stats, and lines.

CBBD game fields (confirmed from API):
  id, season, start_date, tournament, game_notes, neutral_site,
  home_team, away_team, home_points, away_points,
  home_seed, away_seed   <-- direct seed fields!
  game_notes contains round info, e.g.:
    "Men's Basketball Championship - West Region - 1st Round"
    "Men's Basketball Championship - Midwest Region - First Four"
"""
import sys
import logging
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cbbd
from config import CBBD_API_KEY

logger = logging.getLogger(__name__)


# Map CBBD game_notes round strings -> "teams remaining" round encoding
_ROUND_MAP = {
    "first four":     65,   # play-in; we'll skip these for betting
    "1st round":      64,
    "first round":    64,
    "2nd round":      32,
    "second round":   32,
    "sweet 16":       16,
    "sweet sixteen":  16,
    "elite eight":     8,
    "elite 8":         8,
    "regional final":  8,
    "final four":      4,
    "national semifinal": 4,
    "championship":    2,
    "national championship": 2,
}


def _parse_round_from_notes(notes: str | None) -> int | None:
    """
    Parse tournament round from CBBD game_notes string.
    Notes format: "Men's Basketball Championship - {Region} Region - {Round}"
    We only look at the LAST segment (after the final " - ") for round matching.
    """
    if not notes:
        return None
    # Use only the last segment to avoid matching "Championship" in the title
    parts = [p.strip() for p in notes.split(" - ")]
    round_segment = parts[-1].lower() if parts else notes.lower()
    for key, val in _ROUND_MAP.items():
        if key in round_segment:
            return val
    return None


def _parse_region_from_notes(notes: str | None) -> str | None:
    """Parse region name from CBBD game_notes string."""
    if not notes:
        return None
    # e.g. "Men's Basketball Championship - West Region - 1st Round"
    m = re.search(r"(East|West|South|Midwest|Southeast|Southwest)\s+Region", notes, re.IGNORECASE)
    return m.group(1).title() if m else None


def _get_api_client():
    config = cbbd.Configuration(access_token=CBBD_API_KEY)
    api_client = cbbd.ApiClient(config)
    return api_client


def get_tournament_games(season: int, tournament: str = "NCAA") -> list[dict]:
    """
    Fetch all tournament games for a given season.
    Returns a list of dicts with normalized fields including seeds and round.
    """
    with _get_api_client() as api_client:
        games_api = cbbd.GamesApi(api_client)
        games = games_api.get_games(season=season, tournament=tournament)

    results = []
    for g in games:
        try:
            notes = getattr(g, "game_notes", None)
            game_round = _parse_round_from_notes(notes)
            region = _parse_region_from_notes(notes)

            results.append({
                "cbbd_game_id": str(g.id),
                "season": g.season,
                "game_date": str(g.start_date)[:10] if g.start_date else None,
                "tournament": g.tournament,
                "notes": notes,
                "round": game_round,
                "region": region,
                "home_team": g.home_team,
                "away_team": g.away_team,
                "home_score": g.home_points,
                "away_score": g.away_points,
                "home_seed": g.home_seed,
                "away_seed": g.away_seed,
                "neutral_site": getattr(g, "neutral_site", True),
            })
        except Exception as e:
            logger.warning(f"Error parsing game {getattr(g, 'id', '?')}: {e}")

    logger.info(f"Fetched {len(results)} tournament games for {season}")
    return results


def get_adjusted_efficiency_ratings(season: int) -> list[dict]:
    """
    Fetch adjusted offensive/defensive efficiency ratings for a season.
    Maps to mm_team_metrics fields.

    CBBD method: RatingsApi.get_adjusted_efficiency(season=Y)
    Response fields: team, conference, offensive_rating, defensive_rating,
                     net_rating, rankings.net, rankings.defense, rankings.offense
    """
    with _get_api_client() as api_client:
        ratings_api = cbbd.RatingsApi(api_client)
        ratings = ratings_api.get_adjusted_efficiency(season=season)

    results = []
    for r in ratings:
        try:
            rankings = getattr(r, "rankings", None)
            results.append({
                "team": r.team,
                "conference": getattr(r, "conference", None),
                "adj_off_rating": r.offensive_rating,
                "adj_def_rating": r.defensive_rating,
                "net_rating": r.net_rating,
                "net_rank": rankings.net if rankings else None,
                "def_rank": rankings.defense if rankings else None,
            })
        except Exception as e:
            logger.warning(f"Error parsing rating for {getattr(r, 'team', '?')}: {e}")

    logger.info(f"Fetched efficiency ratings for {len(results)} teams ({season})")
    return results


def get_team_season_stats(season: int, season_type: str = "regular") -> list[dict]:
    """
    Fetch team season stats (four factors + pace + FT%) for pre-tournament metrics.
    season_type='regular' avoids postseason contamination.
    """
    with _get_api_client() as api_client:
        stats_api = cbbd.StatsApi(api_client)
        stats = stats_api.get_team_season_stats(season=season, season_type=season_type)

    results = []
    for s in stats:
        try:
            # Navigate nested stats object
            team_stats = getattr(s, "team_stats", s)
            opp_stats = getattr(s, "opponent_stats", None)
            four_factors = getattr(team_stats, "four_factors", None)
            opp_four_factors = getattr(opp_stats, "four_factors", None) if opp_stats else None
            free_throws = getattr(team_stats, "free_throws", None)

            # Handle both camelCase and snake_case field names
            def _ff(obj, *keys):
                if obj is None:
                    return None
                for k in keys:
                    v = getattr(obj, k, None)
                    if v is not None:
                        return v
                return None

            results.append({
                "team": s.team,
                "conference": getattr(s, "conference", None),
                "pace": getattr(s, "pace", None),
                "efg_pct": _ff(four_factors,
                               "effective_field_goal_pct", "effectiveFieldGoalPct", "efg_pct"),
                "opp_efg_pct": _ff(opp_four_factors,
                                   "effective_field_goal_pct", "effectiveFieldGoalPct", "efg_pct"),
                "tov_ratio": _ff(four_factors, "turnover_ratio", "turnoverRatio", "tov_ratio"),
                "oreb_pct": _ff(four_factors,
                                "offensive_rebound_pct", "offensiveReboundPct", "oreb_pct"),
                "ft_rate": _ff(four_factors, "free_throw_rate", "freeThrowRate", "ft_rate"),
                "ft_pct": _ff(free_throws, "pct", "free_throw_pct", "ft_pct"),
            })
        except Exception as e:
            logger.warning(f"Error parsing stats for {getattr(s, 'team', '?')}: {e}")

    logger.info(f"Fetched season stats for {len(results)} teams ({season} {season_type})")
    return results


def get_lines_providers() -> list[str]:
    """Return available betting line providers from CBBD."""
    with _get_api_client() as api_client:
        lines_api = cbbd.LinesApi(api_client)
        try:
            providers = lines_api.get_lines_providers()
            return [p.name if hasattr(p, "name") else str(p) for p in providers]
        except Exception as e:
            logger.warning(f"Could not fetch providers: {e}")
            return []


def get_betting_lines(season: int, game_ids: list[str] | None = None) -> list[dict]:
    """
    Fetch betting lines for NCAA tournament games in a season.
    Uses date range filtering to capture postseason games (which aren't returned
    by the default lines API call without a team filter).

    NCAA Tournament date ranges:
      First Four + R64 start ~March 19; Championship ~April 8.
    """
    from datetime import datetime as _dt
    start_date = _dt(season, 3, 15)
    end_date = _dt(season, 4, 10)

    with _get_api_client() as api_client:
        lines_api = cbbd.LinesApi(api_client)
        lines = lines_api.get_lines(
            season=season,
            start_date_range=start_date,
            end_date_range=end_date,
        )

    game_lines: dict[str, dict] = {}

    for entry in lines:
        try:
            gid = str(entry.game_id)
            if game_ids and gid not in game_ids:
                continue

            providers_data = getattr(entry, "lines", []) or []
            if not providers_data:
                continue

            # Prefer BetOnline
            betonline = [p for p in providers_data
                         if "betonline" in str(getattr(p, "provider", "")).lower()]
            chosen = betonline[0] if betonline else providers_data[0]

            home_ml = getattr(chosen, "home_moneyline", None)
            away_ml = getattr(chosen, "away_moneyline", None)
            provider_name = getattr(chosen, "provider", "unknown")

            if home_ml is None and away_ml is None:
                continue

            home_nv, away_nv = None, None
            if home_ml is not None and away_ml is not None:
                from shared.implied_probs import american_to_implied_prob, remove_vig
                h_raw = american_to_implied_prob(int(home_ml))
                a_raw = american_to_implied_prob(int(away_ml))
                home_nv, away_nv = remove_vig(h_raw, a_raw)

            game_lines[gid] = {
                "cbbd_game_id": gid,
                "provider": provider_name,
                "home_team": getattr(entry, "home_team", None),
                "away_team": getattr(entry, "away_team", None),
                "home_moneyline": int(home_ml) if home_ml is not None else None,
                "away_moneyline": int(away_ml) if away_ml is not None else None,
                "home_novig_prob": round(home_nv, 6) if home_nv else None,
                "away_novig_prob": round(away_nv, 6) if away_nv else None,
            }
        except Exception as e:
            logger.warning(f"Error parsing lines for game {getattr(entry, 'id', '?')}: {e}")

    results = list(game_lines.values())
    logger.info(f"Fetched lines for {len(results)} games ({season})")
    return results
