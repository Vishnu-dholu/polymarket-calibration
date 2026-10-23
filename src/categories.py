"""Map Polymarket events to broad categories.

1. Tag rules: first matching category wins.
2. Fallback for markets whose tags are uninformative (e.g. only 'all'): keyword rules on the question.
3. Thin categories are merged (see MERGE).
"""
import json
import re

TAG_RULES = [
    ("Sports", {"sports", "nfl", "nba", "mlb", "nhl", "soccer", "football", "tennis", "golf",
                "ufc", "mma", "f1", "formula-1", "olympics", "esports", "cricket", "boxing",
                "epl", "ncaa", "basketball", "baseball", "hockey", "world-cup", "fifa",
                "premier-league", "champions-league", "super-bowl", "march-madness", "chess"}),
    ("Crypto", {"crypto", "bitcoin", "ethereum", "solana", "defi", "nft", "memecoin", "xrp",
                "crypto-prices", "airdrops", "stablecoins", "sbf"}),
    ("Economy", {"economy", "fed", "fed-rates", "inflation", "business", "finance", "stocks",
                 "earnings", "ipo", "commodities", "oil", "jobs", "recession", "tariffs",
                 "interest-rates", "economic-policy", "big-tech", "tesla", "nvidia"}),
    ("Geopolitics", {"geopolitics", "world", "ukraine", "russia", "israel", "gaza",
                     "middle-east", "china", "iran", "war", "taiwan", "nato", "north-korea",
                     "eu", "europe", "invasion"}),
    ("Politics", {"politics", "elections", "us-politics", "trump", "potus", "presidential-election",
                  "congress", "senate", "house", "democrats", "republicans", "primaries", "biden",
                  "global-elections", "world-elections", "us-election", "governor", "supreme-court",
                  "government-shutdown", "kamala-harris", "kamala", "vance", "government",
                  "legal-issues", "court-cases", "crime", "arrest", "law-enforcement"}),
    ("Tech & Science", {"tech", "ai", "science", "space", "spacex", "openai", "climate",
                        "weather", "health", "covid", "technology", "chatgpt", "sam-altman",
                        "hacking", "cybersecurity"}),
    ("Culture", {"pop-culture", "culture", "movies", "music", "awards", "oscars", "entertainment",
                 "celebrities", "tv", "twitter", "elon-musk", "musk", "social-media", "youtube",
                 "tiktok", "mrbeast", "box-office", "taylor-swift"}),
]

KEYWORD_RULES = [
    ("Sports", r"\b(nfl|nba|mlb|nhl|ufc|super bowl|world cup|premier league|champions league|"
               r"stanley cup|world series|grand slam|wimbledon|olympics?|playoffs?|"
               r"championship|f1|grand prix|boxing|vs\.?)\b"),
    ("Crypto", r"\b(bitcoin|btc|ethereum|eth|solana|crypto|doge|dogecoin|xrp|coinbase|binance|"
               r"token|airdrop|stablecoin)\b"),
    ("Economy", r"\b(fed|fomc|rate cuts?|rate hikes?|inflation|cpi|gdp|recession|unemployment|"
                r"tariffs?|stocks?|s&p|nasdaq|dow|oil|gold|ipo|earnings|treasury|jobs report)\b"),
    ("Geopolitics", r"\b(ukraine|russia|putin|zelensky|israel|gaza|hamas|hezbollah|iran|china|"
                    r"taiwan|nato|ceasefire|invasion|war|north korea|venezuela)\b"),
    ("Politics", r"\b(trump|biden|harris|election|senate|congress|republican|democrat|primary|"
                 r"president|governor|speaker|shutdown|nominee|prime minister|supreme court|"
                 r"indicted|convicted|impeach\w*)\b"),
    ("Tech & Science", r"\b(openai|chatgpt|gpt-?\d|spacex|nasa|starship|artificial intelligence|ai|"
                       r"covid|vaccine|hurricane|earthquake|apple|google|microsoft|meta|nvidia)\b"),
    ("Culture", r"\b(oscars?|grammys?|emmys?|movie|box office|album|taylor swift|netflix|twitter|"
                r"tweets?|elon|musk|youtube|mrbeast|tiktok|celebrity|spotify)\b"),
]
_KEYWORDS = [(name, re.compile(pat, re.I)) for name, pat in KEYWORD_RULES]

# Categories with fewer than ~30 events are merged so per-category Brier scores mean something.
MERGE = {"Culture": "Culture & Tech", "Tech & Science": "Culture & Tech"}


def _from_tags(tag_slugs_json):
    slugs = set(json.loads(tag_slugs_json)) if tag_slugs_json else set()
    for name, keys in TAG_RULES:
        if slugs & keys:
            return name
    return None


def _from_question(question):
    q = question or ""
    for name, rx in _KEYWORDS:
        if rx.search(q):
            return name
    return None


def assign_category(tag_slugs_json, question=""):
    name = _from_tags(tag_slugs_json) or _from_question(question) or "Other"
    return MERGE.get(name, name)