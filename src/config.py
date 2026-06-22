FORMATIONS = {
    "4-3-3": [4, 3, 3],
    "4-4-2": [4, 4, 2],
    "4-2-3-1": [4, 2, 3, 1],
    "3-5-2": [3, 5, 2],
    "5-3-2": [5, 3, 2],
}

SAMPLE_FRAME_COUNT = 10
EXPECTED_OUTFIELD_PLAYERS = 10
MIN_TRACK_OBSERVATIONS = 2
DEFAULT_TRACK_FRAME_STRIDE = 5

TEAM_ABBREVIATIONS = {}
TEAM_NAME_ALIASES = {}


def register_team(canonical, abbreviations=(), aliases=()):
    TEAM_NAME_ALIASES[canonical.lower()] = canonical
    TEAM_NAME_ALIASES[canonical.replace("&", "and").lower()] = canonical
    TEAM_NAME_ALIASES[canonical.replace(".", "").lower()] = canonical

    for alias in aliases:
        TEAM_NAME_ALIASES[alias.lower()] = canonical

    for abbreviation in abbreviations:
        TEAM_ABBREVIATIONS[abbreviation.upper()] = canonical


# Premier League
register_team("Arsenal", ["ARS"], ["arsenal"])
register_team("Aston Villa", ["AVL"], ["aston villa", "villa"])
register_team("Bournemouth", ["BOU"], ["bournemouth"])
register_team("Brentford", ["BRE"], ["brentford"])
register_team(
    "Brighton & Hove Albion",
    ["BHA", "BRI"],
    ["brighton", "brighton & hove albion", "brighton and hove albion"],
)
register_team("Burnley", ["BUR"], ["burnley"])
register_team("Chelsea", ["CHE"], ["chelsea"])
register_team("Crystal Palace", ["CRY"], ["crystal palace", "palace"])
register_team("Everton", ["EVE"], ["everton"])
register_team("Fulham", ["FUL"], ["fulham"])
register_team("Ipswich Town", ["IPS"], ["ipswich", "ipswich town"])
register_team("Leeds United", ["LEE"], ["leeds", "leeds united", "leeds utd"])
register_team("Leicester City", ["LEI"], ["leicester", "leicester city"])
register_team("Liverpool", ["LIV"], ["liverpool", "liverpool fc"])
register_team("Luton Town", ["LUT"], ["luton", "luton town"])
register_team("Manchester City", ["MCI"], ["man city", "mancity", "manchester city"])
register_team(
    "Manchester United",
    ["MUN"],
    ["man united", "man utd", "manchester united", "united", "mufc"],
)
register_team("Newcastle United", ["NEW"], ["newcastle", "newcastle united"])
register_team("Nottingham Forest", ["NFO", "NOT"], ["nottingham forest", "forest"])
register_team("Sheffield United", ["SHU"], ["sheffield united", "sheffield utd", "blades"])
register_team("Southampton", ["SOU"], ["southampton", "saints"])
register_team("Sunderland", ["SUN"], ["sunderland", "black cats"])
register_team("Tottenham Hotspur", ["TOT"], ["tottenham", "spurs", "tottenham hotspur"])
register_team("West Bromwich Albion", ["WBA"], ["west brom", "west bromwich albion", "baggies"])
register_team("West Ham United", ["WHU"], ["west ham", "west ham united", "hammers"])
register_team("Wolverhampton Wanderers", ["WOL"], ["wolves", "wolverhampton", "wolverhampton wanderers"])

# La Liga
register_team("Alaves", ["ALA"], ["alaves", "deportivo alaves"])
register_team("Athletic Club", ["ATH"], ["athletic club", "athletic bilbao", "bilbao"])
register_team("Atletico Madrid", ["ATM", "ATL"], ["atletico madrid", "atleti", "atletico", "atlético madrid"])
register_team("Barcelona", ["BAR", "FCB"], ["barcelona", "barca", "fc barcelona"])
register_team("Real Betis", ["BET"], ["real betis", "betis"])
register_team("Cadiz", ["CAD"], ["cadiz", "cádiz", "cadiz cf"])
register_team("Celta Vigo", ["CEL"], ["celta", "celta vigo", "celta de vigo"])
register_team("Elche", ["ELC"], ["elche", "elche cf"])
register_team("Espanyol", ["ESP"], ["espanyol", "rcd espanyol"])
register_team("Getafe", ["GET"], ["getafe"])
register_team("Girona", ["GIR"], ["girona"])
register_team("Granada", ["GRA"], ["granada"])
register_team("Las Palmas", ["LPA", "LAS"], ["las palmas", "ud las palmas"])
register_team("Leganes", ["LEG"], ["leganes", "leganés"])
register_team("Mallorca", ["MLL"], ["mallorca", "real mallorca"])
register_team("Osasuna", ["OSA"], ["osasuna", "ca osasuna"])
register_team("Rayo Vallecano", ["RAY"], ["rayo", "rayo vallecano"])
register_team("Real Madrid", ["RMA", "RMD"], ["real madrid", "madrid", "los blancos"])
register_team("Real Sociedad", ["RSO"], ["real sociedad", "sociedad", "la real"])
register_team("Sevilla", ["SEV"], ["sevilla", "sevilla fc"])
register_team("Valencia", ["VAL"], ["valencia", "valencia cf"])
register_team("Villarreal", ["VIL", "VLL"], ["villarreal", "villarreal cf", "submarino amarillo"])

# Serie A
register_team("Atalanta", ["ATA"], ["atalanta", "atalanta bc"])
register_team("Bologna", ["BOL"], ["bologna"])
register_team("Cagliari", ["CAG"], ["cagliari"])
register_team("Como", ["COM"], ["como", "como 1907"])
register_team("Empoli", ["EMP"], ["empoli", "empoli fc"])
register_team("Fiorentina", ["FIO"], ["fiorentina", "la viola", "viola"])
register_team("Genoa", ["GEN"], ["genoa", "genoa cfc"])
register_team("Inter", ["INT"], ["inter", "inter milan", "internazionale"])
register_team("Juventus", ["JUV"], ["juventus", "juve"])
register_team("Lazio", ["LAZ"], ["lazio", "ss lazio"])
register_team("Lecce", ["LEC"], ["lecce", "us lecce"])
register_team("Milan", ["MIL"], ["milan", "ac milan", "rossoneri"])
register_team("Monza", ["MON"], ["monza", "ac monza"])
register_team("Napoli", ["NAP"], ["napoli", "ssc napoli"])
register_team("Roma", ["ROM"], ["roma", "as roma"])
register_team("Torino", ["TOR"], ["torino"])
register_team("Udinese", ["UDI"], ["udinese"])
register_team("Verona", ["VER"], ["verona", "hellas verona"])

# Bundesliga
register_team("Augsburg", ["AUG"], ["augsburg"])
register_team("Bayern Munich", ["BAY"], ["bayern", "bayern munich", "fc bayern"])
register_team("Bochum", ["BOC"], ["bochum", "vfl bochum"])
register_team("Borussia Dortmund", ["BVB", "DOR"], ["dortmund", "borussia dortmund", "bvb"])
register_team("Eintracht Frankfurt", ["FRA"], ["frankfurt", "eintracht frankfurt"])
register_team("Freiburg", ["SCF"], ["freiburg", "sc freiburg"])
register_team("Heidenheim", ["HDH"], ["heidenheim", "fc heidenheim"])
register_team("Hoffenheim", ["HOF"], ["hoffenheim", "tsg hoffenheim"])
register_team("Holstein Kiel", ["KIE"], ["holstein kiel", "kiel"])
register_team("Bayer Leverkusen", ["LEV", "B04"], ["leverkusen", "bayer leverkusen", "werkself"])
register_team("Mainz", ["M05"], ["mainz", "mainz 05"])
register_team("Borussia Monchengladbach", ["BMG"], ["monchengladbach", "mönchengladbach", "gladbach"])
register_team("RB Leipzig", ["RBL"], ["rb leipzig", "leipzig"])
register_team("St. Pauli", ["STP"], ["st pauli", "st. pauli", "fc st pauli"])
register_team("Stuttgart", ["STU"], ["stuttgart", "vfb stuttgart"])
register_team("Union Berlin", ["UNB"], ["union berlin", "union"])
register_team("Werder Bremen", ["SVW"], ["werder bremen", "bremen"])
register_team("Wolfsburg", ["WOB"], ["wolfsburg", "vfl wolfsburg"])

# Ligue 1
register_team("Angers", ["ANG"], ["angers", "angers sco"])
register_team("Auxerre", ["AUX"], ["auxerre", "aja"])
register_team("Brest", ["BST"], ["brest", "sb29"])
register_team("Le Havre", ["LHA"], ["le havre", "lehavre"])
register_team("Lens", ["LEN"], ["lens", "rc lens"])
register_team("Lille", ["LIL"], ["lille", "losc"])
register_team("Lyon", ["LYO"], ["lyon", "ol", "olympique lyonnais"])
register_team("Marseille", ["MAR"], ["marseille", "om", "olympique marseille"])
register_team("Monaco", ["ASM"], ["monaco", "as monaco"])
register_team("Montpellier", ["MTP"], ["montpellier", "mhsc"])
register_team("Nantes", ["NAN"], ["nantes", "fc nantes"])
register_team("Nice", ["NIC"], ["nice", "ogc nice"])
register_team(
    "Paris Saint-Germain",
    ["PSG", "PAR"],
    ["psg", "paris sg", "paris saint germain", "paris saint-germain"],
)
register_team("Reims", ["REI"], ["reims", "stade de reims"])
register_team("Rennes", ["REN"], ["rennes", "stade rennais"])
register_team("Saint-Etienne", ["STE"], ["saint etienne", "saint-étienne", "asse", "st etienne"])
register_team("Strasbourg", ["STR"], ["strasbourg", "rc strasbourg"])
register_team("Toulouse", ["TOU"], ["toulouse", "tfc"])