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

TEAM_ABBREVIATIONS = {}   # abbr (str) → list[str] of canonical names
TEAM_NAME_ALIASES = {}     # alias (str) → canonical name (str)
TEAM_TYPE = {}             # canonical name (str) → "club" | "international"


def register_team(canonical, abbreviations=(), aliases=(), team_type="club"):
    """Register a team with its abbreviations, aliases, and type."""
    TEAM_TYPE[canonical] = team_type
    TEAM_NAME_ALIASES[canonical.lower()] = canonical
    TEAM_NAME_ALIASES[canonical.replace("&", "and").lower()] = canonical
    TEAM_NAME_ALIASES[canonical.replace(".", "").lower()] = canonical
    for alias in aliases:
        TEAM_NAME_ALIASES[alias.lower()] = canonical
    for abbreviation in abbreviations:
        key = abbreviation.upper()
        if key not in TEAM_ABBREVIATIONS:
            TEAM_ABBREVIATIONS[key] = []
        if canonical not in TEAM_ABBREVIATIONS[key]:
            TEAM_ABBREVIATIONS[key].append(canonical)


# ============================================================
# Premier League
# ============================================================
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
register_team("Nottingham Forest", ["NFO"], ["nottingham forest", "forest", "nffc"])
register_team("Sheffield United", ["SHU"], ["sheffield united", "sheffield utd", "blades"])
register_team("Southampton", ["SOU"], ["southampton", "saints"])
register_team("Tottenham Hotspur", ["TOT"], ["tottenham", "spurs", "tottenham hotspur"])
register_team("West Ham United", ["WHU"], ["west ham", "west ham united", "hammers"])
register_team(
    "Wolverhampton Wanderers",
    ["WOL"],
    ["wolves", "wolverhampton", "wolverhampton wanderers"],
)

# ============================================================
# La Liga
# ============================================================
register_team("Alaves", ["ALA"], ["alaves", "deportivo alaves"])
register_team("Athletic Club", ["ATH"], ["athletic club", "athletic bilbao", "bilbao"])
register_team(
    "Atletico Madrid",
    ["ATM", "ATL"],
    ["atletico madrid", "atleti", "atletico", "atlético madrid"],
)
register_team("Barcelona", ["BAR", "FCB"], ["barcelona", "barca", "fc barcelona", "barça"])
register_team("Betis", ["BET"], ["betis", "real betis"])
register_team("Cadiz", ["CAD"], ["cadiz", "cadiz cf", "cádiz"])
register_team("Celta Vigo", ["CEL"], ["celta vigo", "celta", "rc celta"])
register_team("Getafe", ["GET"], ["getafe", "getafe cf"])
register_team("Girona", ["GIR"], ["girona", "girona fc"])
register_team("Granada", ["GRA"], ["granada", "granada cf"])
register_team("Las Palmas", ["LPA"], ["las palmas", "ud las palmas"])
register_team("Mallorca", ["MLL"], ["mallorca", "rcd mallorca"])
register_team("Osasuna", ["OSA"], ["osasuna", "ca osasuna"])
register_team("Rayo Vallecano", ["RAY"], ["rayo vallecano", "rayo"])
register_team("Real Madrid", ["RMA", "RMD"], ["real madrid", "madrid", "rmcf"])
register_team("Real Sociedad", ["RSO"], ["real sociedad", "sociedad", "la real"])
register_team("Sevilla", ["SEV"], ["sevilla", "sevilla fc"])
register_team("Valencia", ["VAL"], ["valencia", "valencia cf"])
register_team(
    "Villarreal", ["VIL", "VLL"], ["villarreal", "villarreal cf", "submarino amarillo"]
)

# ============================================================
# Serie A
# ============================================================
register_team("Atalanta", ["ATA"], ["atalanta", "atalanta bc"])
register_team("Bologna", ["BOL"], ["bologna"])
register_team("Cagliari", ["CAG"], ["cagliari"])
register_team("Como", ["COM"], ["como", "como 1907"])
register_team("Empoli", ["EMP"], ["empoli"])
register_team("Fiorentina", ["FIO"], ["fiorentina", "acf fiorentina"])
register_team("Genoa", ["GEN"], ["genoa", "genoa cfc"])
register_team("Hellas Verona", ["VER"], ["verona", "hellas verona"])
register_team("Inter Milan", ["INT"], ["inter", "inter milan", "internazionale"])
register_team("Juventus", ["JUV"], ["juventus", "juve"])
register_team("Lazio", ["LAZ"], ["lazio", "ss lazio"])
register_team("Lecce", ["LEC"], ["lecce", "us lecce"])
register_team("AC Milan", ["MIL"], ["ac milan", "milan"])
register_team("Monza", ["MON"], ["monza"])
register_team("Napoli", ["NAP"], ["napoli", "ssc napoli"])
register_team("Roma", ["ROM"], ["roma", "as roma"])
register_team("Salernitana", ["SAL"], ["salernitana"])
register_team("Sassuolo", ["SAS"], ["sassuolo"])
register_team("Torino", ["TOR"], ["torino", "torino fc"])
register_team("Udinese", ["UDI"], ["udinese"])
register_team("Venezia", ["VNZ"], ["venezia"])

# ============================================================
# Bundesliga
# ============================================================
register_team("Augsburg", ["AUG"], ["augsburg", "fc augsburg"])
register_team("Bayer Leverkusen", ["LEV", "B04"], ["leverkusen", "bayer leverkusen"])
register_team("Bayern Munich", ["BAY", "FCB"], ["bayern", "bayern munich", "bayern münchen", "fc bayern"])
register_team("Borussia Dortmund", ["BVB", "DOR"], ["dortmund", "borussia dortmund", "bvb"])
register_team(
    "Borussia Monchengladbach",
    ["BMG"],
    ["gladbach", "monchengladbach", "borussia monchengladbach", "mönchengladbach"],
)
register_team("Darmstadt", ["DAR"], ["darmstadt", "sv darmstadt"])
register_team("Eintracht Frankfurt", ["SGE", "FRA"], ["frankfurt", "eintracht frankfurt"])
register_team("Freiburg", ["FRE"], ["freiburg", "sc freiburg"])
register_team("Heidenheim", ["HEI"], ["heidenheim", "fc heidenheim"])
register_team("Hoffenheim", ["HOF", "TSG"], ["hoffenheim", "tsg hoffenheim"])
register_team("Koln", ["KOE"], ["koln", "köln", "fc köln", "fc koln"])
register_team("Mainz", ["MAI", "M05"], ["mainz", "mainz 05"])
register_team("RB Leipzig", ["RBL", "LEP"], ["leipzig", "rb leipzig"])
register_team("Stuttgart", ["STU", "VFB"], ["stuttgart", "vfb stuttgart"])
register_team("Union Berlin", ["UNB", "FCU"], ["union berlin", "fc union berlin"])
register_team("Werder Bremen", ["SVW", "BRE"], ["werder", "werder bremen"])
register_team("Wolfsburg", ["WOB"], ["wolfsburg", "vfl wolfsburg"])

# ============================================================
# Ligue 1
# ============================================================
register_team("Angers", ["ANG"], ["angers", "sco angers"])
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
register_team(
    "Saint-Etienne",
    ["STE"],
    ["saint etienne", "saint-étienne", "asse", "st etienne"],
)
register_team("Strasbourg", ["STR"], ["strasbourg", "rc strasbourg"])
register_team("Toulouse", ["TOU"], ["toulouse", "tfc"])

# ============================================================
# International Teams (FIFA) — AFC (Asia)
# ============================================================
register_team("Afghanistan", ["AFG"], ["afghanistan"], team_type="international")
register_team("Australia", ["AUS"], ["australia", "socceroos"], team_type="international")
register_team("Bahrain", ["BHR"], ["bahrain"], team_type="international")
register_team("Bangladesh", ["BAN"], ["bangladesh"], team_type="international")
register_team("China PR", ["CHN"], ["china", "china pr"], team_type="international")
register_team("India", ["IND"], ["india"], team_type="international")
register_team("Indonesia", ["IDN"], ["indonesia"], team_type="international")
register_team("Iran", ["IRN"], ["iran", "team melli"], team_type="international")
register_team("Iraq", ["IRQ"], ["iraq"], team_type="international")
register_team("Japan", ["JPN"], ["japan", "samurai blue"], team_type="international")
register_team("Jordan", ["JOR"], ["jordan"], team_type="international")
register_team("Kuwait", ["KUW"], ["kuwait"], team_type="international")
register_team("Kyrgyzstan", ["KGZ"], ["kyrgyzstan"], team_type="international")
register_team("Lebanon", ["LBN"], ["lebanon"], team_type="international")
register_team("Malaysia", ["MAS"], ["malaysia"], team_type="international")
register_team("Myanmar", ["MYA"], ["myanmar"], team_type="international")
register_team("North Korea", ["PRK"], ["north korea", "dpr korea"], team_type="international")
register_team("Oman", ["OMA"], ["oman"], team_type="international")
register_team("Palestine", ["PLE"], ["palestine"], team_type="international")
register_team("Philippines", ["PHI"], ["philippines", "azkals"], team_type="international")
register_team("Qatar", ["QAT"], ["qatar"], team_type="international")
register_team(
    "Saudi Arabia", ["KSA"], ["saudi arabia", "saudi", "the green falcons"], team_type="international"
)
register_team("Singapore", ["SIN"], ["singapore"], team_type="international")
register_team(
    "South Korea", ["KOR"], ["south korea", "korea republic", "korea"], team_type="international"
)
register_team("Syria", ["SYR"], ["syria"], team_type="international")
register_team("Tajikistan", ["TJK"], ["tajikistan"], team_type="international")
register_team("Thailand", ["THA"], ["thailand", "war elephants"], team_type="international")
register_team("Turkmenistan", ["TKM"], ["turkmenistan"], team_type="international")
register_team("UAE", ["UAE"], ["uae", "united arab emirates"], team_type="international")
register_team("Uzbekistan", ["UZB"], ["uzbekistan", "white wolves"], team_type="international")
register_team("Vietnam", ["VIE"], ["vietnam"], team_type="international")
register_team("Yemen", ["YEM"], ["yemen"], team_type="international")

# ============================================================
# International Teams (FIFA) — CAF (Africa)
# ============================================================
register_team("Algeria", ["ALG"], ["algeria", "les fennecs"], team_type="international")
register_team("Angola", ["ANG"], ["angola"], team_type="international")
register_team("Benin", ["BEN"], ["benin"], team_type="international")
register_team("Burkina Faso", ["BFA"], ["burkina faso", "les étalons"], team_type="international")
register_team("Cameroon", ["CMR"], ["cameroon", "indomitable lions"], team_type="international")
register_team("Cape Verde", ["CPV"], ["cape verde", "cabo verde"], team_type="international")
register_team("Central African Republic", ["CTA"], ["central african republic"], team_type="international")
register_team("Chad", ["CHA"], ["chad"], team_type="international")
register_team("Comoros", ["COM"], ["comoros"], team_type="international")
register_team("Congo", ["CGO"], ["congo", "congo brazzaville"], team_type="international")
register_team(
    "Congo DR", ["COD"], ["congo dr", "dr congo", "democratic republic of congo"], team_type="international"
)
register_team(
    "Cote d'Ivoire",
    ["CIV"],
    ["ivory coast", "cote d'ivoire", "côte d'ivoire", "les elephants"],
    team_type="international",
)
register_team("Egypt", ["EGY"], ["egypt", "the pharaohs"], team_type="international")
register_team("Equatorial Guinea", ["EQG"], ["equatorial guinea"], team_type="international")
register_team("Eritrea", ["ERI"], ["eritrea"], team_type="international")
register_team("Eswatini", ["SWZ"], ["eswatini", "swaziland"], team_type="international")
register_team("Ethiopia", ["ETH"], ["ethiopia"], team_type="international")
register_team("Gabon", ["GAB"], ["gabon", "les panthères"], team_type="international")
register_team("Gambia", ["GAM"], ["gambia"], team_type="international")
register_team("Ghana", ["GHA"], ["ghana", "black stars"], team_type="international")
register_team("Guinea", ["GUI"], ["guinea"], team_type="international")
register_team("Guinea-Bissau", ["GNB"], ["guinea-bissau", "guinea bissau"], team_type="international")
register_team("Kenya", ["KEN"], ["kenya", "harambee stars"], team_type="international")
register_team("Lesotho", ["LES"], ["lesotho"], team_type="international")
register_team("Liberia", ["LBR"], ["liberia"], team_type="international")
register_team("Libya", ["LBY"], ["libya"], team_type="international")
register_team("Madagascar", ["MAD"], ["madagascar"], team_type="international")
register_team("Malawi", ["MWI"], ["malawi", "flames"], team_type="international")
register_team("Mali", ["MLI"], ["mali", "les aigles"], team_type="international")
register_team("Mauritania", ["MTN"], ["mauritania"], team_type="international")
register_team("Mauritius", ["MRI"], ["mauritius"], team_type="international")
register_team("Morocco", ["MAR"], ["morocco", "atlas lions"], team_type="international")
register_team("Mozambique", ["MOZ"], ["mozambique"], team_type="international")
register_team("Namibia", ["NAM"], ["namibia", "brave warriors"], team_type="international")
register_team("Niger", ["NIG"], ["niger"], team_type="international")
register_team("Nigeria", ["NGA"], ["nigeria", "super eagles"], team_type="international")
register_team("Rwanda", ["RWA"], ["rwanda"], team_type="international")
register_team("Senegal", ["SEN"], ["senegal", "lions of teranga"], team_type="international")
register_team("Sierra Leone", ["SLE"], ["sierra leone"], team_type="international")
register_team("Somalia", ["SOM"], ["somalia"], team_type="international")
register_team("South Africa", ["RSA"], ["south africa", "bafana bafana"], team_type="international")
register_team("South Sudan", ["SSD"], ["south sudan"], team_type="international")
register_team("Sudan", ["SDN"], ["sudan"], team_type="international")
register_team("Tanzania", ["TAN"], ["tanzania", "taifa stars"], team_type="international")
register_team("Togo", ["TOG"], ["togo"], team_type="international")
register_team("Tunisia", ["TUN"], ["tunisia", "eagles of carthage"], team_type="international")
register_team("Uganda", ["UGA"], ["uganda", "the cranes"], team_type="international")
register_team("Zambia", ["ZAM"], ["zambia", "chipolopolo"], team_type="international")
register_team("Zimbabwe", ["ZIM"], ["zimbabwe", "the warriors"], team_type="international")

# ============================================================
# International Teams (FIFA) — CONCACAF
# ============================================================
register_team("Canada", ["CAN"], ["canada", "les rouges", "canmnt"], team_type="international")
register_team("Costa Rica", ["CRC"], ["costa rica", "los ticos"], team_type="international")
register_team("Cuba", ["CUB"], ["cuba"], team_type="international")
register_team("Curacao", ["CUW"], ["curacao", "curaçao"], team_type="international")
register_team("El Salvador", ["SLV"], ["el salvador", "la selecta"], team_type="international")
register_team("Guatemala", ["GUA"], ["guatemala"], team_type="international")
register_team("Haiti", ["HAI"], ["haiti", "les grenadiers"], team_type="international")
register_team("Honduras", ["HON"], ["honduras", "los catrachos"], team_type="international")
register_team("Jamaica", ["JAM"], ["jamaica", "reggae boyz"], team_type="international")
register_team("Mexico", ["MEX"], ["mexico", "el tri"], team_type="international")
register_team("Nicaragua", ["NCA"], ["nicaragua"], team_type="international")
register_team("Panama", ["PAN"], ["panama", "los canaleros"], team_type="international")
register_team(
    "Trinidad and Tobago",
    ["TRI"],
    ["trinidad and tobago", "trinidad", "soca warriors"],
    team_type="international",
)
register_team(
    "United States", ["USA"], ["usa", "united states", "usmnt", "us"], team_type="international"
)

# ============================================================
# International Teams (FIFA) — CONMEBOL
# ============================================================
register_team("Argentina", ["ARG"], ["argentina", "la albiceleste"], team_type="international")
register_team("Bolivia", ["BOL"], ["bolivia", "la verde"], team_type="international")
register_team("Brazil", ["BRA"], ["brazil", "brasil", "selecao", "seleção"], team_type="international")
register_team("Chile", ["CHI"], ["chile", "la roja"], team_type="international")
register_team("Colombia", ["COL"], ["colombia", "los cafeteros"], team_type="international")
register_team("Ecuador", ["ECU"], ["ecuador", "la tri"], team_type="international")
register_team("Paraguay", ["PAR"], ["paraguay", "la albirroja"], team_type="international")
register_team("Peru", ["PER"], ["peru", "la blanquirroja"], team_type="international")
register_team("Uruguay", ["URU"], ["uruguay", "la celeste"], team_type="international")
register_team("Venezuela", ["VEN"], ["venezuela", "la vinotinto"], team_type="international")

# ============================================================
# International Teams (FIFA) — UEFA (Europe)
# ============================================================
register_team("Albania", ["ALB"], ["albania"], team_type="international")
register_team("Andorra", ["AND"], ["andorra"], team_type="international")
register_team("Armenia", ["ARM"], ["armenia"], team_type="international")
register_team("Austria", ["AUT"], ["austria", "das team"], team_type="international")
register_team("Azerbaijan", ["AZE"], ["azerbaijan"], team_type="international")
register_team("Belarus", ["BLR"], ["belarus"], team_type="international")
register_team(
    "Belgium", ["BEL"], ["belgium", "red devils", "rode duivels"], team_type="international"
)
register_team(
    "Bosnia and Herzegovina",
    ["BIH"],
    ["bosnia", "bosnia and herzegovina", "bosnia & herzegovina"],
    team_type="international",
)
register_team("Bulgaria", ["BUL"], ["bulgaria"], team_type="international")
register_team("Croatia", ["CRO"], ["croatia", "vatreni", "the blazers"], team_type="international")
register_team("Cyprus", ["CYP"], ["cyprus"], team_type="international")
register_team("Czech Republic", ["CZE"], ["czech republic", "czechia"], team_type="international")
register_team("Denmark", ["DEN"], ["denmark", "danish dynamite"], team_type="international")
register_team("England", ["ENG"], ["england", "three lions"], team_type="international")
register_team("Estonia", ["EST"], ["estonia"], team_type="international")
register_team("Faroe Islands", ["FRO"], ["faroe islands"], team_type="international")
register_team("Finland", ["FIN"], ["finland", "huuhkajat"], team_type="international")
register_team("France", ["FRA"], ["france", "les bleus"], team_type="international")
register_team("Georgia", ["GEO"], ["georgia"], team_type="international")
register_team("Germany", ["GER"], ["germany", "die mannschaft"], team_type="international")
register_team("Gibraltar", ["GIB"], ["gibraltar"], team_type="international")
register_team("Greece", ["GRE"], ["greece", "ethniki"], team_type="international")
register_team("Hungary", ["HUN"], ["hungary", "magyarok"], team_type="international")
register_team("Iceland", ["ISL"], ["iceland", "strákarnir okkar"], team_type="international")
register_team(
    "Ireland", ["IRL"], ["ireland", "republic of ireland", "boys in green"], team_type="international"
)
register_team("Israel", ["ISR"], ["israel"], team_type="international")
register_team("Italy", ["ITA"], ["italy", "gli azzurri", "azzurri"], team_type="international")
register_team("Kazakhstan", ["KAZ"], ["kazakhstan"], team_type="international")
register_team("Kosovo", ["KVX"], ["kosovo"], team_type="international")
register_team("Latvia", ["LVA"], ["latvia"], team_type="international")
register_team("Liechtenstein", ["LIE"], ["liechtenstein"], team_type="international")
register_team("Lithuania", ["LTU"], ["lithuania"], team_type="international")
register_team("Luxembourg", ["LUX"], ["luxembourg"], team_type="international")
register_team("Malta", ["MLT"], ["malta"], team_type="international")
register_team("Moldova", ["MDA"], ["moldova"], team_type="international")
register_team("Montenegro", ["MNE"], ["montenegro"], team_type="international")
register_team(
    "Netherlands", ["NED"], ["netherlands", "holland", "oranje", "dutch"], team_type="international"
)
register_team("North Macedonia", ["MKD"], ["north macedonia", "macedonia"], team_type="international")
register_team("Northern Ireland", ["NIR"], ["northern ireland"], team_type="international")
register_team("Norway", ["NOR"], ["norway"], team_type="international")
register_team("Poland", ["POL"], ["poland", "bialo-czerwoni"], team_type="international")
register_team(
    "Portugal", ["POR"], ["portugal", "selecao das quinas", "a seleção"], team_type="international"
)
register_team("Romania", ["ROU"], ["romania", "tricolorii"], team_type="international")
register_team("Russia", ["RUS"], ["russia"], team_type="international")
register_team("San Marino", ["SMR"], ["san marino"], team_type="international")
register_team("Scotland", ["SCO"], ["scotland", "tartan army"], team_type="international")
register_team("Serbia", ["SRB"], ["serbia", "orlovi"], team_type="international")
register_team("Slovakia", ["SVK"], ["slovakia"], team_type="international")
register_team("Slovenia", ["SVN"], ["slovenia"], team_type="international")
register_team("Spain", ["ESP"], ["spain", "la roja", "la furia roja"], team_type="international")
register_team("Sweden", ["SWE"], ["sweden", "blågult"], team_type="international")
register_team(
    "Switzerland", ["SUI"], ["switzerland", "nati", "schweiz", "suisse"], team_type="international"
)
register_team(
    "Turkey", ["TUR"], ["turkey", "türkiye", "ay-yildizlilar"], team_type="international"
)
register_team("Ukraine", ["UKR"], ["ukraine", "zbirna"], team_type="international")
register_team("Wales", ["WAL"], ["wales", "y dreigiau"], team_type="international")

# ============================================================
# International Teams (FIFA) — OFC (Oceania)
# ============================================================
register_team("Fiji", ["FIJ"], ["fiji"], team_type="international")
register_team("New Caledonia", ["NCL"], ["new caledonia"], team_type="international")
register_team("New Zealand", ["NZL"], ["new zealand", "all whites"], team_type="international")
register_team("Papua New Guinea", ["PNG"], ["papua new guinea"], team_type="international")
register_team("Samoa", ["SAM"], ["samoa"], team_type="international")
register_team("Solomon Islands", ["SOL"], ["solomon islands"], team_type="international")
register_team("Tahiti", ["TAH"], ["tahiti"], team_type="international")
register_team("Tonga", ["TGA"], ["tonga"], team_type="international")
register_team("Vanuatu", ["VAN"], ["vanuatu"], team_type="international")