import math
import os
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import random
from urllib.parse import urljoin
from prediction_pipeline import PredictionPipeline

@st.cache_resource
def get_prediction_pipeline():
    return PredictionPipeline()

# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(page_title="Track Pulse France", page_icon="Gemini_Generated_Image_py9xxopy9xxopy9x.png", layout="wide")

# Codes d'épreuves (Basés sur ton URL : 106 = 60m)
FFA_EVENT_IDS = {
    "60m (Salle)": "106",
    "100m": "110",
    "200m": "120", 
    "400m": "140"
}

# Minima (Exemple ajusté pour la salle et plein air)
MINIMA_CF = {
    "60m (Salle)": {"M": 6.85, "F": 7.60}, # Exemples
    "100m": {"M": 10.60, "F": 11.90},
    "200m": {"M": 21.50, "F": 24.20}
}

# ==========================================
# 2. MOTEUR DE SCRAPING ROBUSTE
# ==========================================

class AthleteIndex:
    """
    Gère un index local des athlètes pour la recherche rapide.
    """
    INDEX_FILE = "athletes_index.csv"
    
    def __init__(self):
        self.index = {} # Name -> URL
        self.load_index()
        
    def load_index(self):
        if os.path.exists(self.INDEX_FILE):
            try:
                df = pd.read_csv(self.INDEX_FILE)
                if 'Athlète' in df.columns and 'Lien' in df.columns:
                    # Create dictionary for fast lookup
                    self.index = pd.Series(df.Lien.values, index=df.Athlète).to_dict()
            except Exception as e:
                print(f"Erreur chargement index: {e}")
                self.index = {}
        else:
            self.index = {}

    def search(self, name):
        """Retourne l'URL pour un nom donné"""
        return self.index.get(name)

class ScoringSystem:
    # Coefficients Table Hongroise (IAAF 2017 / WMA 2023)
    # Format: [A, D, B, C]
    # Points = floor( A * (Perf + B)^2 / 10^D + C )
    HUNGARIAN_COEFFS = {
        'M': {
            '60m':      [  686,   5,   -1070,       0], # 60m-M-i
            '100m':     [ 2463,   6,   -1700,       0],
            '200m':     [   508,   6,   -3550,       0],
            '400m':     [  1021,   7,   -7900,       0],
            '800m':     [   198,   7,  -18200,       0],
            '1000m':    [  1074,   8,  -24000,       0],
            '1500m':    [  4066,   9,  -38500,       0],
            '110mH':    [   766,   6,   -2580,       0], # 110mh-M
            '400mH':    [   546,   7,   -9550,       0], # 400mh-M
            'Hauteur':  [  3229,   6,    1153.4, -5000], # hauteur-M
            'Perche':   [  3042,   7,    3939,   -5000], # perche-M
            'Longueur': [  1929,   7,    4841,   -5000], # longueur-M
            'Triple':   [  4611,   8,    9863,   -5000], # triple-M
            'Poids':    [ 42172,  10,   68770,  -20000], # poids-M
            'Disque':   [  4007,  10,  223260,  -20000], # disque-M
            'Marteau':  [ 28038,  11,  266940,  -20000], # marteau-M
            'Javelot':  [ 23974,  11,  288680,  -20000], # javelot-M
            'Decathlon':[  9774,  10,   71173,   -5000], # decathlon-M
            'Heptathlon':[ 1752,   9,   53175,   -5000], # heptathlon-M-i
        },
        'F': {
            '60m':      [   249,   5,   -1400,       0], # 60m-F-i
            '100m':     [   992,   6,   -2200,       0],
            '200m':     [  2242,   7,   -4550,       0],
            '400m':     [   335,   7,  -11000,       0],
            '800m':     [   688,   8,  -25000,       0],
            '1000m':    [   382,   8,  -33000,       0],
            '1500m':    [   134,   8,  -54000,       0],
            '100mH':    [   398,   6,   -3000,       0], # 100mh-F
            '400mH':    [208567,  10,  -13000,       0], # 400mh-F
            'Hauteur':  [  3934,   6,    1057.4, -5000], # hauteur-F
            'Perche':   [  3953,   7,    3483,   -5000], # perche-F
            'Longueur': [  1966,   7,    4924,   -5000], # longueur-F
            'Triple':   [  4282,   8,   10553,   -5000], # triple-F
            'Poids':    [   462,   8,   65753,  -20000], # poids-F
            'Disque':   [ 40277,  11,  222730,  -20000], # disque-F
            'Marteau':  [ 30965,  11,  254000,  -20000], # marteau-F
            'Javelot':  [  4073,  10,  221490,  -20000], # javelot-F
            'Heptathlon':[ 1581,   9,   55990,   -5000], # heptathlon-F
            'Pentathlon':[29445,  10,   41033,   -5000], # pentathlon-F-i
        }
    }

    # Seuils de niveau FFA (Barèmes)
    # Format: [(Niveau, Perf_Min), ...] trié par niveau décroissant de difficulté
    # Note: Pour les courses, Perf_Min est un temps max. Pour les sauts/lancers, c'est une distance min.
    FFA_LEVELS = {
        'M': {
            '60m':  [('IA', 6.56), ('IB', 6.66), ('N1', 6.74), ('N2', 6.84), ('N3', 6.94), ('N4', 7.14), ('IR1', 7.24), ('IR2', 7.34), ('IR3', 7.44), ('IR4', 7.54)],
            '100m': [('IA', 10.10), ('IB', 10.20), ('N1', 10.34), ('N2', 10.64), ('N3', 10.84), ('N4', 10.94), ('IR1', 11.14), ('IR2', 11.34), ('IR3', 11.44), ('IR4', 11.54)],
            '200m': [('IA', 20.44), ('IB', 20.64), ('N1', 20.94), ('N2', 21.44), ('N3', 21.84), ('N4', 22.24), ('IR1', 22.54), ('IR2', 22.84), ('IR3', 23.14), ('IR4', 23.44)],
            '400m': [('IA', 45.24), ('IB', 45.84), ('N1', 46.64), ('N2', 47.64), ('N3', 48.64), ('N4', 49.64), ('IR1', 50.44), ('IR2', 51.44), ('IR3', 52.44), ('IR4', 53.44)],
        },
        'F': {
            '60m':  [('IA', 7.24), ('IB', 7.34), ('N1', 7.44), ('N2', 7.64), ('N3', 7.84), ('N4', 7.94), ('IR1', 8.04), ('IR2', 8.14), ('IR3', 8.24), ('IR4', 8.34)],
            '100m': [('IA', 11.20), ('IB', 11.40), ('N1', 11.70), ('N2', 12.00), ('N3', 12.30), ('N4', 12.60), ('IR1', 12.90), ('IR2', 13.20), ('IR3', 13.50), ('IR4', 13.80)],
            '200m': [('IA', 22.90), ('IB', 23.30), ('N1', 24.00), ('N2', 24.60), ('N3', 25.40), ('N4', 26.00), ('IR1', 26.60), ('IR2', 27.20), ('IR3', 27.80), ('IR4', 28.40)],
            '400m': [('IA', 51.80), ('IB', 52.80), ('N1', 54.50), ('N2', 56.00), ('N3', 57.50), ('N4', 59.00), ('IR1', 60.50), ('IR2', 62.00), ('IR3', 63.50), ('IR4', 65.00)],
        }
    }

    @staticmethod
    def get_points(event, perf, gender):
        """
        Calcule les points Hongrois pour une performance donnée.
        """
        if not perf or not event: return ""
        
        coeffs = ScoringSystem.HUNGARIAN_COEFFS.get(gender, {}).get(event)
        if not coeffs:
            # Tentative de mapping approximatif pour les noms d'épreuves
            normalized = ScoringSystem.normalize_event_name_for_scoring(event)
            coeffs = ScoringSystem.HUNGARIAN_COEFFS.get(gender, {}).get(normalized)
            
        if not coeffs: return ""

        A, D, B, C = coeffs
        
        try:
            val = float(perf)
            
            # Conversion en unité de base (centièmes ou centimètres)
            is_course = event in ['60m', '100m', '200m', '400m', '110mH', '100mH', '400mH', '800m', '1000m', '1500m']
            is_saut = event in ['Hauteur', 'Perche', 'Longueur', 'Triple']
            is_lancer = event in ['Poids', 'Disque', 'Marteau', 'Javelot']
            
            if is_course:
                val_calc = val * 100
            elif is_saut:
                val_calc = val * 100
            elif is_lancer:
                val_calc = val * 100
            else:
                val_calc = val

            term = (val_calc + B)
            
            if is_course and B < 0 and (val_calc + B) >= 0:
                return 0
                
            points = math.floor( (A * (term ** 2)) / (10 ** D) + C )
            return max(0, points)
            
        except Exception as e:
            print(f"Erreur calcul points: {e}")
            return ""

    @staticmethod
    def normalize_event_name_for_scoring(event_name):
        """
        Normalise les noms d'épreuves pour correspondre aux clés du dictionnaire coeffs.
        """
        name = event_name.upper()
        if "60M" in name and "HAIES" not in name: return "60m"
        if "100M" in name and "HAIES" not in name: return "100m"
        if "200M" in name and "HAIES" not in name: return "200m"
        if "400M" in name and "HAIES" not in name: return "400m"
        if "800M" in name: return "800m"
        if "1000M" in name: return "1000m"
        if "1500M" in name: return "1500m"
        if "110M" in name and "HAIES" in name: return "110mH"
        if "100M" in name and "HAIES" in name: return "100mH"
        if "400M" in name and "HAIES" in name: return "400mH"
        if "HAUTEUR" in name: return "Hauteur"
        if "PERCHE" in name: return "Perche"
        if "LONGUEUR" in name: return "Longueur"
        if "TRIPLE" in name: return "Triple"
        if "POIDS" in name: return "Poids"
        if "DISQUE" in name: return "Disque"
        if "MARTEAU" in name: return "Marteau"
        if "JAVELOT" in name: return "Javelot"
        if "DECATHLON" in name: return "Decathlon"
        if "HEPTATHLON" in name: return "Heptathlon"
        if "PENTATHLON" in name: return "Pentathlon"
        return None

    @staticmethod
    def get_level(event, perf, gender):
        """
        Détermine le niveau FFA (IA, IB, N1...) pour une perf donnée.
        Uniquement pour les sprints pour l'instant.
        """
        if not perf or not event: return ""
        
        # On utilise le nom normalisé pour chercher dans FFA_LEVELS
        # (qui pour l'instant ne contient que du sprint)
        normalized = ScoringSystem.normalize_event_name_for_scoring(event)
        levels = ScoringSystem.FFA_LEVELS.get(gender, {}).get(normalized)
        
        if not levels: return ""
        
        try:
            val = float(perf)
            for lvl, threshold in levels:
                if val <= threshold:
                    return lvl
            return "D1" # Ou autre niveau par défaut si moins bon que IR4
        except:
            return ""

class FFAScraper:
    def __init__(self):
        self.base_url = "https://www.athle.fr/bases/liste.aspx"
        self.session = requests.Session()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
        ]
        self.headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://www.athle.fr/',
            'Connection': 'keep-alive'
        }

    def _make_request(self, url, params=None):
        """
        Effectue une requête robuste avec retries et rotation d'User-Agent
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Délai aléatoire pour éviter le blocage (0.5s à 2s)
                time.sleep(random.uniform(0.5, 2.0))
                
                # Rotation User-Agent
                current_headers = self.headers.copy()
                current_headers['User-Agent'] = random.choice(self.user_agents)
                
                # Utilisation de la session persistante
                response = self.session.get(url, params=params, headers=current_headers, timeout=30)
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                # Exponential backoff : 1s, 2s, 4s...
                wait_time = 2 ** attempt
                print(f"Erreur connexion ({e}), nouvel essai dans {wait_time}s...")
                time.sleep(wait_time)
        return None

    def clean_performance(self, perf_str):
        """
        Analyse une performance pour extraire le chrono, le vent, le type de chrono et les infos (RP, SB).
        Retourne un dictionnaire complet.
        """
        if not perf_str: return None
        
        # 1. Nettoyage de base
        clean = perf_str.replace("''", ".").replace("'", ".").strip()
        
        # Gestion du format saut/lancer "7m50" -> "7.50"
        if "m" in clean and not "Vent" in clean:
            clean = clean.replace("m", ".")
        
        # 2. Extraction du chrono
        match_chrono = re.search(r"(\d+\.\d+)", clean)
        if not match_chrono:
            return None
            
        chrono_str = match_chrono.group(1)
        chrono_val = float(chrono_str)
        
        # 3. Détection chrono manuel
        is_manual = False
        if "." in chrono_str:
            decimals = chrono_str.split(".")[1]
            if len(decimals) == 1:
                is_manual = True
        
        # 4. Extraction du vent
        wind_val = None
        wind_str = ""
        match_wind = re.search(r"\(([+-]?\d+\.\d+)\)", clean)
        if match_wind:
            wind_str = match_wind.group(1)
            try:
                wind_val = float(wind_str)
            except:
                pass
        
        # 5. Extraction des badges (RP, SB, etc.)
        # On cherche tout ce qui est entre parenthèses mais qui n'est pas le vent
        badges = []
        parts = re.findall(r"\((.*?)\)", clean)
        for p in parts:
            # Si ce n'est pas un chiffre (vent), c'est une info
            if not re.match(r"^[+-]?\d+\.\d+$", p) and p != "NC":
                badges.append(p)
        
        badge_str = " ".join(badges)
                
        return {
            'chrono': chrono_val,
            'wind': wind_val,
            'wind_str': wind_str,
            'is_manual': is_manual,
            'badge': badge_str
        }

    def normalize_event(self, event_name):
        """
        Normalise le nom de l'épreuve pour le sprint.
        Ex: "100m / SEM" -> "100m"
        Retourne None si ce n'est pas une épreuve de sprint cible.
        """
        event_upper = event_name.upper()
        if "60M" in event_upper and "HAIES" not in event_upper: return "60m"
        if "100M" in event_upper and "HAIES" not in event_upper: return "100m"
        if "200M" in event_upper and "HAIES" not in event_upper: return "200m"
        if "400M" in event_upper and "HAIES" not in event_upper: return "400m"
        return None

    def is_homologated(self, event_name, perf_data):
        """
        Vérifie si la performance est homologable pour les bilans.
        Règles :
        - Vent <= 2.0 m/s pour 100m, 200m (et haies/sauts extérieurs)
        - Pas de chrono manuel pour les sprints extérieurs (100m, 200m, 400m)
        """
        if not perf_data: return False
        
        event_norm = self.normalize_event(event_name)
        if not event_norm: return True # On ne filtre que ce qu'on connait (sprint)
        
        # Sprints extérieurs
        outdoor_sprints = ["100m", "200m", "400m"]
        
        if event_norm in outdoor_sprints:
            # Règle 1 : Chrono manuel
            if perf_data['is_manual']:
                return False
                
            # Règle 2 : Vent (pour 100m et 200m uniquement)
            if event_norm in ["100m", "200m"]:
                if perf_data['wind'] is not None and perf_data['wind'] > 2.0:
                    return False
                    
        return True

    def get_rankings(self, event_code, year, gender_code, category_code="", page=1):
        """
        Récupère le tableau exact de ton screenshot
        """
        # Pagination : 0 pour page 1, 1 pour page 2, etc. (Index de page)
        position = page - 1
        
        params = {
            "frmbase": "bilans",
            "frmmode": "1",
            "frmespace": "0",
            "frmannee": year,
            "frmepreuve": event_code,
            "frmsexe": gender_code,
            "frmcategorie": category_code,
            "frmposition": position,
            "frmpostback": "true"
        }
        
        # CORRECTION 60m (Salle) AVANT 2025
        # Avant 2025, le code 106 était le 60m Plein Air (ou générique) et 107 le 60m Salle.
        # A partir de 2025, c'est unifié ou différent.
        if event_code == "106":
            try:
                if int(year) < 2025:
                    params["frmepreuve"] = "107"
            except:
                pass # Si l'année n'est pas un entier valide
        
        # Optimisation : Filtrage serveur pour le vent (VR = Vent Régulier)
        # Uniquement pour les sprints extérieurs où le vent compte (100m=110, 200m=120)
        if event_code in ["110", "120"]:
            params["frmvent"] = "VR"
        
        try:
            response = self._make_request(self.base_url, params=params)
            soup = BeautifulSoup(response.text, 'html.parser')
            data = []
            rows = soup.find_all('tr')
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 8:
                    try:
                        raw_perf = cols[1].text.strip()
                        perf_info = self.clean_performance(raw_perf)
                        
                        if not perf_info: continue
                        
                        clean_perf = perf_info['chrono']
                        
                        name_cell = cols[2]
                        name = name_cell.text.strip()
                        link = name_cell.find('a')['href'] if name_cell.find('a') else None
                        
                        full_link = None
                        if link:
                            full_link = urljoin("https://www.athle.fr/bases/", link)
                        
                        club = cols[3].text.strip()
                        cat_info = cols[6].text.strip() if len(cols) > 6 else ""

                        # FILTRAGE HOMOLOGATION
                        event_name = "Unknown"
                        for k, v in FFA_EVENT_IDS.items():
                            if v == event_code:
                                event_name = k
                                break
                        
                        if not self.is_homologated(event_name, perf_info):
                            continue

                        # CALCUL DES POINTS
                        pts = ""
                        try:
                            # Mapping event_name to scoring_event (ex: "60m (Salle)" -> "60m")
                            scoring_event = event_name
                            if "60m" in event_name: scoring_event = "60m"
                            elif "100m" in event_name: scoring_event = "100m"
                            elif "200m" in event_name: scoring_event = "200m"
                            elif "400m" in event_name: scoring_event = "400m"
                            
                            pts = ScoringSystem.get_points(scoring_event, clean_perf, gender_code)
                        except:
                            pts = ""

                        data.append({
                            "Athlète": name,
                            "Lien": full_link,
                            "Chrono": clean_perf, # Float pour tri
                            "Perf": f"{clean_perf:.2f}", # String pour affichage
                            "Pts": f"{pts} pts" if pts else "-",
                            "Vent": perf_info['wind_str'] if perf_info['wind_str'] else "",
                            "Info": perf_info['badge'],
                            "Club": club,
                            "Catégorie": cat_info,
                            "Date": cols[-2].text.strip(),
                            "Lieu": cols[-1].text.strip()
                        })

                    except Exception as e:
                        continue

            df = pd.DataFrame(data)
            # Note: On ne recalcule pas le rang ici car on va concaténer plusieurs pages
            return df
            
        except Exception as e:
            st.error(f"Erreur de connexion : {e}")
            return pd.DataFrame()

    def get_athlete_profile(self, url, gender='M'):
        """
        Scrape la fiche athlète pour récupérer les records et infos
        """
        if not url: return None
        try:
            response = self._make_request(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            info = {"url": url}
            
            # --- Metadata Extraction ---
            page_text = soup.get_text()
            
            # Club
            club_match = re.search(r"Club\s*:\s*(.+?)(?:\n|\r|\t|  )", page_text)
            info['club'] = club_match.group(1).strip() if club_match else "N/A"
            
            # Category
            cat_match = re.search(r"Catégorie\s*:\s*(\w+)", page_text)
            info['category'] = cat_match.group(1).strip() if cat_match else "N/A"
            
            # Birth Year
            birth_match = re.search(r"Né\(e\) en\s*(\d{4})", page_text)
            info['birth_year'] = birth_match.group(1).strip() if birth_match else ""

            # --- Auto-détection du genre ---
            cats_f = re.findall(r"\b(SEF|ESF|JUF|CAF|MIF|BEF|POF|VEF)\b", page_text)
            cats_m = re.findall(r"\b(SEM|ESM|JUM|CAM|MIM|BEM|POM|VEM)\b", page_text)
            
            if len(cats_f) > len(cats_m):
                gender = 'F'
            elif len(cats_m) > len(cats_f):
                gender = 'M'
                
            info['gender'] = gender
            
            tables = soup.find_all('table')
            records_data = []
            
            for table in tables:
                if "Epreuve" in table.text or "Perf" in table.text:
                    rows = table.find_all('tr')
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 2:
                            epreuve = cols[0].text.strip()
                            perf = cols[1].text.strip()
                            
                            if "Epreuve" in epreuve or "Perf" in perf: continue
                            if epreuve.startswith("Performance"): continue
                                
                            perf_info = self.clean_performance(perf)
                            if not perf_info: continue
                            
                            clean_perf = perf_info['chrono']
                            
                            if not self.is_homologated(epreuve, perf_info): continue
                            
                            scoring_event = ScoringSystem.normalize_event_name_for_scoring(epreuve)
                            normalized_event = self.normalize_event(epreuve)
                            
                            pts = ""
                            niveau = ""
                            if scoring_event and clean_perf:
                                pts = ScoringSystem.get_points(scoring_event, clean_perf, gender)
                                niveau = ScoringSystem.get_level(scoring_event, clean_perf, gender)
                            
                            if clean_perf:
                                records_data.append({
                                    "Epreuve": normalized_event if normalized_event else epreuve,
                                    "Epreuve_Originale": epreuve,
                                    "IsSprint": bool(normalized_event),
                                    "Perf": perf,
                                    "Chrono": clean_perf,
                                    "Pts": pts if pts else (cols[2].text.strip() if len(cols) > 2 else ""),
                                    "Niveau": niveau if niveau else (cols[3].text.strip() if len(cols) > 3 else ""),
                                    "Lieu": cols[-1].text.strip() if len(cols) > 2 else "",
                                    "Date": cols[-2].text.strip() if len(cols) > 3 else ""
                                })
            
            if records_data:
                df_recs = pd.DataFrame(records_data)
                df_recs = df_recs.sort_values("Chrono")
                df_recs = df_recs.drop_duplicates(subset="Epreuve", keep="first")
                info['records'] = df_recs.to_dict('records')
            else:
                info['records'] = []
                
            return info

        except Exception as e:
            print(f"Erreur scraping profil {url}: {e}")
            return None

    def get_competition_id(self, year, category, is_indoor):
        """
        Cherche l'ID de la compétition "Championnats de France" pour l'année et la catégorie données.
        Retourne l'ID ou None.
        """
        season_code = int(year) - 1994
        
        # Tentative de recherche via la page liste.aspx (plus robuste ?)
        # On cherche dans les résultats par année/catégorie
        params = {
            "frmbase": "resultats", # Ou "calendrier" ? Essayons resultats avec liste.aspx
            "frmmode": "1",
            "frmespace": "0",
            "frmannee": year,
            "frmsaison": season_code,
            "frmcategorie": category,
            "frmniveau": "CHPT" 
        }
        
        try:
            # URL corrigée : liste.aspx est souvent utilisée pour les listes de résultats
            url = "https://www.athle.fr/bases/liste.aspx" 
            response = self._make_request(url, params=params)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links = soup.find_all('a')
            for link in links:
                href = link.get('href')
                text = link.text.upper()
                if href and "FRANCE" in text:
                    if is_indoor and "SALLE" in text:
                        match = re.search(r"frmcompetition=(\d+)", href)
                        if match: return match.group(1)
                    elif not is_indoor and "SALLE" not in text:
                         match = re.search(r"frmcompetition=(\d+)", href)
                         if match: return match.group(1)
            return None
        except:
            return None

    def get_actual_qualification(self, year, event_code, gender, category, is_indoor):
        """
        Récupère la performance du dernier qualifié (Réel) en scrapant la page des qualifiés.
        """
        cat_search = category[:2] 
        if cat_search == "SE": cat_search = "EL" 
        
        comp_id = self.get_competition_id(year, cat_search, is_indoor)
        if not comp_id: return "N/A"
        
        season_code = int(year) - 1994
        params = {
            "frmbase": "qualifies",
            "frmmode": "1",
            "frmespace": "0",
            "frmsaison": season_code,
            "frmcompetition": comp_id,
            "frmepreuve": event_code, 
            "frmsexe": gender,
            "frmcategorie": cat_search,
            "frmpostback": "true"
        }
        
        try:
            response = self._make_request(self.base_url, params=params)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parsing plus robuste
            # On cherche le tableau principal
            # Souvent c'est le dernier tableau ou celui avec "Perf"
            tables = soup.find_all('table')
            for table in tables:
                if "Perf" in table.text:
                    rows = table.find_all('tr')
                    last_perf = "N/A"
                    for row in rows:
                        cols = row.find_all('td')
                        # Structure variable, on cherche une perf (X.XX ou XX.XX)
                        for col in cols:
                            txt = col.text.strip()
                            if re.match(r"^\d{1,2}\.\d{2}$", txt):
                                last_perf = txt
                                # On continue pour trouver la dernière
                    
                    if last_perf != "N/A":
                        return last_perf
            
            return "N/A" 
            
        except:
            return "Err"

# ==========================================
# 3. INTERFACE UTILISATEUR (MODULAIRE)
# ==========================================

# ==========================================
# 3. INTERFACE UTILISATEUR (PREMIUM UI)
# ==========================================

def load_css():
    st.markdown("""
        <style>
        /* Main Background */
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #1f2937;
        }
        
        /* Custom Cards */
        .metric-card {
            background-color: #1f2937;
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        
        /* Status Bar */
        .status-bar {
            background-color: #1f2937;
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 10px 20px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9em;
        }
        
        /* Qualification Line */
        .qualif-line {
            border-top: 2px dashed #f59e0b;
            margin: 20px 0;
            text-align: center;
            position: relative;
        }
        .qualif-line span {
            background-color: #0e1117;
            padding: 0 10px;
            color: #f59e0b;
            font-weight: bold;
            position: relative;
            top: -12px;
        }
        
        /* Table Styling */
        [data-testid="stDataFrame"] {
            border: 1px solid #374151;
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }
        
        /* Custom Buttons */
        .stButton button {
            background-color: #2563eb;
            color: white;
            border: none;
            border-radius: 6px;
        }
        .stButton button:hover {
            background-color: #1d4ed8;
        }
        
        /* Blinking Animation */
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.4; }
            100% { opacity: 1; }
        }
        .blink {
            animation: blink 1.5s infinite;
            color: #22c55e; /* Green */
        }
        /* Result Card */
        .result-card {
            background-color: #1f2937;
            border: 1px solid #374151;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: transform 0.2s;
        }
        .result-card:hover {
            transform: translateY(-2px);
            border-color: #4b5563;
        }
        .result-card.qualified {
            border-left: 4px solid #22c55e;
        }
        .result-card.not-qualified {
            border-left: 4px solid #ef4444;
        }
        .rc-rank {
            font-size: 1.2em;
            font-weight: bold;
            color: #9ca3af;
            width: 50px; /* Wider for 3 digits */
            text-align: center;
            flex-shrink: 0;
        }
        .rc-rank.gold { color: #fbbf24; text-shadow: 0 0 10px rgba(251, 191, 36, 0.3); }
        .rc-rank.silver { color: #9ca3af; text-shadow: 0 0 10px rgba(156, 163, 175, 0.3); }
        .rc-rank.bronze { color: #b45309; text-shadow: 0 0 10px rgba(180, 83, 9, 0.3); }
        
        .rc-info {
            flex-grow: 1;
            padding: 0 15px;
            overflow: hidden;
        }
        .rc-name {
            font-size: 1.1em;
            font-weight: bold;
            color: #f3f4f6;
            margin-bottom: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .rc-club {
            font-size: 0.85em;
            color: #9ca3af;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .rc-details {
            font-size: 0.75em;
            color: #6b7280;
            margin-top: 4px;
        }
        .rc-perf-box {
            text-align: right;
            min-width: 90px;
            flex-shrink: 0;
        }
        .rc-perf {
            font-size: 1.4em;
            font-weight: bold;
            color: #ffffff;
        }
        .rc-wind {
            font-size: 0.8em;
            color: #9ca3af;
        }
        .rc-gap {
            font-size: 0.75em;
            font-weight: bold;
            margin-top: 2px;
        }
        .rc-gap.cut-ok { color: #22c55e; }
        .rc-gap.cut-ko { color: #ef4444; }
        .rc-status {
            font-size: 0.75em;
            color: #3b82f6;
            font-weight: bold;
        }
        .rc-points {
            color: #60a5fa; /* Blue-400 */
            font-weight: bold;
            font-size: 0.9em;
        }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Gère la navigation latérale style Dashboard"""
    with st.sidebar:
        # Logo Area
        # Utilisation du logo local
        try:
            # st.image("Gemini_Generated_Image_py9xxopy9xxopy9x.png", width=80)
            # Hack pour éviter le fullscreen au survol : utilisation de HTML img avec pointer-events: none
            import base64
            def get_base64_of_bin_file(bin_file):
                with open(bin_file, 'rb') as f:
                    data = f.read()
                return base64.b64encode(data).decode()

            img_base64 = get_base64_of_bin_file("Gemini_Generated_Image_py9xxopy9xxopy9x.png")
            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 10px;">
                    <img src="data:image/png;base64,{img_base64}" width="80" style="pointer-events: none;">
                </div>
                """,
                unsafe_allow_html=True
            )
        except:
            st.image("https://img.icons8.com/color/96/sprint.png", width=60) # Fallback
            
        st.markdown("### TRACK PULSE\n**FRANCE**")
        st.caption("v.01")
        st.divider()
        
        # Navigation
        nav = st.radio(
            "Navigation",
            ["CLASSEMENTS", "ATHLÈTES", "BOÎTE À OUTILS"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Bottom Menu (Visual only)
        st.caption("👤 Mon Compte")
        st.caption("⚙️ Paramètres")
        st.caption("❓ Aide")
        
        return nav

def render_header(annee, epreuve_label, genre_label, cat_label):
    """Affiche le Header style Breadcrumbs"""
    st.markdown(f"""
        <h1 style='font-size: 24px; margin-bottom: 20px;'>
            <span style='color: #9ca3af;'>Saison {annee} > </span>
            <span style='color: #3b82f6;'>{epreuve_label} {genre_label}</span>
            <span style='color: #9ca3af;'> > {cat_label if cat_label else "Toutes Catégories"}</span>
        </h1>
    """, unsafe_allow_html=True)

def render_filters_horizontal():
    """Affiche les filtres horizontalement"""
    with st.container():
        c1, c2, c3, c4, c5 = st.columns([1.5, 2, 2, 2, 1.5])
        
        with c1:
            annee = st.selectbox("Saison", [str(y) for y in range(2026, 2019, -1)], index=0)
        
        with c2:
            epreuve_label = st.selectbox("Epreuve", list(FFA_EVENT_IDS.keys()))
            epreuve_code = FFA_EVENT_IDS[epreuve_label]
            
        with c3:
            genre_label = st.selectbox("Genre", ["Masculin", "Féminin"])
            genre_code = "M" if genre_label == "Masculin" else "F"
            
        with c4:
            cat_options = {"Toutes": "", "Senior": "SEM" if genre_code=="M" else "SEF", "Espoir": "ESM", "Junior": "JUM"}
            cat_label = st.selectbox("Catégorie", list(cat_options.keys()))
            cat_code = cat_options[cat_label]
            
        with c5:
            st.write("") # Spacer
            if st.button("ACTUALISER", type="primary", use_container_width=True):
                with st.spinner("Chargement..."):
                    st.session_state['current_page'] = 1
                    # Sauvegarde du contexte de recherche
                    st.session_state['current_genre_code'] = genre_code
                    
                    df = scraper.get_rankings(epreuve_code, annee, genre_code, cat_code, page=1)
                    if not df.empty:
                        df['Rang'] = df['Chrono'].rank(method='min').astype(int)
                    st.session_state['df_results'] = df
                    
    return annee, epreuve_label, epreuve_code, genre_label, genre_code, cat_label, cat_code

def render_status_bar(cutoff_val, count):
    """Affiche la barre de statut"""
    st.markdown(f"""
        <div class="status-bar">
            <div><span class="blink">●</span> <b>Bilans EN COURS</b></div>
            <div>🎯 <b>Cut Estimé</b> : {cutoff_val}</div>
            <div>🔄 <b>MAJ</b> : Aujourd'hui {time.strftime("%H:%M")}</div>
        </div>
    """, unsafe_allow_html=True)

def render_results_split(df, cutoff_rank=24):
    """Affiche le tableau des résultats (Unifié)"""
    
    # Préparation des données
    df["Q"] = ["✅" if i < cutoff_rank else "❌" for i in range(len(df))]
    
    # Gap Calculation
    if len(df) >= cutoff_rank:
        cutoff_perf = df.iloc[cutoff_rank-1]['Chrono']
        df["Ecart"] = df.apply(lambda r: r['Chrono'] - cutoff_perf, axis=1)
        df["Ecart_Str"] = df["Ecart"].apply(lambda x: f"{x:+.2f}s" if x != 0 else "CUT")
    else:
        df["Ecart_Str"] = "-"
        cutoff_perf = "N/A"

    # Permanent Card View
    # CUSTOM HTML CARD VIEW
    for index, row in df.iterrows():
        is_qualified = row['Q'] == "✅"
        card_class = "qualified" if is_qualified else "not-qualified"
        
        # Data preparation
        rank = row['Rang']
        name = row['Athlète']
        club = row['Club']
        perf = row['Perf']
        wind = row.get('Vent', '')
        pts = row.get('Pts', '-')
        date = row.get('Date', '')
        place = row['Lieu']
        info = row.get('Info', '')
        ecart = row.get('Ecart_Str', '')
        
        # Medal Styling
        rank_class = "rc-rank"
        if rank == 1: rank_class += " gold"
        elif rank == 2: rank_class += " silver"
        elif rank == 3: rank_class += " bronze"
        
        # Gap Styling
        gap_html = ""
        if ecart != "-" and ecart != "CUT":
             gap_class = "cut-ok" if is_qualified else "cut-ko"
             gap_html = f'<div class="rc-gap {gap_class}">Cut {ecart}</div>'
        
        # Status HTML (RP/SB)
        status_html = ""
        if info:
            status_html = f'<span class="rc-status">{info}</span> • '

        # Link handling
        link_html = ""
        if row.get('Lien'):
            link_html = f'<a href="{row["Lien"]}" target="_blank" style="text-decoration:none; color:inherit;">'
        
        close_link_html = "</a>" if link_html else ""
        
        # Points Styling
        pts_html = f'<span class="rc-points">{pts}</span>' if pts != "-" else "-"

        st.markdown(f"""
        {link_html}
        <div class="result-card {card_class}">
            <div class="{rank_class}">#{rank}</div>
            <div class="rc-info">
                <div class="rc-name">{name}</div>
                <div class="rc-club">{status_html}{club}</div>
                <div class="rc-details">{place} • {date} • {pts_html}</div>
            </div>
            <div class="rc-perf-box"><div class="rc-perf">{perf}</div><div class="rc-wind">{wind}</div>{gap_html}</div>
        </div>
        {close_link_html}
        """, unsafe_allow_html=True)

def render_profile_tab(genre_code, athlete_url=None, athlete_name=None):
    """Affiche l'onglet Profil Athlète"""
    st.header("👤 Fiche Athlète")
    
    selected_athlete = None
    url = athlete_url
    
    # Mode Recherche (URL fournie directement)
    if athlete_url:
        selected_athlete = athlete_name if athlete_name else "Athlète Sélectionné" 
    
    # Mode Liste (Depuis le classement)
    elif 'df_results' in st.session_state and not st.session_state['df_results'].empty:
        df = st.session_state['df_results']
        athlete_list = df['Athlète'].tolist()
        selected_athlete = st.selectbox("Rechercher un athlète dans le classement", athlete_list)
        if selected_athlete:
            row = df[df['Athlète'] == selected_athlete].iloc[0]
            url = row.get('Lien')
    else:
        if not athlete_url:
            st.info("Utilisez la recherche ci-dessus ou chargez un classement.")
            return

    if url:
        # Auto-load if URL is provided (search mode) or button clicked (list mode)
        should_load = True if athlete_url else st.button(f"Voir la fiche de {selected_athlete}", use_container_width=True)
        
        if should_load:
            with st.spinner("Chargement du profil..."):
                profile = scraper.get_athlete_profile(url, genre_code)
                if profile and profile.get('records'):
                    # Header Profil
                    # Nom et Infos
                    name_display = selected_athlete
                    st.subheader(name_display)
                    
                    infos = []
                    if profile.get('club') and profile['club'] != "N/A":
                        infos.append(f"🏢 {profile['club']}")
                    if profile.get('category') and profile['category'] != "N/A":
                        infos.append(f"🏷️ {profile['category']}")
                    if profile.get('birth_year'):
                        try:
                            age = 2025 - int(profile['birth_year'])
                            infos.append(f"🎂 {age} ans ({profile['birth_year']})")
                        except: pass
                        
                    if infos:
                        st.caption(" | ".join(infos))
                        
                    st.markdown(f"[Voir sur le site FFA]({url})")

                    st.divider()
                    
                    # --- PREDICTION SECTION (V3) ---
                    st.subheader("🔮 Potentiel Saison 2025")
                    
                    pipeline = get_prediction_pipeline()
                    with st.spinner("Analyse de la forme et du potentiel..."):
                        pred_res = pipeline.run_pipeline(url, season_year=2025, verbose=False)
                        
                    if pred_res:
                        p_real = pred_res['p_real_pred']
                        features = pred_res['features']
                        
                        # Confidence Interval (Based on MAPE 1.35% -> approx +/- 0.13s for 10s)
                        # Let's use a fixed margin or percentage.
                        # MAPE is Mean Absolute Percentage Error.
                        # Margin = p_real * 0.0135
                        margin = p_real * 0.0135
                        low = p_real - margin
                        high = p_real + margin
                        
                        # Layout
                        c_pred, c_conf, c_feat = st.columns([1.5, 1.5, 3])
                        
                        with c_pred:
                            st.metric(
                                "Chrono Potentiel", 
                                f"{p_real:.2f}s", 
                                help="Performance réalisable à 0 vent en pic de forme"
                            )
                        
                        with c_conf:
                            st.metric(
                                "Intervalle Confiance", 
                                f"{low:.2f}s - {high:.2f}s",
                                delta="± 1.35%",
                                delta_color="off",
                                help="Basé sur la précision du modèle V3 (MAPE)"
                            )
                            
                        with c_feat:
                            st.caption("🔍 Facteurs Clés détectés par l'IA :")
                            # Display key features
                            sb_early = features.get('SB_Early')
                            slope = features.get('Slope_Early')
                            
                            feat_text = ""
                            if not pd.isna(sb_early):
                                feat_text += f"- **Début de saison** : {sb_early:.2f}s (Intrinsèque)\n"
                            if slope != 0:
                                trend_str = "📈 En progression" if slope < 0 else "📉 En baisse" # Lower time is better
                                feat_text += f"- **Tendance** : {trend_str} ({slope:.3f})\n"
                            
                            st.markdown(feat_text if feat_text else "Données historiques insuffisantes pour explication détaillée.")
                        
                        # Wind Simulation Button
                        if st.button("💨 Voir avec +2.0 m/s", type="secondary"):
                            from prediction_pipeline import MureikaModel
                            # Calculate with +2.0 wind
                            p_wind = MureikaModel.predict_perf_at_wind(p_real, 2.0, dist="100m")
                            
                            # Calculate Interval with +2.0 wind
                            low_wind = MureikaModel.predict_perf_at_wind(low, 2.0, dist="100m")
                            high_wind = MureikaModel.predict_perf_at_wind(high, 2.0, dist="100m")
                            
                            st.markdown(f"""
                            <div style="margin-top: 20px;">
                                <div style="text-align: center; font-size: 24px; color: #555; margin-bottom: 8px;">⬇</div>
                                <div style="
                                    background-color: #1E1E1E; 
                                    border: 1px solid #333; 
                                    border-left: 5px solid #4CAF50;
                                    border-radius: 10px; 
                                    padding: 20px; 
                                    text-align: center; 
                                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                                    <div style="font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 5px;">
                                        Potentiel (Vent +2.0 m/s)
                                    </div>
                                    <div style="font-size: 42px; color: #4CAF50; font-weight: 800; line-height: 1.1; margin-bottom: 5px;">
                                        🚀 {p_wind:.2f}s
                                    </div>
                                    <div style="font-size: 16px; color: #CCC;">
                                        Intervalle : <span style="color: #FFF; font-weight: bold;">{low_wind:.2f}s - {high_wind:.2f}s</span>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            
                    else:
                        st.info("Pas assez de données récentes (3 courses min) pour une prédiction fiable.")

                    st.divider()
                    
                    df_sprint = pd.DataFrame([r for r in profile['records'] if r['IsSprint']]).copy()
                    df_other = pd.DataFrame([r for r in profile['records'] if not r['IsSprint']]).copy()
                    
                    # SPRINT CARDS
                    st.subheader("⚡ Records Sprint")
                    if not df_sprint.empty:
                        order = {"60m": 1, "100m": 2, "200m": 3, "400m": 4}
                        df_sprint['Order'] = df_sprint['Epreuve'].map(order)
                        df_sprint = df_sprint.sort_values("Order")
                        
                        cols = st.columns(4)
                        for idx, row_rec in enumerate(df_sprint.itertuples()):
                            with cols[idx % 4]:
                                st.metric(
                                    label=row_rec.Epreuve,
                                    value=row_rec.Perf,
                                    delta=f"{row_rec.Pts} pts ({row_rec.Niveau})" if row_rec.Pts else None,
                                    delta_color="normal"
                                )
                                st.caption(f"📍 {row_rec.Lieu} ({row_rec.Date})")
                    else:
                        st.info("Aucun record de sprint.")

                    # OTHER RECORDS
                    if not df_other.empty:
                        st.subheader("🏅 Autres Records")
                        st.dataframe(
                            df_other[["Epreuve", "Perf"]],
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.error("Impossible de charger le profil.")
    else:
        st.warning("Pas de lien disponible.")

def render_duel_tab(genre_code="M"):
    """Affiche l'onglet Duel"""
    st.header("⚔️ Duel")
    
    if 'df_results' in st.session_state and not st.session_state['df_results'].empty:
        df = st.session_state['df_results']
        athlete_list = df['Athlète'].tolist()
        
        c1, c2 = st.columns(2)
        with c1:
            athlete_1 = st.selectbox("Athlète 1", athlete_list, key="duel_1")
        with c2:
            athlete_2 = st.selectbox("Athlète 2", athlete_list, key="duel_2")
            
        if athlete_1 and athlete_2:
            if athlete_1 == athlete_2:
                st.warning("Sélectionnez deux athlètes différents.")
            else:
                if st.button("Lancer le Duel", type="primary"):
                    row1 = df[df['Athlète'] == athlete_1].iloc[0]
                    row2 = df[df['Athlète'] == athlete_2].iloc[0]
                    
                    st.markdown("### 🏁 Saison Actuelle")
                    
                    # Custom CSS for cards
                    st.markdown("""
                    <style>
                    .duel-card {
                        background-color: #1e293b;
                        padding: 15px;
                        border-radius: 10px;
                        text-align: center;
                        border: 1px solid #334155;
                    }
                    .duel-card.winner {
                        border: 2px solid #22c55e;
                        box-shadow: 0 0 10px rgba(34, 197, 94, 0.3);
                    }
                    .duel-perf {
                        font-size: 2em;
                        font-weight: bold;
                        color: #f8fafc;
                    }
                    .duel-name {
                        font-size: 1.1em;
                        color: #94a3b8;
                        margin-bottom: 5px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Comparison Logic (Season)
                    perf1 = row1['Chrono']
                    perf2 = row2['Chrono']
                    diff = perf1 - perf2
                    winner_season = athlete_1 if diff < 0 else athlete_2
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        is_win = (athlete_1 == winner_season)
                        st.markdown(f"""
                        <div class="duel-card {'winner' if is_win else ''}">
                            <div class="duel-name">{athlete_1}</div>
                            <div class="duel-perf">{row1['Perf']}</div>
                            <div>Rang: {row1['Rang']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with c2:
                        is_win = (athlete_2 == winner_season)
                        st.markdown(f"""
                        <div class="duel-card {'winner' if is_win else ''}">
                            <div class="duel-name">{athlete_2}</div>
                            <div class="duel-perf">{row2['Perf']}</div>
                            <div>Rang: {row2['Rang']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.divider()
                    
                    # RECORDS COMPARISON
                    st.subheader("📊 Confrontation des Records")
                    url1 = row1.get('Lien')
                    url2 = row2.get('Lien')
                    
                    if url1 and url2:
                        try:
                            with st.spinner("Récupération des records..."):
                                # Fetch profiles
                                p1 = scraper.get_athlete_profile(url1, genre_code)
                                p2 = scraper.get_athlete_profile(url2, genre_code)
                                
                                if p1 and p2 and 'records' in p1 and 'records' in p2:
                                    # Find common events
                                    r1_map = {r['Epreuve']: r for r in p1['records']}
                                    r2_map = {r['Epreuve']: r for r in p2['records']}
                                    
                                    common_events = set(r1_map.keys()) & set(r2_map.keys())
                                    
                                    if common_events:
                                        # Sort events
                                        sorted_events = sorted(list(common_events))
                                        
                                        for evt in sorted_events:
                                            rec1 = r1_map[evt]
                                            rec2 = r2_map[evt]
                                            
                                            dc1, dc2 = st.columns(2)
                                            with dc1:
                                                st.metric(f"{evt} ({athlete_1})", rec1['Perf'], help=f"{rec1['Lieu']} - {rec1['Date']}")
                                            with dc2:
                                                st.metric(f"{evt} ({athlete_2})", rec2['Perf'], help=f"{rec2['Lieu']} - {rec2['Date']}")
                                            st.divider()
                                    else:
                                        st.info("Aucune épreuve commune trouvée dans les records.")
                                else:
                                    st.warning("Profils incomplets ou sans records.")
                        except Exception as e:
                            st.error(f"Erreur lors de la récupération des profils: {e}")
                    else:
                        st.warning("Liens profils manquants.")

def render_tools_main():
    """Affiche les outils dans la zone principale"""
    st.title("🛠️ Boîte à Outils")
    
    tab_conv, tab_vent, tab_qualif = st.tabs(["Convertisseur", "Calculateur Vent", "Qualification"])
    
    with tab_conv:
        st.header("Convertisseur Points ↔️ Performance")
        st.caption("Table Hongroise IAAF")
        
        c1, c2 = st.columns(2)
        with c1:
            t_event = st.selectbox("Epreuve", list(ScoringSystem.HUNGARIAN_COEFFS['M'].keys()), key="tool_evt")
        with c2:
            t_gender = st.radio("Genre", ["M", "F"], horizontal=True, key="tool_gender")
            
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### Perf ➡️ Points")
            val_perf = st.number_input("Performance", value=10.0, step=0.01, format="%.2f")
            if val_perf:
                pts = ScoringSystem.get_points(t_event, val_perf, t_gender)
                st.metric("Points", pts)
                
        with col_b:
            st.markdown("### Points ➡️ Perf")
            val_pts = st.number_input("Points", value=1000, step=1)
            if val_pts:
                coeffs = ScoringSystem.HUNGARIAN_COEFFS.get(t_gender, {}).get(t_event)
                if coeffs:
                    A, D, B, C = coeffs
                    try:
                        term = (val_pts - C) * (10**D) / A
                        if term >= 0:
                            is_course = t_event in ['60m', '100m', '200m', '400m', '110mH', '100mH', '400mH', '800m', '1000m', '1500m']
                            is_saut = t_event in ['Hauteur', 'Perche', 'Longueur', 'Triple']
                            is_lancer = t_event in ['Poids', 'Disque', 'Marteau', 'Javelot']
                            
                            if is_course:
                                res = -math.sqrt(term) - B
                                res = res / 100
                            elif is_saut or is_lancer:
                                res = math.sqrt(term) - B
                                res = res / 100
                            else:
                                res = math.sqrt(term) - B
                                
                            st.metric("Performance estimée", f"{res:.2f}")
                        else:
                            st.error("Impossible (Points trop bas)")
                    except Exception as e:
                        st.error(f"Erreur: {e}")

    with tab_vent:
        st.header("💨 Calculateur d'ajustement de vent")
        st.caption("Basé sur les travaux de Jonas Mureika (50m, 60m, 100m, 200m)")
        
        c1, c2 = st.columns(2)
        with c1:
            w_dist = st.selectbox("Distance", ["50m", "60m", "100m", "200m"], index=2)
        with c2:
            w_gender = st.radio("Genre", ["M", "F"], horizontal=True, key="vent_gender")
            
        c3, c4, c5 = st.columns(3)
        with c3:
            w_perf = st.number_input("Performance (s)", value=10.0, step=0.01, format="%.2f")
        with c4:
            w_wind = st.number_input("Vent (m/s)", value=0.0, step=0.1, format="%.1f")
        with c5:
            w_alt = st.number_input("Altitude (m)", value=0.0, step=100.0)
            
        w_lane = 4
        if w_dist == "200m":
            w_lane = st.selectbox("Couloir", [1, 2, 3, 4, 5, 6, 7, 8], index=3)
            
        w_target = st.number_input("Vent Cible (m/s)", value=0.0, step=0.1, format="%.1f")
        
        if st.button("Calculer"):
            
            def calculate_zero_wind(t, w, alt, dist, gender, lane):
                dens = math.exp(-0.000125 * alt)
                
                if dist in ["50m", "60m", "100m"]:
                    sp = int(dist.replace("m", ""))
                    # Formula: t0 = (1.028 - 0.028 * dens * ((1.0 - w * t / sp) ** 2)) * t
                    # Note: The JS formula uses 't_w' (performance time) in the term (w * t_w / sp)
                    term = 1.0 - (w * t / sp)
                    t0 = (1.028 - 0.028 * dens * (term * term)) * t
                    return t0
                    
                elif dist == "200m":
                    # 200m Logic from Mureika
                    # dt depends on lane, wind, alt
                    # t0 = t - dt
                    
                    # Coefficients
                    # Men < 21.5s (Elite)
                    is_elite_men = (gender == "M" and t < 21.5)
                    
                    dt = 0.0
                    w2 = w * w
                    
                    if is_elite_men:
                        if lane == 1: dt = .6823850739e-2*w2 - .6261746031e-1*w + .6769739610e-3 + (-.8533086093e-6*w2 + .7409523797e-5*w - .9576524416e-4)*alt
                        elif lane == 2: dt = .6037981841e-2*w2 - .6160238093e-1*w - .5337868373e-2 + (-.7012368506e-6*w2 + .6901904772e-5*w - .9484254800e-4)*alt
                        elif lane == 3: dt = .5937950920e-2*w2 - .6508253968e-1*w - .4702741522e-2 + (-.7021645076e-6*w2 + .7283809515e-5*w - .9485541123e-4)*alt
                        elif lane == 4: dt = .5849876305e-2*w2 - .6855952381e-1*w - .5205524601e-2 + (-.6920222619e-6*w2 + .7647619043e-5*w - .9491032781e-4)*alt
                        elif lane == 5: dt = .5727788093e-2*w2 - .7192698412e-1*w - .4418058132e-2 + (-.6745825703e-6*w2 + .8003809515e-5*w - .9506468759e-4)*alt
                        elif lane == 6: dt = .5635281356e-2*w2 - .7525793651e-1*w - .4965367685e-2 + (-.6675324621e-6*w2 + .8375238100e-5*w - .9518787886e-4)*alt
                        elif lane == 7: dt = .5562976702e-2*w2 - .7846031747e-1*w - .4763760038e-2 + alt*(-.6695732846e-6*w2 + .8730476192e-5*w - .9523141616e-4)
                        elif lane == 8: dt = .5497165554e-2*w2 - .8159285713e-1*w - .5002267739e-2 + (-.6547309794e-6*w2 + .9080952375e-5*w - .9529226969e-4)*alt
                    else:
                        # Women or Slower Men
                        if lane == 1: dt = .8164192923e-2*w2 - .7134126985e-1*w - .5253349577e-2 + (-.9581941721e-6*w2 + .7988571428e-5*w - .1015294992e-3)*alt
                        elif lane == 2: dt = .8012420137e-2*w2 - .7549761905e-1*w - .5664811539e-2 + alt*(-.9247990043e-6*w2 + .8438095233e-5*w - .1016949907e-3)
                        elif lane == 3: dt = .7904916485e-2*w2 - .7968492063e-1*w - .5239125688e-2 + (-.9269635173e-6*w2 + .8905714292e-5*w - .1016551638e-3)*alt
                        elif lane == 4: dt = .8307720076e-2*w2 - .8214841271e-1*w - .5935065061e-2 + (-.1204329010e-5*w2 + .8447619050e-5*w - .1016346319e-3)*alt
                        elif lane == 5: dt = .7666048256e-2*w2 - .8769523808e-1*w - .5233972510e-2 + (-.9012368561e-6*w2 + .9736190467e-5*w - .1018901670e-3)*alt
                        elif lane == 6: dt = .7537878777e-2*w2 - .9157222225e-1*w - .4871572736e-2 + (-.8844155844e-6*w2 + .1013999999e-4*w - .1020086579e-3)*alt
                        elif lane == 7: dt = .7438260160e-2*w2 - .9541269842e-1*w - .4911152390e-2 + (-.8664811526e-6*w2 + .1057238095e-4*w - .1021726653e-3)*alt
                        elif lane == 8: dt = .7340342199e-2*w2 - .9905396825e-1*w - .5168418916e-2 + (-.8719851461e-6*w2 + .1096761904e-4*w - .1021486705e-3)*alt
                        
                    if w == 0 and alt == 0:
                        dt = 0
                        
                    return t - dt
                
                return t

            # 1. Calculate Zero Wind/Alt Time (t0)
            t0 = calculate_zero_wind(w_perf, w_wind, w_alt, w_dist, w_gender, w_lane)
            
            # 2. Calculate Target Time (t_target) at w_target
            # For 200m, dt depends only on wind/alt, not time. So t_target = t0 + dt(w_target)
            # For others, t0 depends on t. We need to solve for t given t0 and w_target.
            
            t_target = t0
            
            if w_dist == "200m":
                # Calculate dt for target wind
                # We need to call the logic again but we don't have a clean function for just dt.
                # Let's cheat: t_target_zero = calculate_zero_wind(t_target_guess, w_target, w_alt, ...)
                # t0 = t_target_guess - dt_target
                # => t_target_guess = t0 + dt_target
                # And dt_target is calculated using w_target.
                # Since dt logic uses 't' only to decide Elite vs Non-Elite, we assume the category doesn't change.
                
                # Hack: Calculate dt for w_target by passing a dummy time (w_perf) to preserve Elite status check
                # Then subtract the result from w_perf to get dt.
                # t_dummy_zero = calculate_zero_wind(w_perf, w_target, w_alt, ...)
                # t_dummy_zero = w_perf - dt_target => dt_target = w_perf - t_dummy_zero
                
                t_dummy_zero = calculate_zero_wind(w_perf, w_target, w_alt, w_dist, w_gender, w_lane)
                dt_target = w_perf - t_dummy_zero
                t_target = t0 + dt_target
                
            else:
                # Iterative solver for 100m/60m/50m
                t_guess = t0
                for _ in range(20):
                    # We want to find t_guess such that calculate_zero_wind(t_guess, w_target, w_alt) == t0
                    t_calc_zero = calculate_zero_wind(t_guess, w_target, w_alt, w_dist, w_gender, w_lane)
                    diff = t_calc_zero - t0
                    if abs(diff) < 0.00001: break
                    t_guess -= diff # Simple correction
                t_target = t_guess
            
            st.metric(f"Perf à {w_target} m/s", f"{t_target:.3f} s", delta=f"{t_target - w_perf:.3f} s", delta_color="inverse")
            st.info(f"Equivalent Vent Nul / Alt 0: {t0:.3f} s")

    with tab_qualif:
        st.header("📊 Historique de Qualification")
        st.caption("Moyenne du N-ième performeur sur les 5 dernières années (Bilans)")
        
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            q_event = st.selectbox("Epreuve", ["100m", "200m", "400m", "60m (Salle)"], key="q_evt")
            q_event_code = FFA_EVENT_IDS[q_event]
        with qc2:
            q_gender = st.radio("Genre", ["M", "F"], horizontal=True, key="q_gen")
            q_gender_code = "M" if q_gender == "M" else "F"
        with qc3:
            q_cat = st.selectbox("Catégorie", ["Senior", "Espoir", "Junior", "Cadet"], key="q_cat")
            q_cat_map = {"Senior": "SEM", "Espoir": "ESM", "Junior": "JUM", "Cadet": "CAM"}
            if q_gender == "F":
                q_cat_map = {k: v.replace("M", "F") for k, v in q_cat_map.items()}
            q_cat_code = q_cat_map[q_cat]

        # Determine Cutoff
        is_indoor = "Salle" in q_event
        cutoff_rank = 24 if is_indoor else 32
        st.info(f"Seuil théorique: Top {cutoff_rank}")
        
        if st.button("Lancer l'analyse (2021-2025)"):
            years = range(2021, 2026)
            results = []
            
            progress_bar = st.progress(0)
            
            for i, y in enumerate(years):
                # Correction 60m
                evt_code_iter = q_event_code
                if evt_code_iter == "106" and y < 2025: evt_code_iter = "107"
                
                df_y = scraper.get_rankings(evt_code_iter, str(y), q_gender_code, q_cat_code)
                
                val_cutoff = None
                if not df_y.empty and len(df_y) >= cutoff_rank:
                    df_y = df_y.sort_values("Chrono")
                    row = df_y.iloc[cutoff_rank-1]
                    val_cutoff = row['Chrono']
                
                # Actual Scraping
                val_actual = scraper.get_actual_qualification(str(y), evt_code_iter, q_gender_code, q_cat_code, is_indoor)
                
                results.append({
                    "Année": str(y),
                    f"Top {cutoff_rank}": val_cutoff if val_cutoff else "N/A",
                    "Réel": val_actual
                })
                progress_bar.progress((i + 1) / len(years))
                
            st.dataframe(pd.DataFrame(results), hide_index=True)
            
            # Calculate Average
            valid_vals = [r[f"Top {cutoff_rank}"] for r in results if isinstance(r[f"Top {cutoff_rank}"], (int, float))]
            if valid_vals:
                avg = sum(valid_vals) / len(valid_vals)
                st.success(f"Moyenne Top {cutoff_rank}: **{avg:.2f}**")

# --- MAIN EXECUTION ---
scraper = FFAScraper()
load_css()

# 1. Sidebar Navigation
nav_mode = render_sidebar()

if nav_mode == "CLASSEMENTS":
    # 2. Filters (Top)
    annee, epreuve_label, epreuve_code, genre_label, genre_code, cat_label, cat_code = render_filters_horizontal()
    
    # 3. Header
    render_header(annee, epreuve_label, genre_label, cat_label)
    
    # 4. Content
    if 'df_results' in st.session_state and not st.session_state['df_results'].empty:
        df = st.session_state['df_results']
        
        # Determine Cutoff (Indoor/Outdoor)
        is_indoor = "Salle" in epreuve_label
        cutoff = 24 if is_indoor else 32
        
        # Get Cutoff Value for Status Bar
        cutoff_val = "N/A"
        if len(df) >= cutoff:
            cutoff_val = f"{df.iloc[cutoff-1]['Chrono']:.2f}s"
            
        render_status_bar(f"Top {cutoff} : {cutoff_val}", len(df))
        render_results_split(df, cutoff)
        
        # Pagination Button
        if st.button("Charger plus de résultats ⬇️", use_container_width=True):
            current_page = st.session_state.get('current_page', 1)
            next_page = current_page + 1
            
            with st.spinner(f"Chargement page {next_page}..."):
                new_df = scraper.get_rankings(epreuve_code, annee, genre_code, cat_code, page=next_page)
                
                if not new_df.empty:
                    # Append and re-rank
                    combined_df = pd.concat([st.session_state['df_results'], new_df], ignore_index=True)
                    # Remove duplicates just in case
                    combined_df = combined_df.drop_duplicates(subset=['Athlète', 'Perf', 'Lieu', 'Date'])
                    # Re-calculate rank
                    combined_df['Rang'] = combined_df['Chrono'].rank(method='min').astype(int)
                    
                    st.session_state['df_results'] = combined_df
                    st.session_state['current_page'] = next_page
                    st.rerun()
                else:
                    st.warning("Plus de résultats disponibles.")
        
    else:
        st.info("Cliquez sur ACTUALISER pour charger les résultats.")

elif nav_mode == "ATHLÈTES":
    st.title("Espace Athlètes")
    
    # Initialize Athlete Index
    athlete_index = AthleteIndex()
    athlete_index.load_index()
    
    # Search UI
    col_search, col_btn = st.columns([3, 1])
    with col_search:
        # Get all athlete names for autocomplete
        all_athletes = list(athlete_index.index.keys())
        selected_athlete_name = st.selectbox(
            "🔍 Rechercher un athlète", 
            options=[""] + all_athletes,
            format_func=lambda x: x if x else "Tapez un nom...",
            help="Tapez le nom d'un athlète pour afficher son profil"
        )
    
    # Tabs
    tab_profil, tab_duel = st.tabs(["👤 Fiche & Records", "⚔️ Duel (Head-to-Head)"])
    
    with tab_profil:
        current_genre = st.session_state.get('current_genre_code', 'M')
        
        # If an athlete is selected from search, use their URL
        search_url = None
        if selected_athlete_name:
            search_url = athlete_index.search(selected_athlete_name)
            
        render_profile_tab(current_genre, athlete_url=search_url, athlete_name=selected_athlete_name)
        
    with tab_duel:
        current_genre = st.session_state.get('current_genre_code', 'M')
        render_duel_tab(current_genre)

elif nav_mode == "BOÎTE À OUTILS":
    render_tools_main()
