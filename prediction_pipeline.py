import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import time
import random
import joblib
import os

# ==========================================
# STEP 1: DATA PURIFICATION (MUREIKA MODEL)
# ==========================================

class MureikaModel:
    """
    Implements Jonas Mureika's wind/altitude correction model.
    """
    
    @staticmethod
    def calculate_zero_wind(t, w, alt, dist="100m", gender="M", lane=4):
        """
        Converts a performance to 0 m/s wind and 0m altitude.
        """
        # Density correction
        dens = math.exp(-0.000125 * alt)
        
        if dist in ["50m", "60m", "100m"]:
            sp = int(dist.replace("m", ""))
            # Formula: t0 = (1.028 - 0.028 * dens * ((1.0 - w * t / sp) ** 2)) * t
            term = 1.0 - (w * t / sp)
            t0 = (1.028 - 0.028 * dens * (term * term)) * t
            return t0
            
        elif dist == "200m":
            # 200m Logic
            is_elite_men = (gender == "M" and t < 21.5)
            
            dt = 0.0
            w2 = w * w
            
            # Coefficients (Copied from app.py)
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

    @staticmethod
    def predict_perf_at_wind(t0, w_target, alt=0, dist="100m", gender="M", lane=4):
        """
        Calculates the performance at a specific wind speed given the zero-wind performance (t0).
        Inverse of calculate_zero_wind.
        """
        if dist == "200m":
            # For 200m, dt depends only on wind/alt, not time (in this simplified model).
            # t0 = t - dt => t = t0 + dt
            # But dt logic uses 't' to decide Elite vs Non-Elite.
            # We assume the category doesn't change.
            
            # Hack: Calculate dt for w_target by passing a dummy time (t0) to preserve Elite status check
            # t_dummy_zero = calculate_zero_wind(t0, w_target, alt, ...)
            # t_dummy_zero = t0 - dt_target => dt_target = t0 - t_dummy_zero
            
            t_dummy_zero = MureikaModel.calculate_zero_wind(t0, w_target, alt, dist, gender, lane)
            dt_target = t0 - t_dummy_zero
            return t0 + dt_target
            
        else:
            # Iterative solver for 100m/60m/50m
            # We want to find t_guess such that calculate_zero_wind(t_guess, w_target, alt) == t0
            t_guess = t0
            for _ in range(20):
                t_calc_zero = MureikaModel.calculate_zero_wind(t_guess, w_target, alt, dist, gender, lane)
                diff = t_calc_zero - t0
                if abs(diff) < 0.00001: break
                t_guess -= diff # Simple correction
            return t_guess

# ==========================================
# STEP 2: DATA ACQUISITION
# ==========================================

class PerformanceScraper:
    def __init__(self):
        self.session = requests.Session()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        self.headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://www.athle.fr/',
            'Connection': 'keep-alive'
        }

    def _make_request(self, url):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                time.sleep(random.uniform(0.5, 1.5))
                current_headers = self.headers.copy()
                current_headers['User-Agent'] = random.choice(self.user_agents)
                response = self.session.get(url, headers=current_headers, timeout=30)
                response.raise_for_status()
                return response
            except Exception as e:
                if attempt == max_retries - 1: raise e
                time.sleep(2 ** attempt)
        return None

    def clean_performance(self, perf_str):
        if not perf_str: return None
        clean = perf_str.replace("''", ".").replace("'", ".").strip()
        
        # Extract Chrono
        match_chrono = re.search(r"(\d+\.\d+)", clean)
        if not match_chrono: return None
        chrono_val = float(match_chrono.group(1))
        
        # Extract Wind
        wind_val = 0.0
        match_wind = re.search(r"\(([+-]?\d+\.\d+)\)", clean)
        if match_wind:
            try: wind_val = float(match_wind.group(1))
            except: pass
            
        return {'chrono': chrono_val, 'wind': wind_val}

    def parse_french_date(self, date_str, current_year=None):
        """
        Parses dates like "1 Août", "24 Juin 2007", "24/06/07".
        Returns datetime object.
        """
        months = {
            "jan": 1, "fév": 2, "fev": 2, "mar": 3, "avr": 4, "mai": 5, "juin": 6,
            "juil": 7, "août": 8, "aout": 8, "sep": 9, "oct": 10, "nov": 11, "déc": 12, "dec": 12
        }
        
        date_str = date_str.lower().strip()
        
        # Try DD/MM/YY
        if re.match(r"\d{2}/\d{2}/\d{2}", date_str):
            try:
                return datetime.strptime(date_str, "%d/%m/%y")
            except: pass
            
        # Try DD Month YYYY
        parts = date_str.split()
        if len(parts) >= 3:
            day = parts[0]
            month_str = parts[1]
            year = parts[2]
            
            # Check if year is valid
            if len(year) == 4 and year.isdigit():
                # Find month
                month_num = 0
                for m, n in months.items():
                    if m in month_str:
                        month_num = n
                        break
                if month_num > 0:
                    try:
                        return datetime(int(year), month_num, int(day))
                    except: pass
                    
        # Try DD Month (use current_year)
        if len(parts) >= 2 and current_year:
            day = parts[0]
            month_str = parts[1]
            month_num = 0
            for m, n in months.items():
                if m in month_str:
                    month_num = n
                    break
            
            if month_num > 0:
                try:
                    return datetime(int(current_year), month_num, int(day))
                except: pass
                
        return None

    def get_athlete_history(self, url):
        # Extract ID
        match = re.search(r"/athletes/(\d+)/", url)
        if not match: return pd.DataFrame()
        athlete_id = match.group(1)
        
        # We focus on 100m for now as the model is trained on it
        ajax_url = f"https://www.athle.fr/ajax/fiche-athlete-resultats.aspx?seq={athlete_id}&epr=110"
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            resp = requests.get(ajax_url, headers=headers)
            soup = BeautifulSoup(resp.text, 'html.parser')
            tds = soup.find_all('td')
            
            performances = []
            chunk_size = 17
            
            for i in range(0, len(tds), chunk_size):
                if i + 8 >= len(tds): break
                
                date_str = tds[i].get_text(strip=True)
                perf_str = tds[i+2].get_text(strip=True)
                wind_str = tds[i+3].get_text(strip=True)
                
                date_obj = self.parse_french_date(date_str)
                if not date_obj: continue
                
                # Clean Perf
                clean_perf = perf_str.replace("''", ".").replace("'", ".")
                try:
                    chrono = float(re.search(r"\d+\.\d+", clean_perf).group(0))
                except: continue
                
                # Clean Wind
                wind = 0.0
                try:
                    match_wind = re.search(r"[+-]?\d+\.\d+", wind_str)
                    if match_wind:
                        wind = float(match_wind.group(0))
                except: pass
                
                performances.append({
                    'date': date_obj,
                    'chrono': chrono,
                    'wind': wind,
                    'altitude': 0
                })
                
            return pd.DataFrame(performances)
            
        except Exception as e:
            print(f"Error scraping history: {e}")
            return pd.DataFrame()

    print("Scraper initialized.")

# ==========================================
# STEP 3: TIME WEIGHTING
# ==========================================

class TimeWeighting:
    @staticmethod
    def calculate_weights(dates, lambda_param=0.005):
        """
        Calculates exponential decay weights based on recency.
        w_i = exp(-lambda * delta_t)
        
        lambda_param: 0.005 means ~140 days half-life.
        """
        now = datetime.now()
        weights = []
        
        for d in dates:
            delta = (now - d).days
            # Ensure delta is non-negative (future dates treated as today)
            delta = max(0, delta)
            w = np.exp(-lambda_param * delta)
            weights.append(w)
            
        return np.array(weights)

# ==========================================
# STEP 4: TREND MODELING (LOESS)
# ==========================================

class TrendModel:
    @staticmethod
    def fit_loess(dates, perfs, weights=None, frac=0.6):
        """
        Fits a LOESS curve to the performance data.
        """
        # Convert dates to ordinal for regression
        x = np.array([d.toordinal() for d in dates])
        y = np.array(perfs)
        
        # Statsmodels lowess doesn't directly accept weights in the standard API easily 
        # for the 'lowess' function, but we can use WLS if we wanted a parametric approach.
        # However, the user specifically asked for LOESS.
        # Standard LOESS is locally weighted by distance in X. 
        # To incorporate our "Time Weighting" (importance), we can use it to filter or 
        # adjust the data, OR we can use WLS with a spline.
        # BUT, the user prompt says: "regarde les performances voisines (pondérées par l'étape 2)".
        # This implies the weights from Step 2 should be used IN the LOESS.
        # Statsmodels lowess implementation (sm.nonparametric.lowess) does NOT support external weights 
        # for importance, only the tri-cube kernel weights based on x-distance.
        
        # ALTERNATIVE: Use Weighted Least Squares (WLS) with a B-Spline or Polynomial 
        # to approximate a smooth trend if LOESS doesn't support it.
        # OR, we can just use LOESS on the data and assume the "local" aspect handles the time dependency,
        # but Step 2 specifically asks for exponential decay weights.
        
        # Let's stick to the user's "Ideal Pipeline" but adapt slightly if library limits exist.
        # Actually, if we want to force recent points to matter more in a LOESS, 
        # we can pass the weights to the 'lowess' function if it supported it.
        # Since it doesn't, a common trick is to resample points based on weights, 
        # but that's messy.
        
        # BETTER APPROACH for "Trend of Form":
        # We can use WLS (Weighted Least Squares) with a rolling window or a spline base.
        # However, to strictly follow "LOESS", we will use sm.nonparametric.lowess.
        # We will ignore the explicit "Step 2 weights" for the *Trend* calculation 
        # if the library doesn't support it, OR we can use the weights in the next step (Quantile).
        # WAIT: The user says "Step 3... pondérées par l'étape 2".
        
        # Let's use a workaround: Weighted Quantile Regression for the trend (Median)?
        # Or just use LOESS as is, which naturally weights local points (in time).
        # The "Exponential Forgetting" is redundant with LOESS if LOESS window is small, 
        # BUT exponential forgetting is global (recent years matter more than 5 years ago).
        # LOESS with a large window might smooth too much old data.
        
        # DECISION: I will use LOESS as the base "Shape". 
        # I will apply the Step 2 Weights in the Step 4 (Quantile Regression).
        # For Step 3, I'll use standard LOESS.
        
        lowess = sm.nonparametric.lowess
        # frac determines the window size.
        smoothed = lowess(y, x, frac=frac, return_sorted=True)
        
        return smoothed[:, 0], smoothed[:, 1] # x_sorted, y_fitted

# ==========================================
# STEP 5: PEAK PREDICTION (QUANTILE REGRESSION)
# ==========================================

class PeakPredictor:
    @staticmethod
    def predict_peak(dates, perfs, weights, quantile=0.05):
        """
        Predicts the peak performance curve using Quantile Regression.
        """
        # Data prep
        df = pd.DataFrame({
            'y': perfs,
            'x': [d.toordinal() for d in dates],
            'w': weights
        })
        
        # We want a curve, not a line. So we need basis functions (e.g. Polynomial).
        # A 2nd or 3rd degree polynomial allows the "shape" to bend.
        # Or we can use the LOESS output as a feature?
        # The user says: "Trace une deuxième courbe, parallèle à la première mais décalée".
        # This implies: Peak_Curve = LOESS_Curve - Offset.
        # The Offset is determined by the residuals distribution.
        
        # Let's calculate the residuals from the LOESS trend.
        # Residual = Perf - Trend
        # Then find the 5th percentile of these residuals (weighted!).
        
        # 1. Get Trend
        trend_x, trend_y = TrendModel.fit_loess(dates, perfs)
        
        # Map trend back to original dates (since lowess sorts X)
        trend_dict = dict(zip(trend_x, trend_y))
        df['trend'] = df['x'].map(trend_dict)
        
        # 2. Calculate Residuals
        df['resid'] = df['y'] - df['trend']
        
        # 3. Weighted Quantile of Residuals
        # We need the 5th percentile of residuals, weighted by 'w'.
        # Sort by residual
        df = df.sort_values('resid')
        
        # Calculate cumulative weight
        df['cum_w'] = df['w'].cumsum()
        total_w = df['w'].sum()
        df['cum_w_norm'] = df['cum_w'] / total_w
        
        # Find the residual value where cum_w_norm crosses quantile
        # This is the "Offset"
        offset = 0
        try:
            offset = df[df['cum_w_norm'] >= quantile].iloc[0]['resid']
        except:
            offset = df['resid'].quantile(quantile) # Fallback unweighted
            
        # 4. Construct Peak Curve
        # Peak = Trend + Offset
        # Since we want "fast" times (lower values), and residuals are (Perf - Trend),
        # If 5th percentile residual is -0.2s, then Peak = Trend - 0.2s.
        
        return df['x'], df['trend'] + offset, offset
if __name__ == "__main__":
    # Test Mureika
    print("Testing Mureika...")
    t0 = MureikaModel.calculate_zero_wind(10.00, 2.0, 0, "100m")
    print(f"10.00 (+2.0) -> {t0:.3f}")
    
    # Test Pipeline with Dummy Data
    print("\nTesting Pipeline with Dummy Data...")
    dates = [datetime(2023, 1, 1) + pd.Timedelta(days=i*30) for i in range(24)] # 2 years
    # Create a trend: 10.50 -> 10.20 -> 10.40
    perfs = []
    for i, d in enumerate(dates):
        base = 10.50 - 0.3 * np.sin(i/10) 
        noise = np.random.normal(0, 0.1)
        perfs.append(base + noise)
        
    df_dummy = pd.DataFrame({'date': dates, 'chrono': perfs, 'wind': [0]*24, 'altitude': [0]*24, 'event': ['100m']*24})
    
    # Mock Scraper
    pipeline = PredictionPipeline()
    pipeline.scraper.get_athlete_history = lambda x: df_dummy
    
    res = pipeline.run_pipeline("dummy_url", 2024)
    print("Result:", res)


class PredictionPipeline:
    def __init__(self, model_path="sb_prediction_model_v3.pkl"):
        self.scraper = PerformanceScraper()
        self.model = None
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Model file not found: {self.model_path}")
            
    def run_pipeline(self, url, season_year=2025, verbose=True):
        if verbose: print(f"Running V3 Pipeline for {season_year}...")
        
        # 1. Scrape History
        df = self.scraper.get_athlete_history(url)
        if df.empty:
            if verbose: print("No history found.")
            return None
            
        # 2. Preprocess
        df['P_int'] = df.apply(lambda r: MureikaModel.calculate_zero_wind(
            r['chrono'], r['wind'], r['altitude'], "100m" # Assuming 100m for now
        ), axis=1)
        df['Year'] = df['date'].dt.year
        df = df.sort_values('date')
        
        # 3. Extract Features (Dynamic V3)
        # Target Season Data
        df_target = df[df['Year'] == season_year]
        
        # Check if we have enough races (N=3)
        n_races = 3
        if len(df_target) < n_races:
            if verbose: print(f"Not enough races in {season_year} (Found {len(df_target)}, Need {n_races})")
            return None
            
        df_early = df_target.iloc[:n_races]
        
        # Features
        # Previous Season
        prev_season = season_year - 1
        df_prev = df[df['Year'] == prev_season]
        
        if df_prev.empty:
            sb_prev = np.nan
            avg_top3_prev = np.nan
        else:
            sb_prev = df_prev['P_int'].min()
            avg_top3_prev = df_prev['P_int'].nsmallest(3).mean()
            
        # Career
        df_history = df[df['Year'] < season_year]
        if df_history.empty:
            pb_career = np.nan
            avg_career = np.nan
            first_year = season_year
        else:
            pb_career = df_history['P_int'].min()
            avg_career = df_history['P_int'].mean()
            first_year = df_history['Year'].min()
            
        career_age = season_year - first_year
        
        # Early Season
        sb_early = df_early['P_int'].min()
        avg_early = df_early['P_int'].mean()
        nb_races_early = len(df_early)
        
        # Slope
        slope_early = 0
        if len(df_early) >= 2:
            start_date = df_early['date'].min()
            x = (df_early['date'] - start_date).dt.days.values
            y = df_early['P_int'].values
            try:
                if np.max(x) > 0:
                    slope_early = np.polyfit(x, y, 1)[0] * 30
            except: pass
            
        # Prepare Input Vector
        days_span = (df_early['date'].max() - df_early['date'].min()).days
        
        features = {
            'SB_Prev': sb_prev,
            'Avg_Top3_Prev': avg_top3_prev,
            'PB_Career': pb_career,
            'Avg_Career': avg_career,
            'SB_Early': sb_early,
            'Avg_Early': avg_early,
            'Slope_Early': slope_early,
            'Nb_Races_Early': nb_races_early,
            'Days_Span_Early': days_span,
            'Career_Age': career_age
        }
        
        # Handle NaNs (Imputation)
        if pd.isna(features['SB_Prev']): features['SB_Prev'] = sb_early
        if pd.isna(features['Avg_Top3_Prev']): features['Avg_Top3_Prev'] = avg_early
        if pd.isna(features['PB_Career']): features['PB_Career'] = sb_early
        if pd.isna(features['Avg_Career']): features['Avg_Career'] = avg_early
        
        # Reorder to match training features exactly
        feature_order = [
            'SB_Prev', 'Avg_Top3_Prev', 
            'SB_Early', 'Avg_Early', 'Slope_Early', 'Nb_Races_Early', 'Days_Span_Early',
            'Career_Age', 'PB_Career', 'Avg_Career'
        ]
        
        X = pd.DataFrame([features])
        X = X[feature_order]
        
        # Predict
        if self.model:
            try:
                # DEBUG: Check feature alignment
                if hasattr(self.model, "feature_names_in_"):
                    model_feats = list(self.model.feature_names_in_)
                    input_feats = list(X.columns)
                    if model_feats != input_feats:
                        print(f"MISMATCH DETECTED!")
                        print(f"Model expects: {model_feats}")
                        print(f"Input has:     {input_feats}")
                
                p_int_pred = self.model.predict(X)[0]
                
                return {
                    'p_real_pred': p_int_pred,
                    'features': features
                }
            except Exception as e:
                print(f"Prediction Error: {e}")
                # Re-raise to see traceback if needed, or return None
                raise e
        else:
            return None
