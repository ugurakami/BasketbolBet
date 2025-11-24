import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import requests
import os

# =================================================================
#                         1. KONFİGÜRASYONLAR
# =================================================================

# --- Veri Yolu ---
FILE_NAME = "BasketbolFikstür - Sayfa1.tsv" 
# NOT: NBA verilerinizi bu dosyaya "Lig: NBA" sütunuyla eklemiş olmalısınız.

# --- Telegram Ayarları ---
# Not: GitHub Actions kullanıyorsanız, bu değerler ortam değişkenlerinden (secrets) alınacaktır.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN") 
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

# --- Kelly Kriteri & Model Ayarları ---
VARSAYILAN_ORAN = 1.90         # Simülasyon için kullanılan sabit oran (Gerçekte bu oran dışarıdan gelmeli)
P_ESIGI = 0.55                 # Bahis alınabilmesi için modelin minimum olasılık eşiği
MODEL_ACC_ESIGI = 0.55         # Modelin kabul edilebilir minimum Çapraz Doğrulama Accuracy değeri
N_MAC = 5                      # Form ve H2H hesaplaması için kullanılan son maç sayısı

# =================================================================
#                         2. YARDIMCI FONKSİYONLAR
# =================================================================

def kelly_criterion(p, b):
    """Kelly Kriteri hesaplaması: f* = (b*p - q) / b"""
    q = 1 - p
    if (b * p - q) <= 0:
        return 0.0
    return (b * p - q) / b

def degerli_bahisleri_sec(tahmin_df, odds_varsayimi, p_esigi):
    degerli_bahisler = []
    b = odds_varsayimi - 1
    
    for index, row in tahmin_df.iterrows():
        # Taraf Tahminleri
        for bahis_tipi, p_col, secim in [
            ('Taraf', 'P_Ev', f"{row['Ev_Sahibi']} Kazanır"),
            ('Taraf', 'P_Dep', f"{row['Deplasman']} Kazanır")
        ]:
            if row[p_col] > p_esigi:
                f_kelly = kelly_criterion(row[p_col], b)
                if f_kelly > 0.001: # %0.1'den büyük kelly payı olanları al
                    degerli_bahisler.append({'Maç': f"{row['Ev_Sahibi']} vs {row['Deplasman']} ({row['Lig']})", 'Bahis_Turu': bahis_tipi, 'Seçim': secim, 'Model_Olasilik': row[p_col], 'Varsayilan_Oran': odds_varsayimi, 'Kelly_Payi_Yuzde': f_kelly * 100})
        
        # Sayı Limiti (Alt/Üst) Tahminleri
        limit = int(row['Limit_Cizgisi'])
        for bahis_tipi, p_col, secim in [
            ('Sayı Limiti', 'P_Ust', f"Üst {limit}"),
            ('Sayı Limiti', 'P_Alt', f"Alt {limit}")
        ]:
            if row[p_col] > p_esigi:
                f_kelly = kelly_criterion(row[p_col], b)
                if f_kelly > 0.001:
                    degerli_bahisler.append({'Maç': f"{row['Ev_Sahibi']} vs {row['Deplasman']} ({row['Lig']})", 'Bahis_Turu': bahis_tipi, 'Seçim': secim, 'Model_Olasilik': row[p_col], 'Varsayilan_Oran': odds_varsayimi, 'Kelly_Payi_Yuzde': f_kelly * 100})
    
    return pd.DataFrame(degerli_bahisler)

def telegram_mesaj_gonder(df_kelly, tarih, is_model_acceptable_flag):
    
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN" or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID":
        print("Telegram bot bilgileri ayarlanmadı. Mesaj gönderme atlanıyor.")
        return

    mesaj = f"📅 *{tarih} Tarihli Basketbol Analizi*\n\n"

    if not is_model_acceptable_flag:
        mesaj += "❌ *KRİTİK HATA:* Model Accuracy eşiği sağlanamadı. Güvenliğiniz için bahis önlenmiştir! 🚨"
    elif df_kelly.empty:
        mesaj += "🚫 Kelly Kriterine göre pozitif beklenen değere sahip değerli bir bahis bulunamamıştır."
    else:
        mesaj += "💰 *Kelly Kriterine Göre Değerli Bahis Önerileri (Fractional Kelly)*\n\n"
        
        for index, row in df_kelly.iterrows():
            mesaj += f"🏀 *Maç:* {row['Maç']}\n"
            mesaj += f"   - *Seçim:* {row['Seçim']}\n"
            mesaj += f"   - *Model P:* %{row['Model_Olasilik']:.1%}\n"
            mesaj += f"   - *Kelly Payı:* %{row['Kelly_Payi_Yuzde']:.1f} (Risk Sınırı)\n"
            mesaj += f"   - *Varsayılan Oran:* {row['Varsayilan_Oran']:.2f}\n"
            mesaj += "--------------------------\n"

    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': mesaj, 'parse_mode': 'Markdown'}
    
    try:
        response = requests.post(TELEGRAM_API_URL, data=payload)
        response.raise_for_status()
        print(f"✅ Telegram mesajı başarıyla gönderildi.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Telegram mesajı gönderme hatası: {e}")

def is_model_acceptable(cv_scores, threshold):
    """Model accuracy kabul edilebilir mi?"""
    return cv_scores.mean() > threshold

# =================================================================
#                         3. GÜVENLİ KODLAMA VE ÖZELLİK MÜHENDİSLİĞİ
# =================================================================

def safe_label_encode(train_series, test_series):
    """Bilinmeyen değerleri -1 (Unknown) olarak işleyen güvenli Label Encoding."""
    le = LabelEncoder()
    train_encoded = le.fit_transform(train_series.astype(str))
    
    test_encoded = []
    for val in test_series.astype(str):
        try:
            test_encoded.append(le.transform([val])[0])
        except ValueError:
            test_encoded.append(-1) # Bilinmeyen değer
            
    return train_encoded, np.array(test_encoded)


def calculate_dynamic_features(df_input, n_mac):
    """Dinlenme Günü, Form ve H2H özelliklerini hesaplar (Leakage Safe)"""
    df_temp = df_input.copy()
    
    # 1. Dinlenme Günü Hesaplama (Shift(1) ile geçmişe bakılır)
    all_matches_home = df_temp[['Tarih', 'Ev_Sahibi']].rename(columns={'Ev_Sahibi': 'Takım'})
    all_matches_away = df_temp[['Tarih', 'Deplasman']].rename(columns={'Deplasman': 'Takım'})
    all_matches = pd.concat([all_matches_home, all_matches_away]).sort_values('Tarih').reset_index(drop=True)

    all_matches['Onceki_Tarih'] = all_matches.groupby('Takım')['Tarih'].shift(1)
    all_matches['Dinlenme_Gunu'] = (all_matches['Tarih'] - all_matches['Onceki_Tarih']).dt.days

    rest_days_home = all_matches[['Tarih', 'Takım', 'Dinlenme_Gunu']].rename(columns={'Takım': 'Ev_Sahibi', 'Dinlenme_Gunu': 'Ev_Dinlenme_Gunu'})
    df_temp = pd.merge(df_temp, rest_days_home, on=['Tarih', 'Ev_Sahibi'], how='left', suffixes=('_x', ''))

    rest_days_away = all_matches[['Tarih', 'Takım', 'Dinlenme_Gunu']].rename(columns={'Takım': 'Deplasman', 'Dinlenme_Gunu': 'Dep_Dinlenme_Gunu'})
    df_temp = pd.merge(df_temp, rest_days_away, on=['Tarih', 'Deplasman'], how='left', suffixes=('_x', ''))

    df_temp['Ev_Dinlenme_Gunu'] = df_temp['Ev_Dinlenme_Gunu'].fillna(7)
    df_temp['Dep_Dinlenme_Gunu'] = df_temp['Dep_Dinlenme_Gunu'].fillna(7)
    df_temp['Dinlenme_Gunu_Farki'] = df_temp['Ev_Dinlenme_Gunu'] - df_temp['Dep_Dinlenme_Gunu']
    
    # 2. Form ve H2H Hesaplama (Sadece skoru bilinen maçlara bakılır)
    df_temp['Kazanan_Taraf'] = (df_temp['MS_Ev'] > df_temp['MS_Dep']).astype('float').fillna(-1)
    df_gecmis_local = df_temp[df_temp['MS_Ev'].notnull()].copy()

    def hesapla_ozel_form(takım, tarih, ev_mi, n):
        if ev_mi:
            maclar = df_gecmis_local[(df_gecmis_local['Ev_Sahibi'] == takım) & (df_gecmis_local['Tarih'] < tarih)].tail(n)
            galibiyet_sayisi = (maclar['Kazanan_Taraf'] == 1).sum()
        else:
            maclar = df_gecmis_local[(df_gecmis_local['Deplasman'] == takım) & (df_gecmis_local['Tarih'] < tarih)].tail(n)
            galibiyet_sayisi = (maclar['Kazanan_Taraf'] == 0).sum()
        return galibiyet_sayisi / len(maclar) if len(maclar) > 0 else 0.5

    df_temp['Ev_Sahibi_Ev_Formu'] = df_temp.apply(
        lambda row: hesapla_ozel_form(row['Ev_Sahibi'], row['Tarih'], True, n_mac), axis=1)
    df_temp['Dep_Takim_Dep_Formu'] = df_temp.apply(
        lambda row: hesapla_ozel_form(row['Deplasman'], row['Tarih'], False, n_mac), axis=1)

    def hesapla_h2h_rekoru(ev_takimi, dep_takimi, tarih, n):
        h2h_maclar = df_gecmis_local[
            ((df_gecmis_local['Ev_Sahibi'] == ev_takimi) & (df_gecmis_local['Deplasman'] == dep_takimi)) |
            ((df_gecmis_local['Ev_Sahibi'] == dep_takimi) & (df_gecmis_local['Deplasman'] == ev_takimi))
        ].query('Tarih < @tarih').tail(n)
        
        if len(h2h_maclar) == 0: return 0.5
        
        ev_kazanma_sayisi = h2h_maclar.apply(
            lambda row: 1 if (row['Ev_Sahibi'] == ev_takimi and row['Kazanan_Taraf'] == 1) or 
                              (row['Deplasman'] == ev_takimi and row['Kazanan_Taraf'] == 0) 
                         else 0, axis=1
        ).sum()
        
        return ev_kazanma_sayisi / len(h2h_maclar)

    df_temp['H2H_Rekor'] = df_temp.apply(
        lambda row: hesapla_h2h_rekoru(row['Ev_Sahibi'], row['Deplasman'], row['Tarih'], n_mac), axis=1)
        
    return df_temp

def prepare_data_pipeline(df_input, n_mac):
    """Data Leakage'i tamamen önleyen ana pipeline."""
    
    df_train = df_input[df_input['MS_Ev'].notnull()].copy()
    df_predict = df_input[df_input['MS_Ev'].isnull()].copy()

    # Limit Çizgisi SADECE TRAIN'den Hesaplama
    limit_cizgisi = df_train['Toplam_Skor'].median()
    df_train['Toplam_Skor_Ust'] = (df_train['Toplam_Skor'] > limit_cizgisi).astype('float')

    # Özellik Hesaplama için Train ve Predict verisini birleştirme (Dinlenme Günü için kritik)
    df_full_sorted = pd.concat([df_train, df_predict], ignore_index=True).sort_values(by='Tarih').reset_index(drop=True)
    df_full_featured = calculate_dynamic_features(df_full_sorted, n_mac)
    
    # Sonuçları tekrar ayır
    df_train_featured = df_full_featured[df_full_featured['MS_Ev'].notnull()].copy()
    df_predict_featured = df_full_featured[df_full_featured['MS_Ev'].isnull()].copy()

    # Kategorik Kodlama
    FEATURE_COLS = ['Dinlenme_Gunu_Farki', 'Ev_Sahibi_Ev_Formu', 'Dep_Takim_Dep_Formu', 'H2H_Rekor', 'Ev_Sahibi', 'Deplasman', 'Lig']
    df_train_featured = df_train_featured.dropna(subset=FEATURE_COLS).copy()
    
    X_train = df_train_featured[FEATURE_COLS].copy()
    X_predict = df_predict_featured[FEATURE_COLS].copy()


    for col in ['Ev_Sahibi', 'Deplasman', 'Lig']:
        X_train_encoded, X_predict_encoded = safe_label_encode(X_train[col], X_predict[col])
        X_train.loc[:, col] = X_train_encoded
        X_predict.loc[:, col] = X_predict_encoded
        
    X_predict = X_predict.dropna().copy()
    
    return X_train, X_predict, df_train_featured, df_predict_featured, limit_cizgisi

# =================================================================
#                         4. ANA ÇALIŞTIRMA FONKSİYONU
# =================================================================

def main():
    try:
        df = pd.read_csv(FILE_NAME, sep='\t')
    except FileNotFoundError:
        print(f"HATA: '{FILE_NAME}' dosyası bulunamadı.")
        return

    # Veri Temizleme ve Dönüştürme
    df = df.rename(columns={'MS(Ev)': 'MS_Ev', 'MS(Dep)': 'MS_Dep', 'İY(Ev)': 'IY_Ev', 'İY(Dep)': 'IY_Dep', 'Ev Sahibi': 'Ev_Sahibi', 'Deplasman': 'Deplasman'})
    df['MS_Ev'] = pd.to_numeric(df['MS_Ev'], errors='coerce')
    df['MS_Dep'] = pd.to_numeric(df['MS_Dep'], errors='coerce')
    df['Tarih'] = pd.to_datetime(df['Tarih'], format='%d.%m.%Y')
    df = df.sort_values(by='Tarih').reset_index(drop=True)
    df['Toplam_Skor'] = df['MS_Ev'] + df['MS_Dep']
    df['Kazanan_Taraf'] = (df['MS_Ev'] > df['MS_Dep']).astype('float').fillna(-1)

    # 1. Data Pipeline'ı Uygula
    X_train, X_predict, df_train_featured, df_predict_featured, limit_cizgisi = prepare_data_pipeline(df.copy(), N_MAC)

    y_taraf = df_train_featured['Kazanan_Taraf']
    y_limit = df_train_featured['Toplam_Skor_Ust']
    
    # 2. Model Eğitimi ve Kalite Kontrolü
    tscv = TimeSeriesSplit(n_splits=5)
    
    # --- Taraf Modeli ---
    model_taraf = RandomForestClassifier(n_estimators=100, random_state=42)
    taraf_scores = cross_val_score(model_taraf, X_train, y_taraf, cv=tscv, scoring='accuracy')
    taraf_acc_ok = is_model_acceptable(taraf_scores, MODEL_ACC_ESIGI)
    print(f"\nModel 1 (Taraf) Çapraz Doğrulama (Accuracy): {taraf_scores.mean():.2f}")
    
    # --- Limit Modeli ---
    model_limit = RandomForestClassifier(n_estimators=100, random_state=42)
    limit_scores = cross_val_score(model_limit, X_train, y_limit, cv=tscv, scoring='accuracy')
    limit_acc_ok = is_model_acceptable(limit_scores, MODEL_ACC_ESIGI)
    print(f"Model 2 (Limit) Çapraz Doğrulama (Accuracy): {limit_scores.mean():.2f}")

    df_kelly = pd.DataFrame() # Varsayılan boş dataframe

    if taraf_acc_ok and limit_acc_ok:
        # Modelleri tüm train seti üzerinde yeniden eğit
        model_taraf.fit(X_train, y_taraf)
        model_limit.fit(X_train, y_limit)
        
        # Tahminleri Yapma
        proba_taraf = model_taraf.predict_proba(X_predict)
        proba_limit = model_limit.predict_proba(X_predict)

        df_predict_featured.loc[:, 'P_Ev'] = proba_taraf[:, 1]
        df_predict_featured.loc[:, 'P_Dep'] = proba_taraf[:, 0]
        df_predict_featured.loc[:, 'P_Ust'] = proba_limit[:, 1]
        df_predict_featured.loc[:, 'P_Alt'] = proba_limit[:, 0]
        df_predict_featured.loc[:, 'Limit_Cizgisi'] = limit_cizgisi

        # Kelly Kriterini Uygulama (Fractional)
        bugunun_tarihi = df_predict_featured['Tarih'].min()
        tahmin_sonuclari_bugun = df_predict_featured[df_predict_featured['Tarih'] == bugunun_tarihi].copy()

        df_kelly_full = degerli_bahisleri_sec(tahmin_sonuclari_bugun, VARSAYILAN_ORAN, P_ESIGI)
        
        # Fractional Kelly ve Bankroll %1 Sınırı (Risk Yönetimi)
        if not df_kelly_full.empty:
            df_kelly_full['Kelly_Payi_Yuzde'] = df_kelly_full['Kelly_Payi_Yuzde'].apply(
                lambda x: min(x * 0.25, 1.0) # Fractional (25%)
            )
            df_kelly = df_kelly_full[df_kelly_full['Kelly_Payi_Yuzde'] > 0.001]
        else:
            df_kelly = pd.DataFrame()

        print(f"\n--- Analiz Tarihi: {bugunun_tarihi.strftime('%Y-%m-%d')} ---")
        if df_kelly.empty:
            print("Kelly kriterine göre pozitif değere sahip bahis bulunamadı. 🚫")
        else:
            print(df_kelly.to_markdown(index=False, floatfmt=".2f"))
    
    else:
        bugunun_tarihi = df_predict_featured['Tarih'].min()
        print("Model Accuracy eşiği sağlanamadığı için tahmin ve Kelly kriteri atlanmıştır.")

    # Telegram Mesajını Gönderme (Model başarısız olsa bile bilgi verir)
    telegram_mesaj_gonder(df_kelly, bugunun_tarihi.strftime('%Y-%m-%d'), taraf_acc_ok and limit_acc_ok)


if __name__ == "__main__":
    main()
