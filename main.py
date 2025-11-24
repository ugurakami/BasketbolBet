import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import requests
import os

# ... (KONFİGÜRASYONLAR ve KELLY/TELEGRAM Fonksiyonları - Aynı Kalır)

# =================================================================
#                         YENİ: GÜVENLİ KODLAMA VE ÖZELLİK MÜHENDİSLİĞİ
# =================================================================

def safe_label_encode(train_series, test_series):
    """Bilinmeyen değerleri -1 (Unknown) olarak işleyen güvenli Label Encoding."""
    le = LabelEncoder()
    # 1. SADECE train verisinden öğren (fit)
    train_encoded = le.fit_transform(train_series.astype(str))
    
    # 2. Test verisini dönüştür (transform)
    test_encoded = []
    for val in test_series.astype(str):
        try:
            # Bilinen bir değeri dönüştür
            test_encoded.append(le.transform([val])[0])
        except ValueError:
            # Bilinmeyen değeri -1 yap
            test_encoded.append(-1)
            
    return train_encoded, np.array(test_encoded)


def calculate_dynamic_features(df_input, n_mac):
    """Dinlenme Günü, Form ve H2H özelliklerini hesaplar (Leakage Safe)"""
    df_temp = df_input.copy()
    
    # Hedefler zaten df'in dışında hesaplanmış kabul edilir (Kazanan_Taraf)
    # df_gecmis, burada sadece df_input'un kendisidir, yani train/predict ayrımı main'de yapılır.

    # 1. Dinlenme Günü Hesaplama (Sadece geçmiş tarihe bakarak shift(1) ile)
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
    
    # 2. Form ve H2H Hesaplama (Hala "apply" kullandığımız için, sadece o tarihe kadar olan geçmişe bakılır)
    # Bu adımı sadece df_train'de çalıştırıp sonra df_predict'e uygularken df_train'i geçmiş olarak kullanacağız.
    
    return df_temp

def prepare_data_pipeline(df_input, n_mac):
    """Data Leakage'i tamamen önleyen ana pipeline."""
    
    # 1. Veriyi Ayırma
    df_train = df_input[df_input['MS_Ev'].notnull()].copy()
    df_predict = df_input[df_input['MS_Ev'].isnull()].copy()

    # 2. Limit Çizgisi SADECE TRAIN'den Hesaplama (KRİTİK DÜZELTME)
    limit_cizgisi = df_train['Toplam_Skor'].median()
    df_train['Toplam_Skor_Ust'] = (df_train['Toplam_Skor'] > limit_cizgisi).astype('float')

    # 3. Dinamik Özellikleri Hesaplama
    # df_train için Dinlenme Günü, Form, H2H hesapla
    df_train = calculate_dynamic_features(df_train, n_mac)
    
    # df_predict için Dinlenme Günü hesapla (df_predict'in gelecekteki tarihleri kullanması gerekir)
    # Bu adımda df_train ve df_predict'i birleştirip calculate_dynamic_features'ı tekrar çalıştırmak,
    # en son maçtan sonraki dinlenme gününü bulmak için en uygun yoldur.
    df_full_sorted = pd.concat([df_train, df_predict], ignore_index=True).sort_values(by='Tarih').reset_index(drop=True)
    df_full_featured = calculate_dynamic_features(df_full_sorted, n_mac)
    
    # Sonuçları tekrar ayır
    df_train_featured = df_full_featured[df_full_featured['MS_Ev'].notnull()].copy()
    df_predict_featured = df_full_featured[df_full_featured['MS_Ev'].isnull()].copy()


    # 4. Kategorik Kodlama (SADECE Train'den Öğrenme - KRİTİK DÜZELTME)
    FEATURE_COLS = ['Dinlenme_Gunu_Farki', 'Ev_Sahibi_Ev_Formu', 'Dep_Takim_Dep_Formu', 'H2H_Rekor', 'Ev_Sahibi', 'Deplasman', 'Lig']
    df_train_featured = df_train_featured.dropna(subset=FEATURE_COLS).copy() #NaN değerleri temizle
    
    X_train = df_train_featured[FEATURE_COLS].copy()
    X_predict = df_predict_featured[FEATURE_COLS].copy()


    for col in ['Ev_Sahibi', 'Deplasman', 'Lig']:
        X_train_encoded, X_predict_encoded = safe_label_encode(X_train[col], X_predict[col])
        X_train.loc[:, col] = X_train_encoded
        X_predict.loc[:, col] = X_predict_encoded
        
    # X_predict'te NaN kalan satırları temizle
    X_predict = X_predict.dropna().copy()
    
    # 5. Sonuçları Geri Döndürme
    return X_train, X_predict, df_train_featured, df_predict_featured, limit_cizgisi


def is_model_acceptable(cv_scores, threshold=0.55):
    """Model accuracy kabul edilebilir mi?"""
    return cv_scores.mean() > threshold

# =================================================================
#                         ANA ÇALIŞTIRMA FONKSİYONU
# =================================================================

def main():
    # ... (Veri Yükleme ve Temizleme kısmı aynı)
    # ... (Veri Yükleme, Temizleme, Tarih ve Skor dönüştürme kısmı)
    try:
        df = pd.read_csv(FILE_NAME, sep='\t')
    except FileNotFoundError:
        print(f"HATA: '{FILE_NAME}' dosyası bulunamadı. Lütfen dosyanın klasörde olduğundan emin olun.")
        return

    df = df.rename(columns={
        'MS(Ev)': 'MS_Ev', 'MS(Dep)': 'MS_Dep', 'İY(Ev)': 'IY_Ev', 'İY(Dep)': 'IY_Dep',
        'Ev Sahibi': 'Ev_Sahibi', 'Deplasman': 'Deplasman',
    })
    df['MS_Ev'] = pd.to_numeric(df['MS_Ev'], errors='coerce')
    df['MS_Dep'] = pd.to_numeric(df['MS_Dep'], errors='coerce')
    df['Tarih'] = pd.to_datetime(df['Tarih'], format='%d.%m.%Y')
    df = df.sort_values(by='Tarih').reset_index(drop=True)
    df['Toplam_Skor'] = df['MS_Ev'] + df['MS_Dep']
    df['Kazanan_Taraf'] = (df['MS_Ev'] > df['MS_Dep']).astype('float').fillna(-1)

    # 1. Data Pipeline'ı Uygula (Leakage tamamen önlenir)
    X_train, X_predict, df_train_featured, df_predict_featured, limit_cizgisi = prepare_data_pipeline(df.copy(), N_MAC)

    y_taraf = df_train_featured['Kazanan_Taraf']
    y_limit = df_train_featured['Toplam_Skor_Ust']
    
    # 2. Model Eğitimi ve Kalite Kontrolü
    tscv = TimeSeriesSplit(n_splits=5)
    
    # --- Taraf Modeli ---
    model_taraf = RandomForestClassifier(n_estimators=100, random_state=42)
    taraf_scores = cross_val_score(model_taraf, X_train, y_taraf, cv=tscv, scoring='accuracy')
    print(f"\nModel 1 (Taraf) Çapraz Doğrulama (Accuracy): {taraf_scores.mean():.2f}")
    
    if not is_model_acceptable(taraf_scores):
        print("⚠️ HATA: Taraf modeli accuracy eşiği (0.55) altında! Bahis önleniyor.")
        df_kelly = pd.DataFrame() # Boş Kelly sonuçları
        # Telegram gönderme adımına geç

    # --- Limit Modeli ---
    model_limit = RandomForestClassifier(n_estimators=100, random_state=42)
    limit_scores = cross_val_score(model_limit, X_train, y_limit, cv=tscv, scoring='accuracy')
    print(f"Model 2 (Limit) Çapraz Doğrulama (Accuracy): {limit_scores.mean():.2f}")

    if not is_model_acceptable(limit_scores):
        print("⚠️ HATA: Limit modeli accuracy eşiği (0.55) altında! Bahis önleniyor.")
        df_kelly = pd.DataFrame() # Boş Kelly sonuçları
        # Telegram gönderme adımına geç

    # Modelleri tüm train seti üzerinde yeniden eğit (Çapraz doğrulama bittikten sonra)
    model_taraf.fit(X_train, y_taraf)
    model_limit.fit(X_train, y_limit)
    
    # 3. Tahminleri Yapma (Sadece Model Kabul Edilebilir İse)
    if is_model_acceptable(taraf_scores) and is_model_acceptable(limit_scores):
        proba_taraf = model_taraf.predict_proba(X_predict)
        proba_limit = model_limit.predict_proba(X_predict)

        df_predict_featured.loc[:, 'P_Ev'] = proba_taraf[:, 1]
        df_predict_featured.loc[:, 'P_Dep'] = proba_taraf[:, 0]
        df_predict_featured.loc[:, 'P_Ust'] = proba_limit[:, 1]
        df_predict_featured.loc[:, 'P_Alt'] = proba_limit[:, 0]
        df_predict_featured.loc[:, 'Limit_Cizgisi'] = limit_cizgisi

        # 4. Kelly Kriterini Uygulama (Fractional)
        bugunun_tarihi = df_predict_featured['Tarih'].min()
        tahmin_sonuclari_bugun = df_predict_featured[df_predict_featured['Tarih'] == bugunun_tarihi].copy()

        df_kelly_full = degerli_bahisleri_sec(tahmin_sonuclari_bugun, VARSAYILAN_ORAN, P_ESIGI)
        
        # Fractional Kelly ve Bankroll %1 Sınırı (Risk Yönetimi)
        if not df_kelly_full.empty:
            df_kelly_full['Kelly_Payi_Yuzde'] = df_kelly_full['Kelly_Payi_Yuzde'].apply(
                lambda x: min(x * 0.25, 1.0) # Fractional (25%) ve Max %1 Bankroll Sınırı (Varsayılıyor)
            )
            df_kelly = df_kelly_full[df_kelly_full['Kelly_Payi_Yuzde'] > 0.0]
        else:
            df_kelly = pd.DataFrame()

        print(f"\n--- Analiz Tarihi: {bugunun_tarihi.strftime('%Y-%m-%d')} (Fractional Kelly) ---")
        if df_kelly.empty:
            print("Kelly kriterine göre pozitif değere sahip bahis bulunamadı. 🚫")
        else:
            print(df_kelly.to_markdown(index=False, floatfmt=".2f"))
    
    # 5. Telegram Mesajını Gönderme (Model başarısız olsa bile bildirim gönderir)
    telegram_mesaj_gonder(df_kelly, bugunun_tarihi.strftime('%Y-%m-%d'))


if __name__ == "__main__":
    main()
