import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import warnings
import requests
import os

warnings.filterwarnings('ignore')

# TELEGRAM AYARLARI - BUNLARI DOLDURUN
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN") 
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

# Görselleştirme ayarları
plt.style.use('default')
sns.set_palette("husl")

print("🏀 Basketbol Fikstür Analiz ve Tahmin Programı")
print("=" * 50)

# Veriyi yükle
try:
    df = pd.read_csv('BasketbolFikstür - Sayfa1.tsv', sep='\t', encoding='utf-8')
    print("✅ Veri başarıyla yüklendi")
except Exception as e:
    print(f"❌ Dosya yükleme hatası: {e}")
    exit()

# Sütun isimlerini temizle
df.columns = df.columns.str.strip()

# Temizlik işlemleri
def clean_data(df):
    df['Tarih'] = pd.to_datetime(df['Tarih'], dayfirst=True, errors='coerce')
    
    score_columns = ['MS(Ev)', 'MS(Dep)', 'İY(Ev)', 'İY(Dep)']
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Toplam_Skor'] = df['MS(Ev)'] + df['MS(Dep)']
    df['Skor_Farkı'] = abs(df['MS(Ev)'] - df['MS(Dep)'])
    df['Kazanan'] = np.where(df['MS(Ev)'] > df['MS(Dep)'], df['Ev Sahibi'], 
                            np.where(df['MS(Dep)'] > df['MS(Ev)'], df['Deplasman'], 'Berabere'))
    
    return df

df = clean_data(df)
print("✅ Veri temizleme tamamlandı")

def calculate_advanced_team_stats(df):
    """Gelişmiş takım istatistikleri"""
    completed = df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    team_stats = {}
    lig_performans = {}
    
    # Lig istatistiklerini hesapla
    for lig in completed['Lig'].unique():
        lig_maclar = completed[completed['Lig'] == lig]
        lig_ort_skor = lig_maclar['Toplam_Skor'].mean()
        lig_performans[lig] = {
            'ort_skor': lig_ort_skor,
            'savunma_gucu': max(30, 100 - (lig_ort_skor / 3)),
            'hiz_katsayisi': lig_ort_skor / 180
        }
    
    for takim in set(list(completed['Ev Sahibi']) + list(completed['Deplasman'])):
        ev_maclar = completed[completed['Ev Sahibi'] == takim]
        dep_maclar = completed[completed['Deplasman'] == takim]
        
        # Temel istatistikler
        ev_galibiyet = len(ev_maclar[ev_maclar['Kazanan'] == takim])
        dep_galibiyet = len(dep_maclar[dep_maclar['Kazanan'] == takim])
        
        ev_galibiyet_yuzdesi = (ev_galibiyet / len(ev_maclar) * 100) if len(ev_maclar) > 0 else 0
        dep_galibiyet_yuzdesi = (dep_galibiyet / len(dep_maclar) * 100) if len(dep_maclar) > 0 else 0
        
        ev_ort_skor = ev_maclar['MS(Ev)'].mean() if len(ev_maclar) > 0 else 0
        dep_ort_skor = dep_maclar['MS(Dep)'].mean() if len(dep_maclar) > 0 else 0
        ev_yenen_ort_skor = ev_maclar['MS(Dep)'].mean() if len(ev_maclar) > 0 else 0
        dep_yenen_ort_skor = dep_maclar['MS(Ev)'].mean() if len(dep_maclar) > 0 else 0
        
        # Form durumu (Son 10 maç)
        tum_maclar = pd.concat([ev_maclar, dep_maclar]).sort_values('Tarih')
        son_10_mac = tum_maclar.tail(10)
        
        form_puani = 0
        for i, (_, mac) in enumerate(son_10_mac.iterrows()):
            mac_agirlik = 1.0 + (i * 0.1)
            if mac['Kazanan'] == takim:
                form_puani += 3 * mac_agirlik
            elif mac['Kazanan'] == 'Berabere':
                form_puani += 1 * mac_agirlik
        
        # Güç hesaplamaları
        hucum_gucu = (ev_ort_skor * 0.6 + dep_ort_skor * 0.4)
        savunma_gucu = max(10, 100 - ((ev_yenen_ort_skor + dep_yenen_ort_skor) / 2))
        
        # Lig katsayısı
        takim_ligleri = list(ev_maclar['Lig']) + list(dep_maclar['Lig'])
        ana_lig = max(set(takim_ligleri), key=takim_ligleri.count) if takim_ligleri else 'Diğer'
        lig_katsayisi = lig_performans.get(ana_lig, {}).get('savunma_gucu', 50) / 50
        
        team_stats[takim] = {
            'Toplam_Maç': len(ev_maclar) + len(dep_maclar),
            'Galibiyet_Yüzdesi': ((ev_galibiyet + dep_galibiyet) / (len(ev_maclar) + len(dep_maclar)) * 100) if (len(ev_maclar) + len(dep_maclar)) > 0 else 0,
            'Ev_Galibiyet_Yuzdesi': ev_galibiyet_yuzdesi,
            'Dep_Galibiyet_Yuzdesi': dep_galibiyet_yuzdesi,
            'Ev_Ort_Skor': ev_ort_skor,
            'Dep_Ort_Skor': dep_ort_skor,
            'Ev_Yenen_Ort_Skor': ev_yenen_ort_skor,
            'Dep_Yenen_Ort_Skor': dep_yenen_ort_skor,
            'Hucum_Gucu': hucum_gucu,
            'Savunma_Gucu': savunma_gucu,
            'Form_Puani': form_puani,
            'Lig_Katsayisi': lig_katsayisi,
            'Güç_Puanı': (hucum_gucu * 0.3 + savunma_gucu * 0.3 + form_puani * 0.2 + ev_galibiyet_yuzdesi * 0.2)
        }
    
    return team_stats, lig_performans

def calculate_opponent_strength(df, team_stats):
    """Rakiplerin gücünü analiz et"""
    completed = df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    opponent_stats = {}
    
    for takim in set(list(completed['Ev Sahibi']) + list(completed['Deplasman'])):
        takim_maclar = completed[
            (completed['Ev Sahibi'] == takim) | 
            (completed['Deplasman'] == takim)
        ]
        
        rakip_guc_toplam = 0
        rakip_sayisi = 0
        zor_mac_sayisi = 0
        
        for _, mac in takim_maclar.iterrows():
            if mac['Ev Sahibi'] == takim:
                rakip = mac['Deplasman']
            else:
                rakip = mac['Ev Sahibi']
            
            rakip_guc = team_stats.get(rakip, {}).get('Güç_Puanı', 50)
            rakip_guc_toplam += rakip_guc
            rakip_sayisi += 1
            
            if rakip_guc > 65:
                zor_mac_sayisi += 1
        
        ortalama_rakip_gucu = rakip_guc_toplam / rakip_sayisi if rakip_sayisi > 0 else 50
        
        if ortalama_rakip_gucu > 70:
            zorluk_derecesi = "ÇOK ZOR"
            zorluk_puani = 1.3
        elif ortalama_rakip_gucu > 65:
            zorluk_derecesi = "ZOR"
            zorluk_puani = 1.15
        elif ortalama_rakip_gucu > 55:
            zorluk_derecesi = "ORTA"
            zorluk_puani = 1.0
        else:
            zorluk_derecesi = "KOLAY"
            zorluk_puani = 0.85
        
        opponent_stats[takim] = {
            'Ortalama_Rakip_Gücü': round(ortalama_rakip_gucu, 1),
            'Zor_Maç_Sayısı': zor_mac_sayisi,
            'Zorluk_Derecesi': zorluk_derecesi,
            'Zorluk_Puanı': zorluk_puani,
            'Zorluk_Yüzdesi': (zor_mac_sayisi / rakip_sayisi * 100) if rakip_sayisi > 0 else 0
        }
    
    return opponent_stats

def get_next_week_matches(df):
    """Önümüzdeki haftanın maçlarını getir (Pazartesi 09:00'dan sonraki 7 gün)"""
    now = datetime.now()
    
    # Eğer şu an Pazartesi 09:00'dan önceyse, bu Pazartesi'yi bekle
    if now.weekday() == 0 and now.hour < 9:  # Pazartesi ve saat 09:00'dan önce
        start_date = now.replace(hour=9, minute=0, second=0, microsecond=0)
    else:
        # Gelecek Pazartesi 09:00'ı bul
        days_until_monday = (7 - now.weekday()) % 7
        if days_until_monday == 0 and now.hour >= 9:
            days_until_monday = 7  # Bu haftanın maçları bitmiş, gelecek haftaya geç
        start_date = (now + timedelta(days=days_until_monday)).replace(hour=9, minute=0, second=0, microsecond=0)
    
    end_date = start_date + timedelta(days=7)
    
    print(f"📅 Tahmin Aralığı: {start_date.strftime('%d.%m.%Y %H:%M')} - {end_date.strftime('%d.%m.%Y %H:%M')}")
    
    gelecek_maclar = df[
        (df['Tarih'] >= start_date) & 
        (df['Tarih'] < end_date) & 
        (df['MS(Ev)'].isna())
    ]
    
    return gelecek_maclar, start_date, end_date

def enhanced_predict_matches(df, team_stats, lig_performans, opponent_stats):
    """Geliştirilmiş tahmin motoru"""
    gelecek_maclar, start_date, end_date = get_next_week_matches(df)
    
    if len(gececek_maclar) == 0:
        print("❌ Önümüzdeki hafta için maç bulunamadı")
        return []
    
    tahminler = []
    
    for _, mac in gelecek_maclar.iterrows():
        ev_takim = mac['Ev Sahibi']
        dep_takim = mac['Deplasman']
        lig = mac['Lig']
        
        ev_stats = team_stats.get(ev_takim, {})
        dep_stats = team_stats.get(dep_takim, {})
        lig_stats = lig_performans.get(lig, {})
        ev_opponent = opponent_stats.get(ev_takim, {})
        dep_opponent = opponent_stats.get(dep_takim, {})
        
        if not ev_stats or not dep_stats:
            continue
        
        # Lig ayarlamaları
        lig_hiz = lig_stats.get('hiz_katsayisi', 1.0)
        
        # Ev avantajı
        if lig in ['NBA']:
            ev_avantaji = 2.5
        elif lig in ['Eurolig', 'Türkiye']:
            ev_avantaji = 3.5
        else:
            ev_avantaji = 3.0
        
        # Temel skor tahmini
        ev_temel_skor = (ev_stats['Ev_Ort_Skor'] * 0.7 + ev_stats['Dep_Ort_Skor'] * 0.3)
        dep_temel_skor = (dep_stats['Dep_Ort_Skor'] * 0.7 + dep_stats['Ev_Ort_Skor'] * 0.3)
        
        # Form etkisi
        ev_form_etkisi = (ev_stats['Form_Puani'] / 10) * 0.5
        dep_form_etkisi = (dep_stats['Form_Puani'] / 10) * 0.5
        
        # Rakip gücü etkisi
        ev_rakip_etkisi = (ev_opponent.get('Ortalama_Rakip_Gücü', 50) - 50) * 0.1
        dep_rakip_etkisi = (dep_opponent.get('Ortalama_Rakip_Gücü', 50) - 50) * 0.1
        
        # Savunma etkisi
        ev_savunma_etkisi = (100 - dep_stats['Hucum_Gucu']) * 0.1
        dep_savunma_etkisi = (100 - ev_stats['Hucum_Gucu']) * 0.1
        
        # Final tahminler
        ev_tahmin_skor = (
            ev_temel_skor + 
            ev_avantaji + 
            ev_form_etkisi + 
            ev_savunma_etkisi +
            ev_rakip_etkisi
        ) * lig_hiz
        
        dep_tahmin_skor = (
            dep_temel_skor + 
            dep_form_etkisi + 
            dep_savunma_etkisi +
            dep_rakip_etkisi
        ) * lig_hiz
        
        # Kazanan ve olasılık
        skor_farki = ev_tahmin_skor - dep_tahmin_skor
        guc_farki = ev_stats['Güç_Puanı'] - dep_stats['Güç_Puanı']
        
        temel_olasilik = 50 + (skor_farki * 1.5)
        form_farki_etkisi = (ev_stats['Form_Puani'] - dep_stats['Form_Puani']) * 0.3
        rakip_farki_etkisi = (ev_opponent.get('Ortalama_Rakip_Gücü', 50) - dep_opponent.get('Ortalama_Rakip_Gücü', 50)) * 0.2
        
        final_olasilik = min(90, max(10, 
            temel_olasilik + form_farki_etkisi + rakip_farki_etkisi
        ))
        
        if final_olasilik > 50:
            kazanan = ev_takim
            kazanma_olasiligi = final_olasilik
        else:
            kazanan = dep_takim
            kazanma_olasiligi = 100 - final_olasilik
        
        tahmin = {
            'Tarih': mac['Tarih'],
            'Lig': lig,
            'Ev_Sahibi': ev_takim,
            'Deplasman': dep_takim,
            'Tahmin_Ev_Skor': max(70, min(130, round(ev_tahmin_skor))),
            'Tahmin_Dep_Skor': max(70, min(130, round(dep_tahmin_skor))),
            'Tahmin_Kazanan': kazanan,
            'Kazanma_Olasiligi': round(kazanma_olasiligi),
            'Tahmin_Toplam_Skor': round(ev_tahmin_skor + dep_tahmin_skor),
            'Tahmin_Fark': abs(round(ev_tahmin_skor - dep_tahmin_skor)),
            'Güç_Farkı': round(guc_farki, 1),
            'Form_Farkı': round(ev_stats['Form_Puani'] - dep_stats['Form_Puani'], 1)
        }
        
        tahminler.append(tahmin)
    
    return tahminler, start_date, end_date

def send_telegram_message(message):
    """Telegram'a mesaj gönder"""
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        print("⚠️  Telegram ayarları yapılandırılmamış")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    
    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Telegram mesajı gönderildi")
            return True
        else:
            print(f"❌ Telegram hatası: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Telegram bağlantı hatası: {e}")
        return False

def create_telegram_prediction_report(tahminler, start_date, end_date):
    """Telegram için tahmin raporu oluştur"""
    if not tahminler:
        return "📅 Önümüzdeki hafta için maç bulunamadı."
    
    mesaj = f"<b>🏀 HAFTALIK BASKETBOL TAHMİNLERİ</b>\n"
    mesaj += f"<i>⏰ {start_date.strftime('%d.%m %H:%M')} - {end_date.strftime('%d.%m %H:%M')}</i>\n"
    mesaj += "═" * 35 + "\n\n"
    
    # Tahminleri liglere göre grupla
    ligler = {}
    for tahmin in tahminler:
        lig = tahmin['Lig']
        if lig not in ligler:
            ligler[lig] = []
        ligler[lig].append(tahmin)
    
    for lig, maclar in ligler.items():
        mesaj += f"<b>🏆 {lig}</b>\n"
        
        for mac in maclar:
            tarih = mac['Tarih'].strftime('%a %H:%M') if not pd.isna(mac['Tarih']) else "TBA"
            ev = mac['Ev_Sahibi']
            dep = mac['Deplasman']
            olasilik = mac['Kazanma_Olasiligi']
            toplam = mac['Tahmin_Toplam_Skor']
            
            # Kazananı vurgula
            if mac['Tahmin_Kazanan'] == ev:
                skor_gosterim = f"<b>{mac['Tahmin_Ev_Skor']}</b>-{mac['Tahmin_Dep_Skor']}"
                kazanan_gosterim = f"🏠 {ev}"
            else:
                skor_gosterim = f"{mac['Tahmin_Ev_Skor']}-<b>{mac['Tahmin_Dep_Skor']}</b>"
                kazanan_gosterim = f"✈️ {dep}"
            
            mesaj += f"• {tarih}\n"
            mesaj += f"  {ev} vs {dep}\n"
            mesaj += f"  🎯 {skor_gosterim} | 📊 %{olasilik}\n"
            mesaj += f"  ✅ {kazanan_gosterim}\n\n"
    
    mesaj += f"⏰ Tahmin Zamanı: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
    mesaj += "🔮 <i>Analiz: Form + Güç + Rakip Zorluğu</i>"
    
    return mesaj

def analyze_prediction_accuracy(df, team_stats):
    """Geçmiş tahmin doğruluğunu analiz et"""
    completed = df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    
    if len(completed) < 10:
        print("⚠️  Doğruluk analizi için yeterli maç yok")
        return
    
    # Son 30 maçı analiz et
    son_maclar = completed.nlargest(30, 'Tarih')
    dogru_tahmin = 0
    toplam_tahmin = 0
    
    for _, mac in son_maclar.iterrows():
        ev_takim = mac['Ev Sahibi']
        dep_takim = mac['Deplasman']
        gercek_kazanan = mac['Kazanan']
        
        ev_stats = team_stats.get(ev_takim, {})
        dep_stats = team_stats.get(dep_takim, {})
        
        if not ev_stats or not dep_stats:
            continue
        
        # Basit tahmin (güç puanına göre)
        if ev_stats['Güç_Puanı'] > dep_stats['Güç_Puanı']:
            tahmin_kazanan = ev_takim
        else:
            tahmin_kazanan = dep_takim
        
        if tahmin_kazanan == gercek_kazanan:
            dogru_tahmin += 1
        toplam_tahmin += 1
    
    if toplam_tahmin > 0:
        dogruluk_yuzdesi = (dogru_tahmin / toplam_tahmin) * 100
        print(f"📊 Son {toplam_tahmin} maç tahmin doğruluğu: %{dogruluk_yuzdesi:.1f}")
        
        if dogruluk_yuzdesi < 55:
            print("⚠️  Tahmin doğruluğu düşük, model ayarları gözden geçirilmeli")
        elif dogruluk_yuzdesi > 65:
            print("✅ Tahmin doğruluğu iyi seviyede")
    
    return dogruluk_yuzdesi if toplam_tahmin > 0 else 0

# ANA PROGRAM
def main():
    print(f"\n📊 Veri Boyutu: {df.shape}")
    print(f"📅 Tarih Aralığı: {df['Tarih'].min().strftime('%d.%m.%Y')} - {df['Tarih'].max().strftime('%d.%m.%Y')}")
    
    completed_matches = df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    print(f"✅ Tamamlanmış Maç Sayısı: {len(completed_matches)}")
    print(f"📅 Planlanmış Maç Sayısı: {len(df) - len(completed_matches)}")
    
    if len(completed_matches) == 0:
        print("❌ Analiz için tamamlanmış maç bulunamadı")
        return
    
    # İstatistikleri hesapla
    print("\n🔍 Takım istatistikleri hesaplanıyor...")
    team_stats, lig_performans = calculate_advanced_team_stats(df)
    
    print("🎯 Rakip analizi yapılıyor...")
    opponent_stats = calculate_opponent_strength(df, team_stats)
    
    # Tahmin doğruluğunu analiz et
    print("📊 Tahmin doğruluğu kontrol ediliyor...")
    dogruluk = analyze_prediction_accuracy(df, team_stats)
    
    # Tahmin yap
    print("🔮 Önümüzdeki hafta tahmin ediliyor...")
    tahminler, start_date, end_date = enhanced_predict_matches(df, team_stats, lig_performans, opponent_stats)
    
    if tahminler:
        print(f"✅ {len(tahminler)} maç tahmin edildi")
        
        # Telegram raporu oluştur ve gönder
        telegram_mesaj = create_telegram_prediction_report(tahminler, start_date, end_date)
        print("\n" + "="*50)
        print("TELEGRAM MESAJI:")
        print("="*50)
        print(telegram_mesaj)
        
        # Telegram'a gönder
        send_telegram_message(telegram_mesaj)
        
        # Tahminleri kaydet
        tahmin_df = pd.DataFrame(tahminler)
        tahmin_df.to_csv('haftalik_tahminler.csv', index=False, encoding='utf-8')
        print(f"\n💾 Tahminler 'haftalik_tahminler.csv' dosyasına kaydedildi")
        
        # Model performansı
        if dogruluk > 0:
            print(f"🎯 Model Doğruluk: %{dogruluk:.1f}")
            
    else:
        print("❌ Tahmin yapılabilecek maç bulunamadı")

if __name__ == "__main__":
    main()
    print(f"\n{' 🎉 PROGRAM TAMAMLANDI ':=^50}")
