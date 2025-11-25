import pandas as pd
import matplotlib
matplotlib.use('Agg') # Görselleştirme için
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import warnings
import requests
import os
import time
from matplotlib.gridspec import GridSpec

# Ayarlar
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("tab10")

# GitHub Secrets'den al
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# --- VERİ YÜKLEME VE TEMİZLİK (AYNEN KORUNDU) ---
def log_environment_info():
    """Çevre değişkenlerini kontrol et"""
    print("🏀 Basketbol Fikstür Analiz ve Tahmin Programı")
    print("=" * 50)
    print(f"📅 Çalışma Zamanı: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    print(f"🔐 Telegram Bot: {'✅ Ayarlı' if TELEGRAM_BOT_TOKEN else '❌ Ayarlı Değil'}")
    print(f"💬 Telegram Chat: {'✅ Ayarlı' if TELEGRAM_CHAT_ID else '❌ Ayarlı Değil'}")

try:
    df = pd.read_csv('BasketbolFikstür - Sayfa1.tsv', sep='\t', encoding='utf-8')
    df.columns = df.columns.str.strip()
    print("✅ Veri başarıyla yüklendi ve temizlendi")
except Exception as e:
    print(f"❌ Dosya yükleme hatası: {e}")
    exit()

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
# --------------------------------------------------------------------------------

# --- YENİ ZAMAN YÖNETİMİ FONKSİYONU ---

def get_analysis_periods(df):
    """
    Bu haftanın kalanını (Şu andan Pazartesi 09:00'a) ve ardından
    Gelecek haftayı (Pazartesi 09:00'dan sonraki 7 gün) hesaplar.
    """
    now = datetime.now()
    periods = []

    # 1. Bitiş Tarihi: Önümüzdeki Pazartesi 09:00
    days_until_next_monday = (7 - now.weekday()) % 7
    if days_until_next_monday == 0 and now.hour >= 9:
        days_until_next_monday = 7 # Eğer Pazartesi 09:00'dan sonra ise, sonraki Pazartesi'yi al
    
    end_of_current_period = (now + timedelta(days=days_until_next_monday)).replace(hour=9, minute=0, second=0, microsecond=0)

    # AŞAMA 1: Bu Haftanın Kalanı (Şu andan Pazartesi 09:00'a)
    if end_of_current_period > now:
        periods.append({
            'name': "BU HAFTANIN KALANI",
            'start_date': now.replace(second=0, microsecond=0),
            'end_date': end_of_current_period
        })
        print(f"1. Tahmin: {periods[0]['name']} ({periods[0]['start_date'].strftime('%d.%m %H:%M')} - {periods[0]['end_date'].strftime('%d.%m %H:%M')})")

    # AŞAMA 2: Gelecek Hafta (Pazartesi 09:00'dan sonraki 7 gün)
    start_of_next_week = end_of_current_period
    end_of_next_week = start_of_next_week + timedelta(days=7)
    
    periods.append({
        'name': "GELECEK HAFTA",
        'start_date': start_of_next_week,
        'end_date': end_of_next_week
    })
    print(f"2. Tahmin: {periods[-1]['name']} ({periods[-1]['start_date'].strftime('%d.%m %H:%M')} - {periods[-1]['end_date'].strftime('%d.%m %H:%M')})")
    
    return periods
    
# --------------------------------------------------------------------------------

# --- İSTATİSTİK VE GÜÇ HESAPLAMALARI (GELİŞTİRİLDİ) ---

def calculate_advanced_team_stats(df):
    """Gelişmiş takım istatistikleri ve lig ortalamaları"""
    completed = df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    team_stats = {}
    lig_performans = {}
    
    # Tüm liglerin genel ortalamasını hesapla (Yeni takımlar için varsayılan değer)
    genel_ort_skor = completed['Toplam_Skor'].mean() if len(completed) > 0 else 180
    genel_ev_skor = completed['MS(Ev)'].mean() if len(completed) > 0 else 90
    genel_dep_skor = completed['MS(Dep)'].mean() if len(completed) > 0 else 90
    
    # Lig istatistiklerini hesapla
    for lig in completed['Lig'].unique():
        lig_maclar = completed[completed['Lig'] == lig]
        lig_ort_skor = lig_maclar['Toplam_Skor'].mean()
        lig_performans[lig] = {
            'ort_skor': lig_ort_skor,
            'savunma_gucu_ort': max(30, 100 - (lig_ort_skor / 3)), # Ligin ortalama savunma gücü
            'hiz_katsayisi': lig_ort_skor / genel_ort_skor, # Ligin genel ortalamaya göre hızı
            'ev_ort': lig_maclar['MS(Ev)'].mean(),
            'dep_ort': lig_maclar['MS(Dep)'].mean(),
        }
    
    # Takım istatistiklerini hesapla
    for takim in set(list(completed['Ev Sahibi']) + list(completed['Deplasman'])):
        ev_maclar = completed[completed['Ev Sahibi'] == takim]
        dep_maclar = completed[completed['Deplasman'] == takim]
        
        tum_maclar = pd.concat([ev_maclar, dep_maclar]).sort_values('Tarih')
        
        # Ana ligi belirle
        takim_ligleri = list(ev_maclar['Lig']) + list(dep_maclar['Lig'])
        ana_lig = max(set(takim_ligleri), key=takim_ligleri.count) if takim_ligleri else 'Diğer'
        lig_stats = lig_performans.get(ana_lig, {})
        
        # Dinamik varsayılan değerler
        LIG_EV_ORT = lig_stats.get('ev_ort', genel_ev_skor)
        LIG_DEP_ORT = lig_stats.get('dep_ort', genel_dep_skor)
        
        # Temel istatistikler (Eksik maçlar için lig ortalamasını kullan)
        ev_ort_skor = ev_maclar['MS(Ev)'].mean() if len(ev_maclar) > 2 else LIG_EV_ORT
        dep_ort_skor = dep_maclar['MS(Dep)'].mean() if len(dep_maclar) > 2 else LIG_DEP_ORT
        ev_yenen_ort_skor = ev_maclar['MS(Dep)'].mean() if len(ev_maclar) > 2 else LIG_DEP_ORT
        dep_yenen_ort_skor = dep_maclar['MS(Ev)'].mean() if len(dep_maclar) > 2 else LIG_EV_ORT
        
        # Form durumu (Son 10 maç - Ağırlıklı puan)
        son_10_mac = tum_maclar.tail(10)
        form_puani = 0
        for i, (_, mac) in enumerate(son_10_mac.iterrows()):
            mac_agirlik = 1.0 + (i * 0.1) # Daha yeni maça daha çok ağırlık
            if mac['Kazanan'] == takim:
                form_puani += 3 * mac_agirlik
            elif mac['Kazanan'] == 'Berabere':
                form_puani += 1 * mac_agirlik
        
        # Güç hesaplamaları
        # Hücum: Kendi attığı skorların ortalaması (Evde daha çok önem, Lig ortalamasına göre normalize)
        hucum_gucu = (ev_ort_skor / LIG_EV_ORT * 50 * 0.6 + dep_ort_skor / LIG_DEP_ORT * 50 * 0.4)
        
        # Savunma: Yediği skorların ortalaması (Az yiyen yüksek puan alır)
        yenen_ort = (ev_yenen_ort_skor + dep_yenen_ort_skor) / 2
        savunma_gucu = max(10, 100 - (yenen_ort / (genel_ort_skor / 2) * 50))
        
        team_stats[takim] = {
            'Toplam_Maç': len(tum_maclar),
            'Ev_Ort_Skor': ev_ort_skor,
            'Dep_Ort_Skor': dep_ort_skor,
            'Hucum_Gucu': hucum_gucu,
            'Savunma_Gucu': savunma_gucu,
            'Form_Puani': form_puani,
            'Lig_Katsayisi': lig_stats.get('hiz_katsayisi', 1.0),
            'Ana_Lig': ana_lig,
            # Toplam Güç Puanı (Ev/Dep galibiyet yüzdesinin ağırlığı %10'a düşürüldü)
            'Güç_Puanı': (hucum_gucu * 0.35 + savunma_gucu * 0.35 + form_puani * 0.30) 
        }
    
    return team_stats, lig_performans

# --------------------------------------------------------------------------------

# --- RAKİP ANALİZİ (AYNEN KORUNDU) ---

def calculate_opponent_strength(df, team_stats):
    # Fonksiyon aynen korunur, Lig zorluk derecesini hesaplar
    # ... (kodun bu kısmı önceki haliyle aynı)
    completed = df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    opponent_stats = {}
    
    for takim in set(list(completed['Ev Sahibi']) + list(completed['Deplasman'])):
        # ... (iç mantık önceki kodunuzdaki gibi kalır)
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
        
        # Zorluk Derecesi ayarları
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
            'Zorluk_Puanı': zorluk_puani,
        }
    return opponent_stats
# --------------------------------------------------------------------------------

# --- TAHMİN MOTORU (GELİŞTİRİLDİ) ---

def enhanced_predict_matches(df, team_stats, lig_performans, opponent_stats, period):
    """Geliştirilmiş tahmin motoru, dinamik periyotlar için"""
    
    start_date = period['start_date']
    end_date = period['end_date']
    
    print(f"🔮 Tahmin Aralığı ({period['name']}): {start_date.strftime('%d.%m %H:%M')} - {end_date.strftime('%d.%m %H:%M')}")
    
    gelecek_maclar = df[
        (df['Tarih'] >= start_date) & 
        (df['Tarih'] < end_date) & 
        (df['MS(Ev)'].isna())
    ].copy()
    
    if len(gelecek_maclar) == 0:
        return [], start_date, end_date
    
    tahminler = []
    
    for _, mac in gelecek_maclar.iterrows():
        ev_takim = mac['Ev Sahibi']
        dep_takim = mac['Deplasman']
        lig = mac['Lig']
        
        ev_stats = team_stats.get(ev_takim, {})
        dep_stats = team_stats.get(dep_takim, {})
        lig_stats = lig_performans.get(lig, {})
        ev_opponent_zorluk = opponent_stats.get(ev_takim, {}).get('Zorluk_Puanı', 1.0)
        dep_opponent_zorluk = opponent_stats.get(dep_takim, {}).get('Zorluk_Puanı', 1.0)
        
        if not ev_stats or not dep_stats:
            continue
        
        # Lig ayarlamaları
        lig_hiz = lig_stats.get('hiz_katsayisi', 1.0)
        
        # Ev avantajı (Daha agresif bir çarpan)
        ev_avantaji = 4.0 if lig in ['Eurolig', 'Türkiye'] else (3.0 if lig != 'NBA' else 2.5)
        
        # --- SKOR TAHMİNİ (REVİZE EDİLDİ) ---
        
        # 1. Temel Ortalama Skor: Ev sahibinin ev ortalaması vs Deplasmanın deplasman ortalaması
        ev_temel = ev_stats['Ev_Ort_Skor'] 
        dep_temel = dep_stats['Dep_Ort_Skor']
        
        # 2. Savunma ve Hücum Gücü Dengesi
        # Evin atacağı skor = (Ev Hucum Gücü + Deplasman Savunma Zayıflığı) * Ev Ortalaması
        ev_hucum_katkisi = ev_stats['Hucum_Gucu'] * 0.5 
        dep_savunma_kırıcı = (100 - dep_stats['Savunma_Gucu']) * 0.5
        
        # Deplasmanın atacağı skor = (Dep Hucum Gücü + Ev Savunma Zayıflığı) * Dep Ortalaması
        dep_hucum_katkisi = dep_stats['Hucum_Gucu'] * 0.5
        ev_savunma_kırıcı = (100 - ev_stats['Savunma_Gucu']) * 0.5
        
        # 3. Final Tahmini Skoru Hesaplama
        
        # Ev Skoru: Ev Temel + Ev Avantajı + Ev Hucum Gücü (Rakip Savunmasına Göre)
        ev_tahmin_skor = (
            ev_temel + 
            ev_avantaji + 
            (ev_hucum_katkisi + dep_savunma_kırıcı) * 0.1 # Güç farkını puana dönüştür
        ) * lig_hiz
        
        # Dep Skoru: Dep Temel + Dep Hucum Gücü (Rakip Savunmasına Göre)
        dep_tahmin_skor = (
            dep_temel + 
            (dep_hucum_katkisi + ev_savunma_kırıcı) * 0.1
        ) * lig_hiz
        
        # --- OLASILIK HESAPLAMASI (REVİZE EDİLDİ) ---
        
        # Güç Puanı farkı (ana faktör)
        guc_farki = ev_stats['Güç_Puanı'] - dep_stats['Güç_Puanı']
        
        # Skor farkı (ikincil faktör)
        skor_farki = ev_tahmin_skor - dep_tahmin_skor
        
        # Zorluk/Form faktörleri
        form_farki = ev_stats['Form_Puani'] - dep_stats['Form_Puani']
        
        # Temel olasılık (50% + güç farkı + skor farkı)
        final_olasilik = 50 + (guc_farki * 0.5) + (skor_farki * 0.3) + (form_farki * 0.2)
        
        # Ev sahibi zorlu rakiple oynuyorsa olasılığı düşür, deplasman zorlu rakiple oynuyorsa yükselt
        final_olasilik *= (dep_opponent_zorluk / ev_opponent_zorluk) # Zorluk çarpanı
        
        # Sınırlama
        final_olasilik = min(95, max(5, final_olasilik)) # Olasılık aralığını 5-95'e çıkardık
        
        if final_olasilik >= 50:
            kazanan = ev_takim
            kazanma_olasiligi = final_olasilik
        else:
            kazanan = dep_takim
            kazanma_olasiligi = 100 - final_olasilik
        
        # Skorları yuvarla ve kısıtla (Yine de lig hızına göre kısıtlama dinamik yapılmalı, şimdilik 140)
        tahmin_ev_skor = max(60, min(140, round(ev_tahmin_skor)))
        tahmin_dep_skor = max(60, min(140, round(dep_tahmin_skor)))
        
        tahmin = {
            'Tarih': mac['Tarih'],
            'Lig': lig,
            'Ev_Sahibi': ev_takim,
            'Deplasman': dep_takim,
            'Tahmin_Ev_Skor': tahmin_ev_skor,
            'Tahmin_Dep_Skor': tahmin_dep_skor,
            'Tahmin_Kazanan': kazanan,
            'Kazanma_Olasiligi': round(kazanma_olasiligi),
            'Tahmin_Toplam_Skor': tahmin_ev_skor + tahmin_dep_skor,
            'Güç_Farkı': round(guc_farki, 1),
            'Form_Farkı': round(form_farki, 1),
            'Zorluk_Oranı': round(dep_opponent_zorluk / ev_opponent_zorluk, 2)
        }
        
        tahminler.append(tahmin)
    
    return tahminler, start_date, end_date
    
# --------------------------------------------------------------------------------

# --- TELEGRAM İŞLEMLERİ (GÖRSEL RAPOR İÇİN GÜNCELLENDİ) ---

def send_telegram_photo(file_path, caption=""):
    """Telegram'a fotoğraf gönder"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram ayarları bulunamadı - Fotoğraf gönderilemedi")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    try:
        with open(file_path, 'rb') as photo:
            response = requests.post(url, 
                                     data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'}, 
                                     files={'photo': photo},
                                     timeout=30)
        
        if response.status_code == 200:
            print(f"✅ Telegram görsel raporu ({file_path}) başarıyla gönderildi")
            return True
        else:
            print(f"❌ Telegram fotoğraf hatası: {response.status_code}")
            print(f"    Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Telegram bağlantı hatası: {e}")
        return False

# --- YENİ GÖRSEL RAPOR FONKSİYONU ---

def create_visual_report(all_tahminler, all_periods, dogruluk, detay_analiz, file_name='tahmin_raporu.png'):
    """Detaylı tahminleri ve lig analizlerini içeren görsel rapor oluşturur (PNG)"""
    
    # Rapor boyutu
    fig = plt.figure(figsize=(12, 16), facecolor='white')
    gs = GridSpec(6, 2, figure=fig, hspace=0.4, wspace=0.2)
    
    # 1. Başlık ve Özet
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    tahmin_sayisi = sum(len(t) for t in all_tahminler)
    
    # Başlık
    ax_title.text(0.02, 0.9, "🏀 BASKETBOL HAFTALIK TAHMİN RAPORU", 
                  fontsize=24, fontweight='bold', color='#1f77b4')
    ax_title.text(0.02, 0.65, f"⏰ Tahmin Periyodu: {all_periods[0]['start_date'].strftime('%d.%m %H:%M')} - {all_periods[-1]['end_date'].strftime('%d.%m %H:%M')}", 
                  fontsize=14, color='gray')
    
    # Model Doğruluğu
    ax_title.text(0.02, 0.4, f"📊 Model Doğruluğu (Son 30 Maç):", fontsize=16, fontweight='bold')
    ax_title.text(0.02, 0.1, f"  %{dogruluk:.1f} ({detay_analiz.get('dogru_tahmin', 0)}/{detay_analiz.get('toplam_tahmin', 0)})", 
                  fontsize=22, color='green' if dogruluk > 55 else 'red', fontweight='bold')
    ax_title.text(0.5, 0.4, f"📈 Tahmin Edilen Toplam Maç:", fontsize=16, fontweight='bold')
    ax_title.text(0.5, 0.1, f"  {tahmin_sayisi}", 
                  fontsize=22, color='#ff7f0e', fontweight='bold')


    # 2. Lig Bazlı Doğruluk Analizi
    ax_acc = fig.add_subplot(gs[1, :])
    ax_acc.set_title("🏆 Lig Bazlı Doğruluk Analizi (Gelişmiş Model)", fontsize=16, fontweight='bold')
    ax_acc.axis('off')
    
    lig_data = detay_analiz.get('lig_bazli', {})
    
    table_data = []
    
    # Başlık satırı
    table_data.append(["LİG", "DOĞRULUK", "MAÇ SAYISI", "HÜCUM/SAVUNMA"])
    
    for lig, stats in lig_data.items():
        if stats['toplam'] > 0:
            lig_dogruluk = (stats['dogru'] / stats['toplam']) * 100
            
            # Ortalama Hücum ve Savunma Puanlarını bul (Basit bir gösterge)
            lig_hg = np.mean([team_stats[t].get('Hucum_Gucu', 50) for t in df[(df['Lig'] == lig) & df['MS(Ev)'].notna()]['Ev Sahibi'].unique() if t in team_stats])
            lig_sg = np.mean([team_stats[t].get('Savunma_Gucu', 50) for t in df[(df['Lig'] == lig) & df['MS(Ev)'].notna()]['Ev Sahibi'].unique() if t in team_stats])
            
            table_data.append([
                lig, 
                f"%{lig_dogruluk:.1f}", 
                str(stats['toplam']),
                f"H:{lig_hg:.0f} / S:{lig_sg:.0f}"
            ])

    if len(table_data) > 1:
        table = ax_acc.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.2, 0.2, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
        # Başlık satırını vurgula
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#d9e5f5')
                cell.set_fontsize(12)


    # 3. Tahmin Detayları (Aşamalar ve Ligler)
    row_start = 2
    for period_idx, tahmin_list in enumerate(all_tahminler):
        if not tahmin_list:
            continue
            
        period = all_periods[period_idx]
        
        # Tahminleri DataFrame'e dönüştür
        tahmin_df = pd.DataFrame(tahmin_list)
        
        # En çok maç olan ilk 3 ligi bul
        top_ligler = tahmin_df['Lig'].value_counts().nlargest(3).index.tolist()
        
        ax = fig.add_subplot(gs[row_start + period_idx * 2, :])
        ax.axis('off')
        
        ax.text(0.02, 1.0, f"📅 {period['name']} TAHMİNLERİ", 
                fontsize=18, fontweight='bold', color='darkred')
        
        # Tahmin Tablosu Hazırlığı
        final_table_data = []
        final_table_data.append(["Tarih", "Maç", "Tahmin Skor", "Kazanan", "Olasılık", "Güç/Form Farkı"])
        
        # Gösterilecek satır sayısı
        max_rows = 15
        
        for i, mac in tahmin_df.sort_values(by='Tarih').head(max_rows).iterrows():
            tarih = mac['Tarih'].strftime('%a %H:%M')
            mac_isim = f"{mac['Ev_Sahibi'][:12]} vs {mac['Deplasman'][:12]} ({mac['Lig'][:4]})"
            skor = f"{mac['Tahmin_Ev_Skor']}-{mac['Tahmin_Dep_Skor']}"
            kazanan = mac['Tahmin_Kazanan'][:15]
            olasilik = f"%{mac['Kazanma_Olasiligi']}"
            fark = f"G:{mac['Güç_Farkı']} / F:{mac['Form_Farkı']}"
            
            final_table_data.append([tarih, mac_isim, skor, kazanan, olasilik, fark])

        table_ax = fig.add_subplot(gs[row_start + period_idx * 2 + 1, :])
        table_ax.axis('off')
        
        if len(final_table_data) > 1:
            table = table_ax.table(cellText=final_table_data, loc='center', cellLoc='center', colWidths=[0.1, 0.3, 0.15, 0.2, 0.1, 0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.2)
            
            # Başlık satırını vurgula
            for (i, j), cell in table.get_celld().items():
                if i == 0:
                    cell.set_facecolor('#d9e5f5')
                    cell.set_fontsize(9)
        else:
            table_ax.text(0.5, 0.5, "Bu periyotta tahmin edilecek maç bulunamadı.", ha='center', va='center', fontsize=12)

    
    # Son bilgi
    ax_footer = fig.add_subplot(gs[-1, :])
    ax_footer.axis('off')
    ax_footer.text(0.02, 0.5, f"Rapor Oluşturma Zamanı: {datetime.now().strftime('%d.%m.%Y %H:%M')}", fontsize=10, color='gray')
    ax_footer.text(0.02, 0.1, f"Analiz Kriterleri: Güç Puanı, Ağırlıklı Form, Savunma/Hücum Dengesi, Rakip Zorluğu.", fontsize=10, color='gray')

    
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Başlık için biraz boşluk
    plt.savefig(file_name, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ Görsel rapor ('{file_name}') başarıyla oluşturuldu.")
    return file_name

# --------------------------------------------------------------------------------

# --- ANA PROGRAM (GÜNCELLENDİ) ---

def main():
    global df, team_stats
    
    log_environment_info()
    
    print("\n" + "="*50)
    print(f"📊 Veri Boyutu: {df.shape}")
    completed_matches = df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    print(f"✅ Tamamlanmış Maç Sayısı: {len(completed_matches)}")
    print("="*50)

    if len(completed_matches) == 0:
        print("❌ Analiz için tamamlanmış maç bulunamadı")
        return
    
    # 1. İstatistikleri hesapla
    print("🔍 İstatistikler ve Güç Puanları hesaplanıyor...")
    team_stats, lig_performans = calculate_advanced_team_stats(df)
    opponent_stats = calculate_opponent_strength(df, team_stats)
    
    # 2. Doğruluk Analizi
    print("📊 Detaylı tahmin doğruluğu analizi...")
    dogruluk, detay_analiz = analyze_prediction_accuracy_detailed(df, team_stats)
    detay_analiz['dogru_tahmin'] = round((detay_analiz.get('gelismis_tahmin', 0) / 100) * detay_analiz.get('toplam_tahmin', 0))
    
    # 3. Tahmin Periyotlarını Al
    periods = get_analysis_periods(df)
    
    # 4. Tahmin Yap (Her periyot için ayrı ayrı)
    print("\n🔮 Tahminler başlıyor...")
    all_tahminler = []
    
    for period in periods:
        tahminler, start_date, end_date = enhanced_predict_matches(df, team_stats, lig_performans, opponent_stats, period)
        if tahminler:
            all_tahminler.append(tahminler)
            print(f"  ✅ {period['name']} için {len(tahminler)} maç tahmin edildi")
    
    flat_tahminler = [t for sublist in all_tahminler for t in sublist] # Tüm tahminleri tek listeye topla

    if flat_tahminler:
        # 5. Görsel Rapor Oluştur ve Gönder
        rapor_dosya_adi = create_visual_report(all_tahminler, periods, dogruluk, detay_analiz)
        
        print("\n" + "="*50)
        print("TELEGRAM GÖNDERİMİ:")
        print("="*50)
        
        caption = f"<b>🏀 HAFTALIK BASKETBOL RAPORU</b>\n\n"\
                  f"⏰ {periods[0]['start_date'].strftime('%d.%m %H:%M')} - {periods[-1]['end_date'].strftime('%d.%m %H:%M')}\n\n"\
                  f"📊 Model Doğruluğu (Son 30): <b>%{dogruluk:.1f}</b>\n"\
                  f"📈 Toplam Tahmin: <b>{len(flat_tahminler)}</b> maç\n"

        send_telegram_photo(rapor_dosya_adi, caption=caption)
        
        # Tahminleri kaydet
        tahmin_df = pd.DataFrame(flat_tahminler)
        tahmin_df.to_csv('haftalik_tahminler.csv', index=False, encoding='utf-8')
        print(f"💾 Tahminler 'haftalik_tahminler.csv' dosyasına kaydedildi")
        
    else:
        print("❌ Tahmin yapılabilecek maç bulunamadı")
        info_message = f"🏀 Haftalık Basketbol Tahminleri\n\n📅 Önümüzdeki 14 gün için maç bulunamadı.\n⏰ Son kontrol: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
        # Boş da olsa Telegram'a bilgi ver (Mesaj olarak)
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
                          data={'chat_id': TELEGRAM_CHAT_ID, 'text': info_message})

if __name__ == "__main__":
    main()
    print(f"\n{' 🎉 PROGRAM TAMAMLANDI ':=^50}")
