import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
plt.style.use('default')
sns.set_palette("husl")

print("🏀 Basketbol Fikstür Analiz Programı")
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

# İlk bakış
print(f"\n📊 Veri Boyutu: {df.shape}")
print(f"📅 Tarih Aralığı: {df['Tarih'].min()} - {df['Tarih'].max()}")
print(f"🏆 Ligler: {', '.join(df['Lig'].unique())}")

# Temizlik işlemleri
def clean_data(df):
    # Tarihi düzenle
    df['Tarih'] = pd.to_datetime(df['Tarih'], dayfirst=True, errors='coerce')
    
    # Skorları sayısala çevir (boş değerleri NaN yap)
    score_columns = ['MS(Ev)', 'MS(Dep)', 'İY(Ev)', 'İY(Dep)']
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Toplam skorları hesapla
    df['Toplam_Skor'] = df['MS(Ev)'] + df['MS(Dep)']
    df['Skor_Farkı'] = abs(df['MS(Ev)'] - df['MS(Dep)'])
    
    # Kazananı belirle
    df['Kazanan'] = np.where(df['MS(Ev)'] > df['MS(Dep)'], df['Ev Sahibi'], 
                            np.where(df['MS(Dep)'] > df['MS(Ev)'], df['Deplasman'], 'Berabere'))
    
    # Ay ve Hafta bilgisi
    df['Ay'] = df['Tarih'].dt.month
    df['Hafta'] = df['Tarih'].dt.isocalendar().week
    
    return df

df = clean_data(df)
print("✅ Veri temizleme tamamlandı")

# 1. GENEL İSTATİSTİKLER
def genel_istatistikler(df):
    print("\n📈 GENEL İSTATİSTİKLER")
    print("-" * 30)
    
    # Tamamlanmış maçlar
    completed_matches = df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    print(f"Tamamlanmış Maç Sayısı: {len(completed_matches)}")
    print(f"Planlanmış Maç Sayısı: {len(df) - len(completed_matches)}")
    
    # Skor istatistikleri
    print(f"\nOrtalama Toplam Skor: {completed_matches['Toplam_Skor'].mean():.1f}")
    print(f"Ortalama Skor Farkı: {completed_matches['Skor_Farkı'].mean():.1f}")
    print(f"Maksimum Skor: {completed_matches['Toplam_Skor'].max()}")
    print(f"Minimum Skor: {completed_matches['Toplam_Skor'].min()}")

genel_istatistikler(df)

# 2. LİG BAZLI ANALİZ
def lig_analizi(df):
    print("\n🏆 LİG BAZLI ANALİZ")
    print("-" * 30)
    
    completed = df.dropna(subset=['Toplam_Skor'])
    
    lig_stats = completed.groupby('Lig').agg({
        'Toplam_Skor': ['count', 'mean', 'std', 'max'],
        'Skor_Farkı': 'mean'
    }).round(1)
    
    lig_stats.columns = ['Maç_Sayısı', 'Ort_Skor', 'Skor_Std', 'Maks_Skor', 'Ort_Fark']
    print(lig_stats.sort_values('Maç_Sayısı', ascending=False))

lig_analizi(df)

# 3. TAKIM PERFORMANSLARI
def takim_performansi(df, lig=None):
    if lig:
        temp_df = df[df['Lig'] == lig].copy()
        print(f"\n🏀 {lig} LİGİ TAKIM PERFORMANSLARI")
    else:
        temp_df = df.copy()
        print(f"\n🏀 TÜM LİGLER TAKIM PERFORMANSLARI")
    
    print("-" * 40)
    
    completed = temp_df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    
    # Ev sahibi istatistikleri
    ev_stats = completed.groupby('Ev Sahibi').agg({
        'MS(Ev)': ['mean', 'count'],
        'Kazanan': lambda x: (x == completed.loc[x.index, 'Ev Sahibi']).sum()
    })
    
    # Deplasman istatistikleri
    dep_stats = completed.groupby('Deplasman').agg({
        'MS(Dep)': ['mean', 'count'],
        'Kazanan': lambda x: (x == completed.loc[x.index, 'Deplasman']).sum()
    })
    
    # Birleştirme
    ev_stats.columns = ['Ev_Ort_Skor', 'Ev_Maç', 'Ev_Galibiyet']
    dep_stats.columns = ['Dep_Ort_Skor', 'Dep_Maç', 'Dep_Galibiyet']
    
    team_stats = pd.concat([ev_stats, dep_stats], axis=1)
    team_stats['Toplam_Maç'] = team_stats['Ev_Maç'] + team_stats['Dep_Maç']
    team_stats['Toplam_Galibiyet'] = team_stats['Ev_Galibiyet'] + team_stats['Dep_Galibiyet']
    team_stats['Galibiyet_Yüzdesi'] = (team_stats['Toplam_Galibiyet'] / team_stats['Toplam_Maç'] * 100).round(1)
    team_stats['Genel_Ort_Skor'] = (
        (team_stats['Ev_Ort_Skor'] * team_stats['Ev_Maç'] + 
         team_stats['Dep_Ort_Skor'] * team_stats['Dep_Maç']) / team_stats['Toplam_Maç']
    ).round(1)
    
    # Sadece yeterli maçı olan takımları göster
    team_stats = team_stats[team_stats['Toplam_Maç'] >= 3]
    
    return team_stats.sort_values('Galibiyet_Yüzdesi', ascending=False)

# Tüm ligler için takım performansı
all_teams = takim_performansi(df)
print(all_teams.head(10))

# 4. GÖRSELLEŞTİRMELER
def create_visualizations(df):
    print("\n📊 GÖRSELLEŞTİRMELER OLUŞTURULUYOR...")
    
    completed = df.dropna(subset=['Toplam_Skor'])
    
    # 1. Liglere Göre Maç Dağılımı
    plt.figure(figsize=(12, 6))
    match_counts = df['Lig'].value_counts()
    plt.subplot(1, 2, 1)
    match_counts.plot(kind='bar', color='lightblue', edgecolor='black')
    plt.title('Liglere Göre Maç Sayıları')
    plt.xlabel('Lig')
    plt.ylabel('Maç Sayısı')
    plt.xticks(rotation=45)
    
    # 2. Skor Dağılımı
    plt.subplot(1, 2, 2)
    plt.hist(completed['Toplam_Skor'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.title('Toplam Skor Dağılımı')
    plt.xlabel('Toplam Skor')
    plt.ylabel('Frekans')
    plt.axvline(completed['Toplam_Skor'].mean(), color='red', linestyle='--', label=f'Ortalama: {completed["Toplam_Skor"].mean():.1f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 3. Lig Bazlı Skor Karşılaştırması
    plt.figure(figsize=(12, 6))
    lig_means = completed.groupby('Lig')['Toplam_Skor'].mean().sort_values(ascending=False)
    plt.subplot(1, 2, 1)
    lig_means.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title('Liglere Göre Ortalama Skor')
    plt.xlabel('Lig')
    plt.ylabel('Ortalama Skor')
    plt.xticks(rotation=45)
    
    # 4. Zaman Serisi Analizi
    plt.subplot(1, 2, 2)
    monthly_avg = completed.groupby(completed['Tarih'].dt.to_period('M'))['Toplam_Skor'].mean()
    monthly_avg.plot(kind='line', marker='o', color='orange')
    plt.title('Aylara Göre Ortalama Skor Trendi')
    plt.xlabel('Ay')
    plt.ylabel('Ortalama Skor')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 5. Galibiyet Dağılımı (En Başarılı 15 Takım)
    plt.figure(figsize=(12, 8))
    successful_teams = all_teams.head(15)
    plt.barh(successful_teams.index, successful_teams['Galibiyet_Yüzdesi'], 
             color='gold', edgecolor='black')
    plt.xlabel('Galibiyet Yüzdesi (%)')
    plt.title('En Başarılı 15 Takım (Galibiyet Yüzdesi)')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

create_visualizations(df)

# 5. DETAYLI LİG RAPORU
def detayli_lig_raporu(lig_adi):
    print(f"\n🔍 DETAYLI LİG RAPORU: {lig_adi}")
    print("=" * 50)
    
    lig_df = df[df['Lig'] == lig_adi].copy()
    completed = lig_df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    
    if len(completed) == 0:
        print("Bu ligde henüz tamamlanmış maç bulunmuyor.")
        return
    
    # Takım performansı
    teams = takim_performansi(df, lig_adi)
    print(teams.head(10))
    
    # Yüksek skorlu maçlar
    print(f"\n🎯 {lig_adi} - EN YÜKSEK SKORLU MAÇLAR")
    high_scoring = completed.nlargest(5, 'Toplam_Skor')[['Tarih', 'Ev Sahibi', 'MS(Ev)', 'MS(Dep)', 'Deplasman', 'Toplam_Skor']]
    print(high_scoring.to_string(index=False))
    
    # En çekişmeli maçlar
    print(f"\n⚔️ {lig_adi} - EN ÇEKİŞMELİ MAÇLAR (En düşük fark)")
    close_matches = completed.nsmallest(5, 'Skor_Farkı')[['Tarih', 'Ev Sahibi', 'MS(Ev)', 'MS(Dep)', 'Deplasman', 'Skor_Farkı']]
    print(close_matches.to_string(index=False))

# 6. LİG BAZLI DETAYLI ANALİZ
def lig_ozel_analiz(df):
    """
    Her lig için özel istatistikler ve güç dengeleri
    """
    print("\n🏆 LİG BAZLI DETAYLI ANALİZ")
    print("=" * 50)
    
    completed = df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    
    for lig in df['Lig'].unique():
        lig_df = completed[completed['Lig'] == lig]
        
        if len(lig_df) == 0:
            continue
            
        print(f"\n📊 {lig} LİGİ ANALİZİ")
        print("-" * 30)
        
        # Temel istatistikler
        print(f"• Maç Sayısı: {len(lig_df)}")
        print(f"• Ortalama Toplam Skor: {lig_df['Toplam_Skor'].mean():.1f}")
        print(f"• Ortalama Skor Farkı: {lig_df['Skor_Farkı'].mean():.1f}")
        print(f"• Maksimum Skor: {lig_df['Toplam_Skor'].max()}")
        print(f"• Minimum Skor: {lig_df['Toplam_Skor'].min()}")
        
        # Güç dengesi analizi
        print(f"\n⚖️ GÜÇ DENGESİ:")
        
        # Ev sahibi avantajı
        ev_galibiyet = len(lig_df[lig_df['Kazanan'] == lig_df['Ev Sahibi']])
        dep_galibiyet = len(lig_df[lig_df['Kazanan'] == lig_df['Deplasman']])
        ev_avantaj = (ev_galibiyet / len(lig_df)) * 100
        
        print(f"• Ev Sahibi Galibiyet: {ev_galibiyet} (%{ev_avantaj:.1f})")
        print(f"• Deplasman Galibiyet: {dep_galibiyet} (%{(100-ev_avantaj):.1f})")
        
        # Skor tipi analizi
        yuksek_skor = len(lig_df[lig_df['Toplam_Skor'] > lig_df['Toplam_Skor'].median()])
        dusuk_skor = len(lig_df[lig_df['Toplam_Skor'] <= lig_df['Toplam_Skor'].median()])
        
        print(f"• Yüksek Skorlu Maç: {yuksek_skor} (> {lig_df['Toplam_Skor'].median():.1f})")
        print(f"• Düşük Skorlu Maç: {dusuk_skor} (≤ {lig_df['Toplam_Skor'].median():.1f})")
        
        # Çekişme oranı
        cekismeli = len(lig_df[lig_df['Skor_Farkı'] <= 10])  # 10 sayı altı çekişmeli
        cekisme_oran = (cekismeli / len(lig_df)) * 100
        print(f"• Çekişmeli Maç Oranı: %{cekisme_oran:.1f} (≤10 fark)")

def lig_bazli_takim_performansi(df, lig_adi):
    """
    Belirli bir ligdeki takım performanslarını gösterir
    """
    lig_df = df[df['Lig'] == lig_adi].dropna(subset=['MS(Ev)', 'MS(Dep)'])
    
    if len(lig_df) == 0:
        print(f"{lig_adi} liginde henüz maç bulunmuyor.")
        return
    
    print(f"\n🏀 {lig_adi} - TAKIM PERFORMANSLARI")
    print("=" * 50)
    
    takim_stats = {}
    
    for takim in set(list(lig_df['Ev Sahibi']) + list(lig_df['Deplasman'])):
        takim_maclar = lig_df[(lig_df['Ev Sahibi'] == takim) | (lig_df['Deplasman'] == takim)]
        
        galibiyet = 0
        ev_skor = []
        dep_skor = []
        
        for _, mac in takim_maclar.iterrows():
            if mac['Ev Sahibi'] == takim:
                ev_skor.append(mac['MS(Ev)'])
                if mac['Kazanan'] == takim:
                    galibiyet += 1
            else:
                dep_skor.append(mac['MS(Dep)'])
                if mac['Kazanan'] == takim:
                    galibiyet += 1
        
        takim_stats[takim] = {
            'Maç': len(takim_maclar),
            'Galibiyet': galibiyet,
            'Galibiyet_Yüzdesi': (galibiyet / len(takim_maclar) * 100) if len(takim_maclar) > 0 else 0,
            'Ev_Ort_Skor': np.mean(ev_skor) if ev_skor else 0,
            'Dep_Ort_Skor': np.mean(dep_skor) if dep_skor else 0,
            'Genel_Ort_Skor': np.mean(ev_skor + dep_skor) if (ev_skor or dep_skor) else 0
        }
    
    # DataFrame'e çevir ve sırala
    stats_df = pd.DataFrame.from_dict(takim_stats, orient='index')
    stats_df = stats_df[stats_df['Maç'] >= 3]  # En az 3 maçı olan takımlar
    
    if len(stats_df) > 0:
        stats_df = stats_df.round(1)
        print(stats_df.sort_values('Galibiyet_Yüzdesi', ascending=False))
    else:
        print("Yeterli maç sayısına ulaşan takım bulunmuyor.")

def lig_karsilastirma_grafik(df):
    """
    Ligleri görsel olarak karşılaştırır
    """
    completed = df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    
    # Lig bazlı istatistikler
    lig_stats = completed.groupby('Lig').agg({
        'Toplam_Skor': ['mean', 'std', 'count'],
        'Skor_Farkı': 'mean',
        'MS(Ev)': 'mean',
        'MS(Dep)': 'mean'
    }).round(1)
    
    lig_stats.columns = ['Ort_Skor', 'Skor_Std', 'Maç_Sayısı', 'Ort_Fark', 'Ort_Ev_Skor', 'Ort_Dep_Skor']
    
    # Görselleştirme
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Ortalama Skorlar
    axes[0,0].bar(lig_stats.index, lig_stats['Ort_Skor'], color='skyblue', alpha=0.7)
    axes[0,0].set_title('Liglere Göre Ortalama Toplam Skor')
    axes[0,0].set_ylabel('Ortalama Skor')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Skor Farkları
    axes[0,1].bar(lig_stats.index, lig_stats['Ort_Fark'], color='lightcoral', alpha=0.7)
    axes[0,1].set_title('Liglere Göre Ortalama Skor Farkı')
    axes[0,1].set_ylabel('Ortalama Fark')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Ev-Deplasman Karşılaştırması
    width = 0.35
    x = np.arange(len(lig_stats.index))
    axes[1,0].bar(x - width/2, lig_stats['Ort_Ev_Skor'], width, label='Ev Skoru', alpha=0.7)
    axes[1,0].bar(x + width/2, lig_stats['Ort_Dep_Skor'], width, label='Deplasman Skoru', alpha=0.7)
    axes[1,0].set_title('Ev ve Deplasman Skor Karşılaştırması')
    axes[1,0].set_ylabel('Ortalama Skor')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(lig_stats.index, rotation=45)
    axes[1,0].legend()
    
    # 4. Maç Sayıları
    axes[1,1].pie(lig_stats['Maç_Sayısı'], labels=lig_stats.index, autopct='%1.1f%%', startangle=90)
    axes[1,1].set_title('Liglere Göre Maç Dağılımı')
    
    plt.tight_layout()
    plt.show()
    
    return lig_stats

# 7. TELEGRAM MESAJ FORMATI
def telegram_raporu(df, lig_adi=None, gunluk=False):
    """
    Telegram için özet rapor oluşturur
    """
    if lig_adi:
        temp_df = df[df['Lig'] == lig_adi].copy()
        baslik = f"🏀 {lig_adi} GÜNLÜK ÖZET" if gunluk else f"🏀 {lig_adi} LİG RAPORU"
    else:
        temp_df = df.copy()
        baslik = "🏀 BASKETBOL GÜNLÜK ÖZET" if gunluk else "🏀 GENEL BASKETBOL RAPORU"
    
    completed = temp_df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    
    if gunluk:
        bugun = pd.Timestamp.now().normalize()
        completed = completed[completed['Tarih'].dt.date == bugun.date()]
    
    if len(completed) == 0:
        return f"📅 {baslik}\n\n⚪ Bugün maç bulunmamaktadır"
    
    # Telegram mesajı oluştur
    mesaj = f"<b>{baslik}</b>\n"
    mesaj += "═" * 30 + "\n\n"
    
    # Genel istatistikler
    mesaj += f"📊 <b>Genel Bilgiler</b>\n"
    mesaj += f"• Maç Sayısı: {len(completed)}\n"
    mesaj += f"• Ort. Skor: {completed['Toplam_Skor'].mean():.1f}\n"
    mesaj += f"• Ort. Fark: {completed['Skor_Farkı'].mean():.1f}\n\n"
    
    # Son maçlar (max 5)
    mesaj += f"🔥 <b>Son Maçlar</b>\n"
    son_maclar = completed.nlargest(5, 'Tarih')
    for _, mac in son_maclar.iterrows():
        ev_skor = int(mac['MS(Ev)']) if not pd.isna(mac['MS(Ev)']) else "?"
        dep_skor = int(mac['MS(Dep)']) if not pd.isna(mac['MS(Dep)']) else "?"
        tarih = mac['Tarih'].strftime('%H:%M')
        
        kazanan = "🏠" if mac['Kazanan'] == mac['Ev Sahibi'] else "✈️" if mac['Kazanan'] == mac['Deplasman'] else "⚪"
        
        mesaj += f"{kazanan} {mac['Ev Sahibi']} <b>{ev_skor}-{dep_skor}</b> {mac['Deplasman']} ({tarih})\n"
    
    mesaj += "\n"
    
    # Yüksek skorlu maçlar
    mesaj += f"🎯 <b>Yüksek Skorlu Maçlar</b>\n"
    yuksek_skor = completed.nlargest(3, 'Toplam_Skor')
    for _, mac in yuksek_skor.iterrows():
        mesaj += f"• {mac['Ev Sahibi']} {int(mac['MS(Ev)'])}-{int(mac['MS(Dep)'])} {mac['Deplasman']} (<b>{int(mac['Toplam_Skor'])}</b>)\n"
    
    mesaj += "\n"
    
    # Çekişmeli maçlar
    mesaj += f"⚔️ <b>Çekişmeli Maçlar</b>\n"
    cekismeli = completed.nsmallest(3, 'Skor_Farkı')
    for _, mac in cekismeli.iterrows():
        mesaj += f"• {mac['Ev Sahibi']} {int(mac['MS(Ev)'])}-{int(mac['MS(Dep)'])} {mac['Deplasman']} (<b>{int(mac['Skor_Farkı'])} fark</b>)\n"
    
    mesaj += "\n"
    
    # Performans liderleri
    if not gunluk and len(completed) > 10:
        mesaj += f"🏆 <b>Performans Liderleri</b>\n"
        
        # Galibiyet liderleri
        takim_stats = completed['Kazanan'].value_counts()
        if len(takim_stats) > 0:
            lider = takim_stats.index[0]
            galibiyet = takim_stats.iloc[0]
            mesaj += f"• En çok kazanan: <b>{lider}</b> ({galibiyet} galibiyet)\n"
    
    mesaj += f"\n⏰ Güncelleme: {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}"
    
    return mesaj

def telegram_lig_karşılaştırma(df):
    """
    Ligler arası karşılaştırma için özet
    """
    mesaj = "<b>🏀 LİGLER KARŞILAŞTIRMA</b>\n"
    mesaj += "═" * 30 + "\n\n"
    
    completed = df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    
    lig_stats = completed.groupby('Lig').agg({
        'Toplam_Skor': ['count', 'mean'],
        'Skor_Farkı': 'mean'
    }).round(1)
    
    lig_stats.columns = ['Maç', 'Ort_Skor', 'Ort_Fark']
    
    for lig, stats in lig_stats.sort_values('Ort_Skor', ascending=False).iterrows():
        mesaj += f"<b>{lig}</b>\n"
        mesaj += f"   📍 Maç: {stats['Maç']} | ⚽ Ort: {stats['Ort_Skor']} | 📏 Fark: {stats['Ort_Fark']}\n\n"
    
    return mesaj

def telegram_takim_ozet(df, takim_adi):
    """
    Belirli bir takım için özet
    """
    takim_maclar = df[((df['Ev Sahibi'] == takim_adi) | (df['Deplasman'] == takim_adi)) & 
                     df['MS(Ev)'].notna()]
    
    if len(takim_maclar) == 0:
        return f"❌ <b>{takim_adi}</b> takımına ait maç bulunamadı"
    
    mesaj = f"<b>🏀 {takim_adi} TAKIM ÖZETİ</b>\n"
    mesaj += "═" * 30 + "\n\n"
    
    # Genel bilgiler
    galibiyet = 0
    toplam_mac = len(takim_maclar)
    
    for _, mac in takim_maclar.iterrows():
        if mac['Kazanan'] == takim_adi:
            galibiyet += 1
    
    galibiyet_yuzde = (galibiyet / toplam_mac * 100) if toplam_mac > 0 else 0
    
    mesaj += f"📊 <b>Genel Performans</b>\n"
    mesaj += f"• Toplam Maç: {toplam_mac}\n"
    mesaj += f"• Galibiyet: {galibiyet}\n"
    mesaj += f"• Galibiyet %: {galibiyet_yuzde:.1f}%\n\n"
    
    # Son 5 maç
    mesaj += f"📅 <b>Son 5 Maç</b>\n"
    son_maclar = takim_maclar.nlargest(5, 'Tarih')
    
    for _, mac in son_maclar.iterrows():
        ev_skor = int(mac['MS(Ev)'])
        dep_skor = int(mac['MS(Dep)'])
        
        if mac['Ev Sahibi'] == takim_adi:
            skor = f"<b>{ev_skor}-{dep_skor}</b>"
            sonuc = "✅" if ev_skor > dep_skor else "❌" if ev_skor < dep_skor else "⚪"
            rakip = mac['Deplasman']
            yer = "🏠"
        else:
            skor = f"{dep_skor}-<b>{ev_skor}</b>"
            sonuc = "✅" if dep_skor > ev_skor else "❌" if dep_skor < ev_skor else "⚪"
            rakip = mac['Ev Sahibi']
            yer = "✈️"
        
        tarih = mac['Tarih'].strftime('%d.%m')
        mesaj += f"{sonuc} {yer} {rakip} {skor} ({tarih})\n"
    
    return mesaj

def telegram_rapor_ornekleri(df):
    """
    Telegram rapor örneklerini göster
    """
    print("\n" + "="*50)
    print("TELEGRAM RAPOR ÖRNEKLERİ")
    print("="*50)
    
    # Günlük özet
    gunluk_ozet = telegram_raporu(df, gunluk=True)
    print("\n1. GÜNLÜK ÖZET:")
    print("-" * 30)
    print(gunluk_ozet)
    
    # Lig raporu
    lig_rapor = telegram_raporu(df, "Eurolig")
    print("\n2. LİG RAPORU:")
    print("-" * 30)
    print(lig_rapor)
    
    # Lig karşılaştırma
    lig_karsilastirma = telegram_lig_karşılaştırma(df)
    print("\n3. LİG KARŞILAŞTIRMA:")
    print("-" * 30)
    print(lig_karsilastirma)
    
    # Takım özeti
    takim_ozet = telegram_takim_ozet(df, "Fenerbahçe Beko")
    print("\n4. TAKIM ÖZETİ:")
    print("-" * 30)
    print(takim_ozet)

# Ana menüye Telegram seçeneği ekle
def telegram_menu_ekle(df):
    """
    Ana menüye Telegram rapor seçeneklerini ekler
    """
    while True:
        print(f"\n{' TELEGRAM RAPOR MENÜSÜ ':=^50}")
        print("1. Günlük Özet Raporu")
        print("2. Lig Raporu")
        print("3. Tüm Ligler Karşılaştırma")
        print("4. Takım Özeti")
        print("5. Rapor Örneklerini Gör")
        print("6. Ana Menüye Dön")
        
        secim = input("\nSeçiminiz (1-6): ").strip()
        
        if secim == '1':
            rapor = telegram_raporu(df, gunluk=True)
            print("\n" + "="*50)
            print("GÜNLÜK ÖZET RAPORU")
            print("="*50)
            print(rapor)
            
            # Panoya kopyala önerisi
            print(f"\n💡 Bu raporu kopyalayı Telegram'a yapıştırabilirsiniz!")
            
        elif secim == '2':
            print(f"\nMevcut ligler: {', '.join(df['Lig'].unique())}")
            lig_sec = input("Lig adını girin: ").strip()
            if lig_sec in df['Lig'].unique():
                rapor = telegram_raporu(df, lig_sec)
                print(f"\n{' LİG RAPORU ':=^50}")
                print(rapor)
            else:
                print("❌ Geçersiz lig adı!")
                
        elif secim == '3':
            rapor = telegram_lig_karşılaştırma(df)
            print(f"\n{' LİG KARŞILAŞTIRMA ':=^50}")
            print(rapor)
            
        elif secim == '4':
            takim_sec = input("Takım adını girin: ").strip()
            rapor = telegram_takim_ozet(df, takim_sec)
            print(f"\n{' TAKIM ÖZETİ ':=^50}")
            print(rapor)
            
        elif secim == '5':
            telegram_rapor_ornekleri(df)
            
        elif secim == '6':
            break
        else:
            print("❌ Geçersiz seçim!")

# 8. ANA MENÜ
def ana_menu():
    """
    Güncellenmiş ana menü
    """
    while True:
        print(f"\n{' MENÜ ':=^50}")
        print("1. Tüm ligleri göster")
        print("2. Belirli bir lig raporu")
        print("3. Takım detayları")
        print("4. Lig Bazlı Detaylı Analiz")
        print("5. Lig Karşılaştırma Grafikleri")
        print("6. Telegram Raporları")
        print("7. Çıkış")

        secim = input("\nSeçiminiz (1-7): ").strip()
        
        if secim == '1':
            lig_analizi(df)
        elif secim == '2':
            print(f"\nMevcut ligler: {', '.join(df['Lig'].unique())}")
            lig_sec = input("Lig adını girin: ").strip()
            if lig_sec in df['Lig'].unique():
                detayli_lig_raporu(lig_sec)
                # Yeni: Lig özel takım performansı
                lig_bazli_takim_performansi(df, lig_sec)
            else:
                print("❌ Geçersiz lig adı!")
        elif secim == '3':
            takim_sec = input("Takım adını girin: ").strip()
            team_matches = df[(df['Ev Sahibi'] == takim_sec) | (df['Deplasman'] == takim_sec)]
            if len(team_matches) > 0:
                print(f"\n{takim_sec} takımının maçları:")
                print(team_matches[['Tarih', 'Lig', 'Ev Sahibi', 'MS(Ev)', 'MS(Dep)', 'Deplasman']].to_string(index=False))
            else:
                print("❌ Takım bulunamadı!")
        elif secim == '4':
            lig_ozel_analiz(df)
        elif secim == '5':
            lig_stats = lig_karsilastirma_grafik(df)
            print("\n📈 Lig İstatistikleri:")
            print(lig_stats)
        elif secim == '6':
            telegram_menu_ekle(df)
        elif secim == '7':
            print("👋 Program sonlandırıldı!")
            break
        else:
            print("❌ Geçersiz seçim!")

# PROGRAMI BAŞLAT
if __name__ == "__main__":
    ana_menu()

    # SON ÖZET
    print(f"\n{' ANALİZ TAMAMLANDI ':=^50}")
    print(f"📊 Toplam işlenen satır: {len(df)}")
    print(f"🏆 Analiz edilen lig sayısı: {len(df['Lig'].unique())}")
    print(f"📅 Veri aralığı: {df['Tarih'].min().strftime('%d.%m.%Y')} - {df['Tarih'].max().strftime('%d.%m.%Y')}")
    print(f"👋 Teşekkürler!")
