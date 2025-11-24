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

# Kullanıcı etkileşimi
while True:
    print(f"\n{' MENÜ ':=^50}")
    print("1. Tüm ligleri göster")
    print("2. Belirli bir lig raporu")
    print("3. Takım detayları")
    print("4. Telegram Raporları")
    print("5. Çıkış")

    secim = input("\nSeçiminiz (1-5): ").strip()
    
    if secim == '1':
        lig_analizi(df)
    elif secim == '2':
        print(f"\nMevcut ligler: {', '.join(df['Lig'].unique())}")
        lig_sec = input("Lig adını girin: ").strip()
        if lig_sec in df['Lig'].unique():
            detayli_lig_raporu(lig_sec)
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
        telegram_menu_ekle(df)
    elif secim == '5':
        print("👋 Program sonlandırıldı!")
        break
    else:
        print("❌ Geçersiz seçim!")
# SON ÖZET
print(f"\n{' ANALİZ TAMAMLANDI ':=^50}")
print(f"📊 Toplam işlenen satır: {len(df)}")
print(f"🏆 Analiz edilen lig sayısı: {len(df['Lig'].unique())}")
print(f"📅 Veri aralığı: {df['Tarih'].min().strftime('%d.%m.%Y')} - {df['Tarih'].max().strftime('%d.%m.%Y')}")
print(f"👋 Teşekkürler!")

🏀 Eurolig Günlük Özet
══════════════════════════════════

📊 Genel Bilgiler
• Maç Sayısı: 8
• Ort. Skor: 168.5
• Ort. Fark: 12.3

🔥 Son Maçlar
✅ 🏠 Fenerbahçe 89-78 Barcelona (20:30)
❌ ✈️ Real Madrid 95-102 Anadolu Efes (22:15)

🎯 Yüksek Skorlu Maçlar
• Olympiakos 112-108 AS Monaco (204)
• Barcelona 98-105 Real Madrid (203)
