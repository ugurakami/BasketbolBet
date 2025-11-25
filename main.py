import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os

# Görselleştirme kütüphanelerini kontrol et
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  Görselleştirme kütüphaneleri yüklenmemiş. Grafikler gösterilmeyecek.")

warnings.filterwarnings('ignore')

print("🏀 Basketbol Fikstür Analiz Programı")
print("=" * 50)

# Veriyi yükle
try:
    if os.path.exists('BasketbolFikstür - Sayfa1.tsv'):
        df = pd.read_csv('BasketbolFikstür - Sayfa1.tsv', sep='\t', encoding='utf-8')
        print("✅ Veri başarıyla yüklendi")
    else:
        # Örnek veri oluştur
        print("📝 Örnek veri oluşturuluyor...")
        data = {
            'Tarih': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18'],
            'Lig': ['Eurolig', 'NBA', 'Eurolig', 'NBA'],
            'Ev Sahibi': ['Fenerbahçe Beko', 'Lakers', 'Anadolu Efes', 'Warriors'],
            'MS(Ev)': [85, 110, 92, 105],
            'MS(Dep)': [78, 105, 88, 98],
            'Deplasman': ['Real Madrid', 'Celtics', 'Barcelona', 'Bulls'],
            'İY(Ev)': [45, 55, 48, 52],
            'İY(Dep)': [40, 50, 44, 45]
        }
        df = pd.DataFrame(data)
        df.to_csv('BasketbolFikstür - Sayfa1.tsv', sep='\t', index=False)
        print("✅ Örnek veri oluşturuldu ve kaydedildi")
        
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
    df['Tarih'] = pd.to_datetime(df['Tarih'], errors='coerce')
    
    # Skorları sayısala çevir
    score_columns = ['MS(Ev)', 'MS(Dep)', 'İY(Ev)', 'İY(Dep)']
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Toplam skorları hesapla
    df['Toplam_Skor'] = df['MS(Ev)'] + df['MS(Dep)']
    df['Skor_Farkı'] = abs(df['MS(Ev)'] - df['MS(Dep)'])
    
    # Kazananı belirle
    df['Kazanan'] = np.where(df['MS(Ev)'] > df['MS(Dep)'], df['Ev Sahibi'], 
                            np.where(df['MS(Dep)'] > df['MS(Ev)'], df['Deplasman'], 'Berabere'))
    
    return df

df = clean_data(df)
print("✅ Veri temizleme tamamlandı")

# 1. GENEL İSTATİSTİKLER
def genel_istatistikler(df):
    print("\n📈 GENEL İSTATİSTİKLER")
    print("-" * 30)
    
    completed_matches = df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    print(f"Tamamlanmış Maç Sayısı: {len(completed_matches)}")
    
    if len(completed_matches) > 0:
        print(f"Ortalama Toplam Skor: {completed_matches['Toplam_Skor'].mean():.1f}")
        print(f"Ortalama Skor Farkı: {completed_matches['Skor_Farkı'].mean():.1f}")
        print(f"Maksimum Skor: {completed_matches['Toplam_Skor'].max()}")
        print(f"Minimum Skor: {completed_matches['Toplam_Skor'].min()}")
    else:
        print("Henüz tamamlanmış maç bulunmuyor.")

genel_istatistikler(df)

# 2. LİG BAZLI ANALİZ
def lig_analizi(df):
    print("\n🏆 LİG BAZLI ANALİZ")
    print("-" * 30)
    
    completed = df.dropna(subset=['Toplam_Skor'])
    
    if len(completed) > 0:
        lig_stats = completed.groupby('Lig').agg({
            'Toplam_Skor': ['count', 'mean', 'max'],
            'Skor_Farkı': 'mean'
        }).round(1)
        
        lig_stats.columns = ['Maç_Sayısı', 'Ort_Skor', 'Maks_Skor', 'Ort_Fark']
        print(lig_stats.sort_values('Maç_Sayısı', ascending=False))
    else:
        print("Henüz tamamlanmış maç bulunmuyor.")

lig_analizi(df)

# 3. GÖRSELLEŞTİRMELER (sadece kütüphaneler yüklüyse)
def create_visualizations(df):
    if not VISUALIZATION_AVAILABLE:
        print("\n⚠️  Görselleştirme kütüphaneleri yüklenmemiş. Grafikler atlanıyor.")
        return
        
    print("\n📊 GÖRSELLEŞTİRMELER OLUŞTURULUYOR...")
    
    completed = df.dropna(subset=['Toplam_Skor'])
    
    if len(completed) == 0:
        print("Görselleştirme için yeterli veri yok.")
        return
    
    try:
        # Basit bir grafik oluştur
        plt.figure(figsize=(10, 6))
        
        # Lig bazlı ortalama skorlar
        lig_means = completed.groupby('Lig')['Toplam_Skor'].mean().sort_values(ascending=False)
        plt.subplot(1, 2, 1)
        lig_means.plot(kind='bar', color='lightblue', edgecolor='black')
        plt.title('Liglere Göre Ortalama Skor')
        plt.xlabel('Lig')
        plt.ylabel('Ortalama Skor')
        plt.xticks(rotation=45)
        
        # Skor dağılımı
        plt.subplot(1, 2, 2)
        plt.hist(completed['Toplam_Skor'], bins=10, color='lightcoral', edgecolor='black', alpha=0.7)
        plt.title('Toplam Skor Dağılımı')
        plt.xlabel('Toplam Skor')
        plt.ylabel('Frekans')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Görselleştirme hatası: {e}")

# 4. TELEGRAM RAPORU
def telegram_raporu(df):
    completed = df.dropna(subset=['MS(Ev)', 'MS(Dep)'])
    
    mesaj = "<b>🏀 BASKETBOL ANALİZ RAPORU</b>\n"
    mesaj += "═" * 30 + "\n\n"
    
    mesaj += f"📊 <b>Genel Bilgiler</b>\n"
    mesaj += f"• Toplam Maç: {len(df)}\n"
    mesaj += f"• Tamamlanan: {len(completed)}\n"
    
    if len(completed) > 0:
        mesaj += f"• Ort. Skor: {completed['Toplam_Skor'].mean():.1f}\n"
        mesaj += f"• Ort. Fark: {completed['Skor_Farkı'].mean():.1f}\n\n"
        
        # Son maçlar
        mesaj += f"🔥 <b>Son Maçlar</b>\n"
        son_maclar = completed.nlargest(3, 'Tarih')
        for _, mac in son_maclar.iterrows():
            mesaj += f"• {mac['Ev Sahibi']} {int(mac['MS(Ev)'])}-{int(mac['MS(Dep)'])} {mac['Deplasman']}\n"
    else:
        mesaj += "• Henüz tamamlanmış maç yok\n"
    
    mesaj += f"\n⏰ Üretilme: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
    
    return mesaj

# ANA PROGRAM
def main():
    # Görselleştirmeyi dene
    create_visualizations(df)
    
    # Telegram raporunu göster
    rapor = telegram_raporu(df)
    print("\n" + "="*50)
    print("TELEGRAM RAPORU:")
    print("="*50)
    print(rapor)
    
    # Dosya bilgisi
    print(f"\n💾 Çalışma dizini: {os.getcwd()}")
    print(f"📁 Dosyalar: {', '.join([f for f in os.listdir('.') if os.path.isfile(f)])}")

if __name__ == "__main__":
    main()
    print(f"\n{' PROGRAM TAMAMLANDI ':=^50}")
