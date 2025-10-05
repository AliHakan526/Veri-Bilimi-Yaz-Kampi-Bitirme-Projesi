

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('50_Startups.csv')



def RDAndProfit():
    plt.figure(figsize=(8,6))
    plt.scatter(data['R&D Spend'], data['Profit'], color='blue', alpha=0.7)
    plt.title("R&D Harcaması ve Kâr Arasındaki İlişki")
    plt.xlabel("R&D Harcaması")
    plt.ylabel("Profit (Kâr)")
    plt.grid(True)
    plt.show()






def ManagementAndProfit():
    plt.figure(figsize=(8,6))
    plt.scatter(data['Administration'], data['Profit'], color='green', alpha=0.7)
    plt.title("Yönetim Harcamaları ve Kâr Arasındaki İlişki")
    plt.xlabel("Administration (Yönetim Harcaması)")
    plt.ylabel("Profit (Kâr)")
    plt.grid(True)
    plt.show()








def StateAndProfit():
    avg_profit_by_state = data.groupby('State')['Profit'].mean()
    plt.figure(figsize=(8,6))
    avg_profit_by_state.plot(kind='bar', color='orange', alpha=0.8)
    plt.title("Eyaletlere Göre Ortalama Kâr")
    plt.xlabel("State (Eyalet)")
    plt.ylabel("Average Profit (Ortalama Kâr)")
    plt.grid(axis='y')
    plt.show()






def boxPlot():
    plt.figure(figsize=(8,6))
    plt.boxplot([data['R&D Spend'], data['Administration'], data['Marketing Spend']],
            tick_labels=['R&D Spend', 'Administration', 'Marketing Spend'],
            patch_artist=True)

    plt.title("Harcama Türlerinin Karşılaştırması (Boxplot)")
    plt.ylabel("Harcama Tutarı")
    plt.grid(axis='y')
    plt.show()











