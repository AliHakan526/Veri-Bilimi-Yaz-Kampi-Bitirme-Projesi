
import pandas as pd
import numpy as np

data = pd.read_csv('country.csv')

print(data.sort_values("Population",ascending=False))


def sortGDP():
    print(data.sort_values("GDP ($ per capita)"))

def selectPopulation():
    print(data[data["Population"]>10000000])

def first5Literacy():
    print(data.sort_values("Literacy (%)").head(5))

def selectGSYIH():
    new_frame = data[data["GDP ($ per capita)"]>10000]
    print("the new frame:")
    print(new_frame)

def selectDensity():
    new_frame = data.sort_values("Pop. Density (per sq. mi.)").head(10)
    print(new_frame)


