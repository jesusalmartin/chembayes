from chembayes import optimize_experiment
import pandas as pd

df = pd.read_csv('C:/Users/jamar/Universitat de les Illes Balears/UIB-TEAMS_MATER_LAB - General/00_NOTEBOOKS/00_SHARED/NOTION/output/hkust.csv')

inputs = [
    'mL_HNO3',
    'EtOH/H2O',
    'Tiempo (h)',
]

outputs = {
    'S_BET_lavado': 0.75,
    'm_HKUST-1_lavado (g) ': 0.25
}

optimize_experiment(df=df, inputs=inputs, output='S_BET_lavado')