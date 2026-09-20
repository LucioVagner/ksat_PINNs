import pandas as pd
import numpy as np
import re

def parse_depth_midpoint(s):
    """Parse strings like '0-20', '7-15', '3–8' into the midpoint depth (cm)."""
    if pd.isna(s):
        return np.nan
    s = str(s).replace('–', '-').strip()
    parts = re.split(r'-', s)
    parts = [p for p in parts if p.strip() != '']
    nums = [float(p) for p in parts]
    return float(np.mean(nums))

def load_data(path):
    df = pd.read_excel(path, sheet_name='Dados')
    df = df.dropna(subset=['Ksat (cm/dia) '])
    df = df.rename(columns={
        'Ksat (cm/dia) ': 'Ksat',
        'Densidade do solo (g/cm³)': 'Densidade',
        'Porosidade Total (cm3/cm3)': 'Porosidade',
        'Macroporosidade (cm3/cm3)': 'Macroporosidade',
        'Teor de carbono orgânico (%)': 'Carbono',
        'Tipo de uso do solo': 'Uso',
        'Prof. (cm)': 'Prof_raw',
    })
    df['Profundidade'] = df['Prof_raw'].apply(parse_depth_midpoint)

    cont_features = ['Areia', 'Silte', 'Argila', 'Densidade', 'Porosidade',
                      'Macroporosidade', 'Carbono', 'Profundidade']
    df = df.dropna(subset=cont_features + ['Ksat'])

    # one-hot encode land use
    uso_dummies = pd.get_dummies(df['Uso'], prefix='Uso').astype(float)
    cat_features = list(uso_dummies.columns)

    X_cont = df[cont_features].values.astype(float)
    X_cat = uso_dummies.values.astype(float)
    y_ksat = df['Ksat'].values.astype(float)
    y_log = np.log10(y_ksat)

    return {
        'df': df.reset_index(drop=True),
        'X_cont': X_cont,
        'X_cat': X_cat,
        'cont_features': cont_features,
        'cat_features': cat_features,
        'y_ksat': y_ksat,
        'y_log': y_log,
    }

if __name__ == '__main__':
    d = load_data('/mnt/user-data/uploads/data.xlsx')
    print('n amostras:', d['X_cont'].shape[0])
    print('features continuas:', d['cont_features'])
    print('features categoricas:', d['cat_features'])
    print('X_cont shape:', d['X_cont'].shape, 'X_cat shape:', d['X_cat'].shape)
    print('y_log stats: min', d['y_log'].min(), 'max', d['y_log'].max(), 'mean', d['y_log'].mean())
