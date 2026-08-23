import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV = r'D:\UnityMLAgentsResults\ml-agents\Wykresy_Analiza\all_training_data.csv'
BASE_OUTPUT_DIR = r'D:\UnityMLAgentsResults\ml-agents\Wykresy_Analiza'

MAX_STEPS = 10000000     
STEP_ROUNDING = 10000    
SMOOTHING_WINDOW = 20    

DIRS = {
    'agg': os.path.join(BASE_OUTPUT_DIR, '1_Zestawienie_Zagregowane'),
    'time': os.path.join(BASE_OUTPUT_DIR, '2_Efektywnosc_Czasowa'),
    'len': os.path.join(BASE_OUTPUT_DIR, '3_Dlugosc_Epizodu'),
    'ent': os.path.join(BASE_OUTPUT_DIR, '4_Entropia_Polityki'),
    'runs': os.path.join(BASE_OUTPUT_DIR, '5_Poszczegolne_Treningi_Nagroda'),
    'runs_len': os.path.join(BASE_OUTPUT_DIR, '6_Poszczegolne_Treningi_Dlugosc'),
    'runs_ent': os.path.join(BASE_OUTPUT_DIR, '7_Poszczegolne_Treningi_Entropia')
}

def setup_directories(suffix=""):
    actual_dirs = {k: v + (f"_{suffix}" if suffix else "") for k, v in DIRS.items()}
    for d in actual_dirs.values():
        if not os.path.exists(d):
            os.makedirs(d)
    return actual_dirs

def smooth_data(df, group_cols):
    df_smoothed = df.sort_values(by=group_cols + ['Step_Binned']).copy()
    df_smoothed['Value_Smoothed'] = df_smoothed.groupby(group_cols)['Value'].transform(
        lambda x: x.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    )
    return df_smoothed

def generate_plots(filter_keyword):
    if not os.path.exists(INPUT_CSV):
        print("Nie znaleziono pliku CSV. Uruchom ekstrakcję.")
        return

    out_dirs = setup_directories(filter_keyword)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    print("Wczytywanie i przygotowywanie danych...")
    df = pd.read_csv(INPUT_CSV, sep=';', decimal=',')
    df['Algorithm'] = df['Algorithm'].replace({'droq': 'droQ'})
    
    df = df[df['Step'] <= MAX_STEPS]

    if filter_keyword == 'base':
        df = df[~df['Algorithm'].str.contains('randomTerrain|looking|expanded', case=False, na=False)]
    elif filter_keyword == 'all':
        pass 
    else:
        df = df[df['Algorithm'].str.contains(filter_keyword, case=False, na=False)]

    algos_in_analysis = df['Algorithm'].unique()
    print(f"Algorytmy w analizie: {algos_in_analysis}")
    
    if len(algos_in_analysis) == 0:
        print("BŁĄD: Brak danych do narysowania po zastosowaniu filtra! Kończę pracę.")
        return

    min_times = df.groupby(['Algorithm', 'Run'])['Wall_Time'].transform('min')
    df['Relative_Time_Hrs'] = (df['Wall_Time'] - min_times) / 3600.0
    df['Step_Binned'] = (df['Step'] // STEP_ROUNDING) * STEP_ROUNDING

    # --- NAGRODA ---
    print("Generowanie wykresów zagregowanych (Nagroda)...")
    df_reward = df[df['Tag'] == 'Environment/Cumulative Reward'].copy()
    if not df_reward.empty:
        df_reward = df_reward.groupby(['Algorithm', 'Run', 'Step_Binned'], as_index=False)['Value'].mean()
        df_reward = smooth_data(df_reward, ['Algorithm', 'Run'])

        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df_reward, x='Step_Binned', y='Value_Smoothed', hue='Algorithm', errorbar=('ci', 95), estimator='median', linewidth=2.5)
        plt.xlabel("Liczba Kroków", fontsize=12)
        plt.ylabel("Skumulowana Nagroda", fontsize=12)
        plt.xlim(0, MAX_STEPS)
        plt.legend(title="Algorytm")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dirs['agg'], f"Porownanie_Nagrody_{filter_keyword}.png"), dpi=300)
        plt.close()

    print("Generowanie wykresów dla poszczególnych treningów (Nagroda)...")
    if not df_reward.empty:
        for algo in algos_in_analysis:
            df_algo = df_reward[df_reward['Algorithm'] == algo]
            
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=df_algo, 
                x='Step_Binned', 
                y='Value_Smoothed', 
                hue='Run', 
                palette='tab10',
                linewidth=2
            )
            plt.xlabel("Liczba Kroków (Steps)", fontsize=12)
            plt.ylabel("Nagroda (Cumulative Reward)", fontsize=12)
            plt.xlim(0, MAX_STEPS)
            plt.legend(title="Numer treningu (Run)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dirs['runs'], f"{algo}_runs_reward_10m.png"), dpi=300)
            plt.close()

    # --- CZAS NAUKI ---
    print("Generowanie wykresów czasu nauki...")
    df_time = df[df['Tag'] == 'Environment/Cumulative Reward'].copy()
    if not df_time.empty:
        df_time['Time_Binned'] = (df_time['Relative_Time_Hrs'] // 0.1) * 0.1
        df_time = df_time.groupby(['Algorithm', 'Run', 'Time_Binned'], as_index=False)['Value'].mean()
        df_time = df_time.sort_values(by=['Algorithm', 'Run', 'Time_Binned'])
        df_time['Value_Smoothed'] = df_time.groupby(['Algorithm', 'Run'])['Value'].transform(lambda x: x.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean())

        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df_time, x='Time_Binned', y='Value_Smoothed', hue='Algorithm', errorbar=('ci', 95), estimator='median', linewidth=2.5)
        plt.xlabel("Czas Nauki (Godziny)", fontsize=12)
        plt.ylabel("Skumulowana Nagroda", fontsize=12)
        plt.legend(title="Algorytm")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dirs['time'], f"Czas_Nauki_{filter_keyword}.png"), dpi=300)
        plt.close()

    # --- DŁUGOŚĆ EPIZODU ---
    print("Generowanie wykresów długości epizodu...")
    df_len = df[df['Tag'] == 'Environment/Episode Length'].copy()
    if not df_len.empty:
        df_len = df_len.groupby(['Algorithm', 'Run', 'Step_Binned'], as_index=False)['Value'].mean()
        df_len = smooth_data(df_len, ['Algorithm', 'Run'])

        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df_len, x='Step_Binned', y='Value_Smoothed', hue='Algorithm', errorbar=('ci', 95), estimator='median', linewidth=2.5)
        plt.xlabel("Liczba Kroków", fontsize=12)
        plt.ylabel("Długość Epizodu", fontsize=12)
        plt.xlim(0, MAX_STEPS)
        plt.legend(title="Algorytm")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dirs['len'], f"Dlugosc_Epizodu_{filter_keyword}.png"), dpi=300)
        plt.close()

    print("Generowanie wykresów dla poszczególnych treningów (Długość Epizodu)...")
    if not df_len.empty:
        for algo in algos_in_analysis:
            df_algo = df_len[df_len['Algorithm'] == algo]
            
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=df_algo, 
                x='Step_Binned', 
                y='Value_Smoothed', 
                hue='Run', 
                palette='tab10',
                linewidth=2
            )
            plt.xlabel("Liczba Kroków (Steps)", fontsize=12)
            plt.ylabel("Długość Epizodu", fontsize=12)
            plt.xlim(0, MAX_STEPS)
            plt.legend(title="Numer treningu (Run)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dirs['runs_len'], f"{algo}_runs_length_10m.png"), dpi=300)
            plt.close()

    # --- ENTROPIA ---
    print("Generowanie wykresów entropii...")
    df_ent = df[df['Tag'] == 'Policy/Entropy'].copy()
    if df_ent.empty:
        print(" UWAGA: W przefiltrowanych danych nie ma metryki 'Policy/Entropy'.")
    else:
        df_ent = df_ent.groupby(['Algorithm', 'Run', 'Step_Binned'], as_index=False)['Value'].mean()
        df_ent = smooth_data(df_ent, ['Algorithm', 'Run'])

        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df_ent, x='Step_Binned', y='Value_Smoothed', hue='Algorithm', errorbar=('ci', 95), estimator='mean', linewidth=2)
        plt.xlabel("Liczba Kroków", fontsize=12)
        plt.ylabel("Wartość Entropii", fontsize=12)
        plt.xlim(0, MAX_STEPS)
        plt.legend(title="Algorytm")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dirs['ent'], f"Entropia_{filter_keyword}.png"), dpi=300)
        plt.close()

    print("Generowanie wykresów dla poszczególnych treningów (Entropia)...")
    if not df_ent.empty:
        for algo in algos_in_analysis:
            df_algo = df_ent[df_ent['Algorithm'] == algo]
            
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=df_algo, 
                x='Step_Binned', 
                y='Value_Smoothed', 
                hue='Run', 
                palette='tab10',
                linewidth=2
            )
            plt.xlabel("Liczba Kroków (Steps)", fontsize=12)
            plt.ylabel("Wartość Entropii", fontsize=12)
            plt.xlim(0, MAX_STEPS)
            plt.legend(title="Numer treningu (Run)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dirs['runs_ent'], f"{algo}_runs_entropy_10m.png"), dpi=300)
            plt.close()

    print("\nZakończono generowanie wykresów!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=str, default='base', 
                        help="Co narysować? np. 'base', 'randomTerrain', 'looking', lub 'all' aby narysować wszystko naraz.")
    args = parser.parse_args()
    
    generate_plots(args.filter)