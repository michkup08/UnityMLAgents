import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==========================================
# 1. Ścieżka do logów OBECNEGO algorytmu (folder Behavior Name, w którym jest plik tfevents)
LOG_DIR = r'C:\Users\Michal\Desktop\Unity\Studia\UnityMLAgents\config\droQ\results\droQ_randomTerrain_looking_expanded_01_model_00\Vertebate'

# 2. Nazwa tego algorytmu (Będzie nagłówkiem kolumny w Excelu)
ALGORITHM_NAME = 'droQ_expanded_looking'

# 3. Folder, do którego skrypt wrzuci wygenerowane, osobne tabelki
OUTPUT_DIR = r'C:\Users\Michal\Desktop\Unity\Studia\UnityMLAgents\Wykresy_looking'
# ==========================================


def append_tensorboard_data(log_dir, algo_name, output_dir):
    # Tworzymy folder docelowy, jeśli nie istnieje
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Wczytanie logów
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Pobranie wszystkich dostępnych tagów (wykresów z TensorBoarda)
    tags = event_acc.Tags().get('scalars', [])
    
    if not tags:
        print(f"Nie znaleziono danych typu 'scalar' w folderze: {log_dir}")
        return

    print(f"Rozpoczynam przetwarzanie algorytmu: {algo_name}...")

    for tag in tags:
        # Wyciągamy dane dla danego parametru
        events = event_acc.Scalars(tag)
        
        # Tworzymy tabelę z indeksami jako 'Step' i wartością w kolumnie o nazwie algorytmu
        df = pd.DataFrame([(e.step, e.value) for e in events], columns=['Step', algo_name])
        df.set_index('Step', inplace=True)

        # Sanityzacja nazwy pliku (Windows nie pozwala na '/' w nazwach plików)
        safe_tag_name = tag.replace('/', '_').replace('\\', '_').replace(' ', '')
        file_path = os.path.join(output_dir, f"{safe_tag_name}.csv")

        # Jeśli plik dla tej metryki już istnieje, wczytujemy go i dołączamy nową kolumnę
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path, index_col='Step', sep=';', decimal=',')
            
            # Usuwamy starą kolumnę o tej samej nazwie, jeśli uruchamiasz ten sam algorytm 2 razy
            if algo_name in existing_df.columns:
                existing_df.drop(columns=[algo_name], inplace=True)
                
            # Złączenie danych. how='outer' gwarantuje, że nie zgubimy kroków, jeśli 
            # jeden algorytm uczył się dłużej (miał więcej Stepów) niż inny.
            combined_df = existing_df.join(df, how='outer')
            combined_df.to_csv(file_path, sep=';', decimal=',')
        else:
            # Jeśli to pierwszy algorytm, po prostu tworzymy nowy plik
            df.to_csv(file_path, sep=';', decimal=',')

    print(f"Gotowe! Dane dla '{algo_name}' zostały dopisane do plików w: {output_dir}")


append_tensorboard_data(LOG_DIR, ALGORITHM_NAME, OUTPUT_DIR)