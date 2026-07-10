import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR = r'C:\Users\Michal\Desktop\Unity\Studia\UnityMLAgents\config\droQ\results\droQ_randomTerrain_looking_expanded_01_model_00\Vertebate'
ALGORITHM_NAME = 'droQ_expanded_looking'
OUTPUT_DIR = r'C:\Users\Michal\Desktop\Unity\Studia\UnityMLAgents\Wykresy_looking'


def append_tensorboard_data(log_dir, algo_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    tags = event_acc.Tags().get('scalars', [])
    
    if not tags:
        print(f"Nie znaleziono danych typu 'scalar' w folderze: {log_dir}")
        return

    print(f"Rozpoczynam przetwarzanie algorytmu: {algo_name}...")

    for tag in tags:
        events = event_acc.Scalars(tag)
        
        df = pd.DataFrame([(e.step, e.value) for e in events], columns=['Step', algo_name])
        df.set_index('Step', inplace=True)

        safe_tag_name = tag.replace('/', '_').replace('\\', '_').replace(' ', '')
        file_path = os.path.join(output_dir, f"{safe_tag_name}.csv")

        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path, index_col='Step', sep=';', decimal=',')
            
            if algo_name in existing_df.columns:
                existing_df.drop(columns=[algo_name], inplace=True)
                
            combined_df = existing_df.join(df, how='outer')
            combined_df.to_csv(file_path, sep=';', decimal=',')
        else:
            df.to_csv(file_path, sep=';', decimal=',')

    print(f"Gotowe! Dane dla '{algo_name}' zostały dopisane do plików w: {output_dir}")


append_tensorboard_data(LOG_DIR, ALGORITHM_NAME, OUTPUT_DIR)