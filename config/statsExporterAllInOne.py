import os
import re
import argparse
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE_DIR = r'D:\ml-agents'
OUTPUT_DIR = r'D:\ml-agents\Wykresy_Analiza'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'all_training_data.csv')

# DODANO: Policy/Entropy
METRICS_TO_EXTRACT = [
    'Environment/Cumulative Reward',
    'Environment/Episode Length',
    'Policy/Entropy' 
]

def find_tfevents_files(base_dir, filter_keyword='base'):
    found_runs = []
    excluded_keywords = ['randomterrain', 'looking', 'expanded']
    
    for root, dirs, files in os.walk(base_dir):
        if any(f.startswith('events.out.tfevents') for f in files):
            level_0 = os.path.basename(root)
            level_1_path = os.path.dirname(root)
            level_1 = os.path.basename(level_1_path)
            level_2_path = os.path.dirname(level_1_path)
            level_2 = os.path.basename(level_2_path)
            
            if level_2.lower() != "new": continue
            
            match = re.match(r'^(.+)_(\d+)_model_(\d+)_tpl$', level_1)
            
            if match:
                algo_name = match.group(1)
                run_number = match.group(2)
                model_number = match.group(3)
                algo_name_lower = algo_name.lower()
                
                if filter_keyword == 'base':
                    if any(ex in algo_name_lower for ex in excluded_keywords): continue
                else:
                    if filter_keyword.lower() not in algo_name_lower: continue

                found_runs.append({
                    'path': root, 'algo': algo_name, 'run': run_number, 'model': model_number
                })
    return found_runs

def extract_data_to_dataframe(runs_info):
    all_data = []
    for run in runs_info:
        print(f" -> Pobieranie: {run['algo']} | Run: {run['run']} | Model: {run['model']}")
        event_acc = EventAccumulator(run['path'], size_guidance={'scalars': 0})
        event_acc.Reload()
        
        tags = event_acc.Tags().get('scalars', [])
        for tag in tags:
            if tag in METRICS_TO_EXTRACT:
                events = event_acc.Scalars(tag)
                for e in events:
                    all_data.append({
                        'Algorithm': run['algo'],
                        'Run': run['run'],
                        'Model': run['model'],
                        'Tag': tag,
                        'Step': e.step,
                        'Wall_Time': e.wall_time,
                        'Value': e.value
                    })
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=str, default='base')
    parser.add_argument('--append', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    runs = find_tfevents_files(BASE_DIR, filter_keyword=args.filter)
    
    if not runs:
        print("Nie znaleziono logów.")
    else:
        new_df = extract_data_to_dataframe(runs)
        if args.append and os.path.exists(OUTPUT_CSV):
            existing_df = pd.read_csv(OUTPUT_CSV, sep=';', decimal=',')
            final_df = pd.concat([existing_df, new_df]).drop_duplicates(
                subset=['Algorithm', 'Run', 'Model', 'Tag', 'Step'], keep='last'
            )
        else:
            final_df = new_df
            
        final_df.to_csv(OUTPUT_CSV, index=False, sep=';', decimal=',')
        print(f"\nGotowe! Zapisano dane w pliku: {OUTPUT_CSV}")