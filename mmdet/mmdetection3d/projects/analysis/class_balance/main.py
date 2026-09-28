import os
import pandas as pd
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm
import warnings
import argparse



# Quick guide:
# 1. Set BASE_DIR to your local NuScenes dataset, or pass --base_dir.
# 2. Add --full_dataset to analyze the complete dataset.
# 3. Set --out_dir to the directory for generated results.
# Example: python main.py --full_dataset --base_dir /path/to/nuscenes --out_dir /path/to/output
# Subset:  python main.py --base_dir /path/to/subset --out_dir /path/to/output
DEFAULT_BASE_DIR = '/path/to/nuscenes'  # Replace with your local NuScenes path

# Target domains
DOMAINS = [
    'boston_day_clear',
    'boston_day_rain',
    'singapore_day_clear',
    'singapore_night_clear',
    'singapore_night_rain'
]


# Visibility mapping
VISIBILITY_MAP = {
    1: '0-40%',
    2: '40-60%',
    3: '60-80%',
    4: '80-100%'
}

# Mapping from NuScenes granular classes to simplified classes
CLASS_MAPPING = {
    'vehicle.car': 'car',
    'vehicle.emergency.police': 'car',
    'vehicle.emergency.ambulance': 'car',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.personal_mobility': 'pedestrian',
    'human.pedestrian.stroller': 'pedestrian',
    'human.pedestrian.wheelchair': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
}

# The target classes you want to keep
TARGET_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

def map_scene_to_domain(nusc, scene):
    log = nusc.get('log', scene['log_token'])
    location = log['location']
    if 'boston' in location:
        city = 'boston'
    elif 'singapore' in location:
        city = 'singapore'
    else:
        return None
    
    description = scene['description'].lower()
    time = 'night' if 'night' in description else 'day'
    weather = 'rain' if 'rain' in description else 'clear'
    
    domain = f'{city}_{time}_{weather}'
    if domain in DOMAINS:
        return domain
    return None

def find_nuscenes_roots(domain_path):
    """
    Detects the NuScenes data structure in the domain path.
    Returns a list of paths containing v1.0-trainval or v1.0-mini folders.
    """
    roots = []
    
    # Check if domain_path directly contains NuScenes data
    if any(os.path.exists(os.path.join(domain_path, f'v1.0-{version}')) for version in ['trainval', 'mini']):
        roots.append(domain_path)
    else:
        # Check immediate subdirectories
        if os.path.exists(domain_path):
            for subdir in os.listdir(domain_path):
                subdir_path = os.path.join(domain_path, subdir)
                if os.path.isdir(subdir_path):
                    if any(os.path.exists(os.path.join(subdir_path, f'v1.0-{version}')) for version in ['trainval', 'mini']):
                        roots.append(subdir_path)
    
    return roots

def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize NuScenes class imbalance and visibility distributions. '
                                                 'Possible plot_dirs: per_class, normalized_per_class, per_class_train, per_class_val, '
                                                 'normalized_per_class_train, normalized_per_class_val, simple_per_class, '
                                                 'simple_normalized_per_class, simple_per_class_train, simple_per_class_val, '
                                                 'simple_normalized_per_class_train, simple_normalized_per_class_val, '
                                                 'simple_avg_per_class. '
                                                 'Use "all" or omit to plot all, "none" to plot only main and summary.')
    parser.add_argument('--base_dir', type=str, default=DEFAULT_BASE_DIR, help='Base directory containing the domain folders or the NuScenes root.')
    parser.add_argument('--out_dir', type=str, default='/YOUR_HOME_DIR/mmdet/scratch/debugging_class_count', help='Output directory for the plot.')
    parser.add_argument('--plot_dirs', nargs='*', default=['all'], help='Subdirectories to plot. Use "all" or omit for all, "none" for only main plots.')
    parser.add_argument('--full_dataset', action='store_true', help='Process the full NuScenes dataset and split logically by domains.')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Get the official list of scene names for the train split
    splits = create_splits_scenes()
    train_scenes = set(splits['train'])

    dirs_to_plot = set(args.plot_dirs)
    if 'all' in dirs_to_plot or not args.plot_dirs:
        dirs_to_plot = {'per_class', 'normalized_per_class', 'per_class_train', 'per_class_val',
                        'normalized_per_class_train', 'normalized_per_class_val', 'simple_per_class',
                        'simple_normalized_per_class', 'simple_per_class_train', 'simple_per_class_val',
                        'simple_normalized_per_class_train', 'simple_normalized_per_class_val',
                        'simple_avg_per_class'}
    elif 'none' in dirs_to_plot:
        dirs_to_plot = set()
    
    if args.full_dataset:
        # Full dataset mode
        version = 'v1.0-trainval' if os.path.exists(os.path.join(args.base_dir, 'v1.0-trainval')) else 'v1.0-mini'
        nusc = NuScenes(version=version, dataroot=args.base_dir, verbose=False)
        
        domain_data = {domain: {
            'train': {}, 
            'val': {}, 
            'simple_train': {cat: {} for cat in TARGET_CLASSES}, 
            'simple_val': {cat: {} for cat in TARGET_CLASSES}
        } for domain in DOMAINS}
        total_ann = {domain: 0 for domain in DOMAINS}
        total_ann_train = {domain: 0 for domain in DOMAINS}
        total_ann_val = {domain: 0 for domain in DOMAINS}
        total_ann_simple_train = {domain: 0 for domain in DOMAINS}
        total_ann_simple_val = {domain: 0 for domain in DOMAINS}
        # Track unique sample tokens per domain/split for avg-per-sample metric
        sample_tokens_per_domain = {domain: {'train': set(), 'val': set()} for domain in DOMAINS}
        
        for ann in tqdm(nusc.sample_annotation, desc="Processing full dataset"):
            category = ann['category_name']
            visibility = int(ann['visibility_token'])
            
            sample = nusc.get('sample', ann['sample_token'])
            scene_token = sample['scene_token']
            scene = nusc.get('scene', scene_token)
            
            domain = map_scene_to_domain(nusc, scene)
            if domain is None:
                continue
            
            scene_num = int(scene['name'].split('-')[1])
            if version == 'v1.0-mini':
                # Mini split is handled differently, usually specific scene names
                # But commonly mini_train is scenes 0-9ish. 
                # The safest way for mini is checking the built-in mini splits 
                # or keeping your < 200 logic ONLY for mini.
                is_train = scene_num < 200 
            else:
                # Check if the scene name exists in the official train list
                is_train = scene['name'] in train_scenes
            
            split = 'train' if is_train else 'val'
            sample_tokens_per_domain[domain][split].add(ann['sample_token'])
            
            counts = domain_data[domain][split]
            simple_counts = domain_data[domain][f'simple_{split}']
            
            if category not in counts:
                counts[category] = {}
            if visibility not in counts[category]:
                counts[category][visibility] = 0
            counts[category][visibility] += 1
            total_ann[domain] += 1
            if split == 'train':
                total_ann_train[domain] += 1
            else:
                total_ann_val[domain] += 1
            
            if category in CLASS_MAPPING:
                simple_cat = CLASS_MAPPING[category]
                if visibility not in simple_counts[simple_cat]:
                    simple_counts[simple_cat][visibility] = 0
                simple_counts[simple_cat][visibility] += 1
                if split == 'train':
                    total_ann_simple_train[domain] += 1
                else:
                    total_ann_simple_val[domain] += 1
        
        # Store unique sample counts per domain/split
        num_samples = {domain: {'train': len(sample_tokens_per_domain[domain]['train']),
                                'val': len(sample_tokens_per_domain[domain]['val'])}
                       for domain in DOMAINS}
        
        summary = []  # No summary for full dataset
    else:
        # Subset mode
        domain_data = {}
        summary = []
        total_ann = {}
        total_ann_train = {}
        total_ann_val = {}
        total_ann_simple_train = {}
        total_ann_simple_val = {}
        num_samples = {}
        
        for domain in DOMAINS:
            domain_path = os.path.join(args.base_dir, domain)
            if not os.path.exists(domain_path):
                warnings.warn(f"Domain {domain} does not exist, skipping.")
                continue
            
            roots = find_nuscenes_roots(domain_path)
            if not roots:
                warnings.warn(f"No NuScenes data found in {domain}, skipping.")
                continue
            
            structure = "Standard" if len(roots) == 1 and roots[0] == domain_path else "Nested"
            
            total_annotations = 0
            total_annotations_train = 0
            total_annotations_val = 0
            total_annotations_simple_train = 0
            total_annotations_simple_val = 0
            domain_sample_tokens_train = set()
            domain_sample_tokens_val = set()
            category_visibility_counts_train = {}
            category_visibility_counts_val = {}
            category_visibility_counts_simple_train = {cat: {} for cat in TARGET_CLASSES}
            category_visibility_counts_simple_val = {cat: {} for cat in TARGET_CLASSES}
            
            for root in roots:
                # Determine version
                version = 'v1.0-trainval' if os.path.exists(os.path.join(root, 'v1.0-trainval')) else 'v1.0-mini'
                nusc = NuScenes(version=version, dataroot=root, verbose=False)
                
                for ann in tqdm(nusc.sample_annotation, desc=f"Processing {domain} ({os.path.basename(root)})"):
                    category = ann['category_name']
                    visibility = int(ann['visibility_token'])
                    
                    # Determine if train or val
                    sample = nusc.get('sample', ann['sample_token'])
                    scene_token = sample['scene_token']
                    scene = nusc.get('scene', scene_token)
                    scene_name = scene['name']
                    # NEW CODE:
                    if version == 'v1.0-mini':
                        # Mini split is handled differently, usually specific scene names
                        # But commonly mini_train is scenes 0-9ish. 
                        # The safest way for mini is checking the built-in mini splits 
                        # or keeping your < 200 logic ONLY for mini.
                        scene_num = int(scene['name'].split('-')[1])
                        is_train = scene_num < 200 
                    else:
                        # Check if the scene name exists in the official train list
                        is_train = scene['name'] in train_scenes
                    
                    if is_train:
                        counts = category_visibility_counts_train
                        total_annotations_train += 1
                        domain_sample_tokens_train.add(ann['sample_token'])
                    else:
                        counts = category_visibility_counts_val
                        total_annotations_val += 1
                        domain_sample_tokens_val.add(ann['sample_token'])
                    
                    if category not in counts:
                        counts[category] = {}
                    if visibility not in counts[category]:
                        counts[category][visibility] = 0
                    counts[category][visibility] += 1
                    total_annotations += 1
                    
                    # Simple categories
                    if category in CLASS_MAPPING:
                        simple_cat = CLASS_MAPPING[category]
                        if is_train:
                            simple_counts = category_visibility_counts_simple_train
                            total_annotations_simple_train += 1
                        else:
                            simple_counts = category_visibility_counts_simple_val
                            total_annotations_simple_val += 1
                        
                        if visibility not in simple_counts[simple_cat]:
                            simple_counts[simple_cat][visibility] = 0
                        simple_counts[simple_cat][visibility] += 1
            
            domain_data[domain] = {'train': category_visibility_counts_train, 'val': category_visibility_counts_val, 'simple_train': category_visibility_counts_simple_train, 'simple_val': category_visibility_counts_simple_val}
            summary.append({
                'Domain': domain,
                'Structure': structure,
                'Total Annotations': total_annotations
            })
            total_ann[domain] = total_annotations
            total_ann_train[domain] = total_annotations_train
            total_ann_val[domain] = total_annotations_val
            total_ann_simple_train[domain] = total_annotations_simple_train
            total_ann_simple_val[domain] = total_annotations_simple_val
            num_samples[domain] = {'train': len(domain_sample_tokens_train),
                                   'val': len(domain_sample_tokens_val)}
    
    # Collect all unique categories across domains and splits
    all_categories = set()
    for domain, splits in domain_data.items():
        for split in ['train', 'val']:
            all_categories.update(splits[split].keys())
    all_categories = sorted(all_categories)
    
    # For main plot, combine train and val
    combined_domain_data = {}
    for domain, splits in domain_data.items():
        combined = {}
        for split in ['train', 'val']:
            for cat, vis_dict in splits[split].items():
                if cat not in combined:
                    combined[cat] = {}
                for vis, count in vis_dict.items():
                    combined[cat][vis] = combined[cat].get(vis, 0) + count
        combined_domain_data[domain] = combined
    
    # Print summary
    print("\nSummary:")
    print(pd.DataFrame(summary))
    
    # Visualization
    num_domains = len(combined_domain_data)
    if num_domains == 0:
        print("No data to visualize.")
        return
    
    fig, axes = plt.subplots(num_domains, 1, figsize=(20, 15), sharex=True)
    if num_domains == 1:
        axes = [axes]
    
    colors = ['red', 'orange', 'yellow', 'green']  # For visibility levels
    
    for i, (domain, counts) in enumerate(combined_domain_data.items()):
        ax = axes[i]
        
        # Prepare data for stacked bar
        categories = all_categories
        visibility_levels = sorted(VISIBILITY_MAP.keys())
        
        bottoms = [0] * len(categories)
        for j, vis in enumerate(visibility_levels):
            values = [counts.get(cat, {}).get(vis, 0) for cat in categories]
            ax.bar(categories, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
            bottoms = [b + v for b, v in zip(bottoms, values)]
        
        ax.set_title(f'{domain}')
        ax.set_ylabel('Number of Instances')
        ax.tick_params(axis='x', rotation=90)
    
    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title='Visibility')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'class_visibility_counts.png'), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(args.out_dir, 'class_visibility_counts.png')}")
    
    # Per class plots
    if 'per_class' in dirs_to_plot:
        per_class_dir = os.path.join(args.out_dir, 'per_class')
        os.makedirs(per_class_dir, exist_ok=True)
        
        domains_list = list(combined_domain_data.keys())
        
        for cat in all_categories:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            visibility_levels = sorted(VISIBILITY_MAP.keys())
            
            bottoms = [0] * len(domains_list)
            for j, vis in enumerate(visibility_levels):
                values = [combined_domain_data[domain].get(cat, {}).get(vis, 0) for domain in domains_list]
                ax.bar(domains_list, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
                bottoms = [b + v for b, v in zip(bottoms, values)]
            
            ax.set_title(cat)
            ax.set_ylabel('Number of Instances')
            ax.set_xlabel('Domains')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Visibility')
            
            plt.tight_layout()
            safe_cat = cat.replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(per_class_dir, f'{safe_cat}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    print(f"Per-class plots saved to {per_class_dir}")
    
    # Normalized per class plots
    if 'normalized_per_class' in dirs_to_plot:
        normalized_per_class_dir = os.path.join(args.out_dir, 'normalized_per_class')
        os.makedirs(normalized_per_class_dir, exist_ok=True)
        
        for cat in all_categories:
        
            bottoms = [0] * len(domains_list)
            for j, vis in enumerate(visibility_levels):
                values = [(combined_domain_data[domain].get(cat, {}).get(vis, 0) / total_ann[domain] * 100) if total_ann[domain] > 0 else 0 for domain in domains_list]
                ax.bar(domains_list, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
                bottoms = [b + v for b, v in zip(bottoms, values)]
        
        ax.set_title(f'{cat} (Normalized)')
        ax.set_ylabel('Percentage of Instances (%)')
        ax.set_xlabel('Domains')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Visibility')
        
        plt.tight_layout()
        safe_cat = cat.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(normalized_per_class_dir, f'{safe_cat}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Normalized per-class plots saved to {normalized_per_class_dir}")
    
    # Per class train plots
    if 'per_class_train' in dirs_to_plot:
        per_class_train_dir = os.path.join(args.out_dir, 'per_class_train')
        os.makedirs(per_class_train_dir, exist_ok=True)
    
    for cat in all_categories:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        visibility_levels = sorted(VISIBILITY_MAP.keys())
        
        bottoms = [0] * len(domains_list)
        for j, vis in enumerate(visibility_levels):
            values = [domain_data[domain]['train'].get(cat, {}).get(vis, 0) for domain in domains_list]
            ax.bar(domains_list, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
            bottoms = [b + v for b, v in zip(bottoms, values)]
        
        ax.set_title(f'{cat} (Train)')
        ax.set_ylabel('Number of Instances')
        ax.set_xlabel('Domains')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Visibility')
        
        plt.tight_layout()
        safe_cat = cat.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(per_class_train_dir, f'{safe_cat}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Per-class train plots saved to {per_class_train_dir}")
    
    # Per class val plots
    per_class_val_dir = os.path.join(args.out_dir, 'per_class_val')
    os.makedirs(per_class_val_dir, exist_ok=True)
    
    for cat in all_categories:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        visibility_levels = sorted(VISIBILITY_MAP.keys())
        
        bottoms = [0] * len(domains_list)
        for j, vis in enumerate(visibility_levels):
            values = [domain_data[domain]['val'].get(cat, {}).get(vis, 0) for domain in domains_list]
            ax.bar(domains_list, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
            bottoms = [b + v for b, v in zip(bottoms, values)]
        
        ax.set_title(f'{cat} (Val)')
        ax.set_ylabel('Number of Instances')
        ax.set_xlabel('Domains')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Visibility')
        
        plt.tight_layout()
        safe_cat = cat.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(per_class_val_dir, f'{safe_cat}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Per-class val plots saved to {per_class_val_dir}")
    
    # Normalized per class train plots
    normalized_per_class_train_dir = os.path.join(args.out_dir, 'normalized_per_class_train')
    os.makedirs(normalized_per_class_train_dir, exist_ok=True)
    
    for cat in all_categories:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        visibility_levels = sorted(VISIBILITY_MAP.keys())
        
        bottoms = [0] * len(domains_list)
        for j, vis in enumerate(visibility_levels):
            values = [(domain_data[domain]['train'].get(cat, {}).get(vis, 0) / total_ann_train[domain] * 100) if total_ann_train[domain] > 0 else 0 for domain in domains_list]
            ax.bar(domains_list, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
            bottoms = [b + v for b, v in zip(bottoms, values)]
        
        ax.set_title(f'{cat} (Train, Normalized)')
        ax.set_ylabel('Percentage of Instances (%)')
        ax.set_xlabel('Domains')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Visibility')
        
        plt.tight_layout()
        safe_cat = cat.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(normalized_per_class_train_dir, f'{safe_cat}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Normalized per-class train plots saved to {normalized_per_class_train_dir}")
    
    # Normalized per class val plots
    normalized_per_class_val_dir = os.path.join(args.out_dir, 'normalized_per_class_val')
    os.makedirs(normalized_per_class_val_dir, exist_ok=True)
    
    for cat in all_categories:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        visibility_levels = sorted(VISIBILITY_MAP.keys())
        
        bottoms = [0] * len(domains_list)
        for j, vis in enumerate(visibility_levels):
            values = [(domain_data[domain]['val'].get(cat, {}).get(vis, 0) / total_ann_val[domain] * 100) if total_ann_val[domain] > 0 else 0 for domain in domains_list]
            ax.bar(domains_list, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
            bottoms = [b + v for b, v in zip(bottoms, values)]
        
        ax.set_title(f'{cat} (Val, Normalized)')
        ax.set_ylabel('Percentage of Instances (%)')
        ax.set_xlabel('Domains')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Visibility')
        
        plt.tight_layout()
        safe_cat = cat.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(normalized_per_class_val_dir, f'{safe_cat}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Normalized per-class val plots saved to {normalized_per_class_val_dir}")
    
    # Simple plots
    if 'simple_per_class' in dirs_to_plot:
        simple_per_class_dir = os.path.join(args.out_dir, 'simple_per_class')
        os.makedirs(simple_per_class_dir, exist_ok=True)
    
    simple_all_categories = TARGET_CLASSES
    
    for cat in simple_all_categories:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        visibility_levels = sorted(VISIBILITY_MAP.keys())
        
        bottoms = [0] * len(domains_list)
        for j, vis in enumerate(visibility_levels):
            values = [domain_data[domain]['simple_train'].get(cat, {}).get(vis, 0) + domain_data[domain]['simple_val'].get(cat, {}).get(vis, 0) for domain in domains_list]
            ax.bar(domains_list, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
            bottoms = [b + v for b, v in zip(bottoms, values)]
        
        ax.set_title(f'{cat} (Simple, Combined)')
        ax.set_ylabel('Number of Instances')
        ax.set_xlabel('Domains')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Visibility')
        
        plt.tight_layout()
        plt.savefig(os.path.join(simple_per_class_dir, f'{cat}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Simple per-class plots saved to {simple_per_class_dir}")
    
    # Simple normalized per class plots
    simple_normalized_per_class_dir = os.path.join(args.out_dir, 'simple_normalized_per_class')
    os.makedirs(simple_normalized_per_class_dir, exist_ok=True)
    
    for cat in simple_all_categories:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        visibility_levels = sorted(VISIBILITY_MAP.keys())
        
        bottoms = [0] * len(domains_list)
        for j, vis in enumerate(visibility_levels):
            values = [(domain_data[domain]['simple_train'].get(cat, {}).get(vis, 0) + domain_data[domain]['simple_val'].get(cat, {}).get(vis, 0)) / (total_ann_simple_train[domain] + total_ann_simple_val[domain]) * 100 if (total_ann_simple_train[domain] + total_ann_simple_val[domain]) > 0 else 0 for domain in domains_list]
            ax.bar(domains_list, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
            bottoms = [b + v for b, v in zip(bottoms, values)]
        
        ax.set_title(f'{cat} (Simple, Normalized)')
        ax.set_ylabel('Percentage of Instances (%)')
        ax.set_xlabel('Domains')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Visibility')
        
        plt.tight_layout()
        plt.savefig(os.path.join(simple_normalized_per_class_dir, f'{cat}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Simple normalized per-class plots saved to {simple_normalized_per_class_dir}")
    
    # Simple per class train plots
    simple_per_class_train_dir = os.path.join(args.out_dir, 'simple_per_class_train')
    os.makedirs(simple_per_class_train_dir, exist_ok=True)
    
    for cat in simple_all_categories:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        visibility_levels = sorted(VISIBILITY_MAP.keys())
        
        bottoms = [0] * len(domains_list)
        for j, vis in enumerate(visibility_levels):
            values = [domain_data[domain]['simple_train'].get(cat, {}).get(vis, 0) for domain in domains_list]
            ax.bar(domains_list, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
            bottoms = [b + v for b, v in zip(bottoms, values)]
        
        ax.set_title(f'{cat} (Simple, Train)')
        ax.set_ylabel('Number of Instances')
        ax.set_xlabel('Domains')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Visibility')
        
        plt.tight_layout()
        plt.savefig(os.path.join(simple_per_class_train_dir, f'{cat}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Simple per-class train plots saved to {simple_per_class_train_dir}")
    
    # Simple per class val plots
    simple_per_class_val_dir = os.path.join(args.out_dir, 'simple_per_class_val')
    os.makedirs(simple_per_class_val_dir, exist_ok=True)
    
    for cat in simple_all_categories:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        visibility_levels = sorted(VISIBILITY_MAP.keys())
        
        bottoms = [0] * len(domains_list)
        for j, vis in enumerate(visibility_levels):
            values = [domain_data[domain]['simple_val'].get(cat, {}).get(vis, 0) for domain in domains_list]
            ax.bar(domains_list, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
            bottoms = [b + v for b, v in zip(bottoms, values)]
        
        ax.set_title(f'{cat} (Simple, Val)')
        ax.set_ylabel('Number of Instances')
        ax.set_xlabel('Domains')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Visibility')
        
        plt.tight_layout()
        plt.savefig(os.path.join(simple_per_class_val_dir, f'{cat}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Simple per-class val plots saved to {simple_per_class_val_dir}")
    
    # Simple normalized per class train plots
    simple_normalized_per_class_train_dir = os.path.join(args.out_dir, 'simple_normalized_per_class_train')
    os.makedirs(simple_normalized_per_class_train_dir, exist_ok=True)
    
    for cat in simple_all_categories:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        visibility_levels = sorted(VISIBILITY_MAP.keys())
        
        bottoms = [0] * len(domains_list)
        for j, vis in enumerate(visibility_levels):
            values = [(domain_data[domain]['simple_train'].get(cat, {}).get(vis, 0) / total_ann_simple_train[domain] * 100) if total_ann_simple_train[domain] > 0 else 0 for domain in domains_list]
            ax.bar(domains_list, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
            bottoms = [b + v for b, v in zip(bottoms, values)]
        
        ax.set_title(f'{cat} (Simple, Train, Normalized)')
        ax.set_ylabel('Percentage of Instances (%)')
        ax.set_xlabel('Domains')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Visibility')
        
        plt.tight_layout()
        plt.savefig(os.path.join(simple_normalized_per_class_train_dir, f'{cat}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Simple normalized per-class train plots saved to {simple_normalized_per_class_train_dir}")
    
    # Simple normalized per class val plots
    simple_normalized_per_class_val_dir = os.path.join(args.out_dir, 'simple_normalized_per_class_val')
    os.makedirs(simple_normalized_per_class_val_dir, exist_ok=True)
    
    for cat in simple_all_categories:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        visibility_levels = sorted(VISIBILITY_MAP.keys())
        
        bottoms = [0] * len(domains_list)
        for j, vis in enumerate(visibility_levels):
            values = [(domain_data[domain]['simple_val'].get(cat, {}).get(vis, 0) / total_ann_simple_val[domain] * 100) if total_ann_simple_val[domain] > 0 else 0 for domain in domains_list]
            ax.bar(domains_list, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
            bottoms = [b + v for b, v in zip(bottoms, values)]
        
        ax.set_title(f'{cat} (Simple, Val, Normalized)')
        ax.set_ylabel('Percentage of Instances (%)')
        ax.set_xlabel('Domains')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Visibility')
        
        plt.tight_layout()
        plt.savefig(os.path.join(simple_normalized_per_class_val_dir, f'{cat}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Simple normalized per-class val plots saved to {simple_normalized_per_class_val_dir}")
    
    # Average Instances per Sample plots (simple_avg_per_class)
    if 'simple_avg_per_class' in dirs_to_plot:
        simple_avg_per_class_dir = os.path.join(args.out_dir, 'simple_avg_per_class')
        os.makedirs(simple_avg_per_class_dir, exist_ok=True)

        simple_all_categories = TARGET_CLASSES

        for cat in simple_all_categories:
            for split in ['train', 'val']:
                fig, ax = plt.subplots(figsize=(10, 6))

                visibility_levels = sorted(VISIBILITY_MAP.keys())

                bottoms = [0.0] * len(domains_list)
                for j, vis in enumerate(visibility_levels):
                    values = []
                    for domain in domains_list:
                        n_samples = num_samples[domain][split]
                        count = domain_data[domain][f'simple_{split}'].get(cat, {}).get(vis, 0)
                        avg = (count / n_samples) if n_samples > 0 else 0
                        values.append(avg)
                    ax.bar(domains_list, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
                    bottoms = [b + v for b, v in zip(bottoms, values)]

                ax.set_title(f'{cat} ({split.capitalize()}, Avg per Sample)')
                ax.set_ylabel('Avg Instances per Sample')
                ax.set_xlabel('Domains')
                ax.tick_params(axis='x', rotation=45)
                ax.legend(title='Visibility')

                plt.tight_layout()
                plt.savefig(os.path.join(simple_avg_per_class_dir, f'{cat}_{split}_avg.png'), dpi=300, bbox_inches='tight')
                plt.close(fig)

        print(f"Simple avg-per-class plots saved to {simple_avg_per_class_dir}")

    # Summary: Average Instances per Sample across all classes and domains
    fig, axes_avg = plt.subplots(len(domains_list), 1, figsize=(20, 20))
    if len(domains_list) == 1:
        axes_avg = [axes_avg]

    x_labels_avg = [f'{cat}_{split}' for cat in TARGET_CLASSES for split in ['train', 'val']]

    for i, domain in enumerate(domains_list):
        ax = axes_avg[i]

        visibility_levels = sorted(VISIBILITY_MAP.keys())
        bottoms = [0.0] * len(x_labels_avg)

        for j, vis in enumerate(visibility_levels):
            values = []
            for cat in TARGET_CLASSES:
                for split in ['train', 'val']:
                    n_samples = num_samples[domain][split]
                    count = domain_data[domain][f'simple_{split}'].get(cat, {}).get(vis, 0)
                    avg = (count / n_samples) if n_samples > 0 else 0
                    values.append(avg)

            ax.bar(x_labels_avg, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
            bottoms = [b + v for b, v in zip(bottoms, values)]

        ax.set_title(f'{domain} - Avg Instances per Sample (Class & Visibility)')
        ax.set_ylabel('Avg Instances per Sample')
        ax.set_xlabel('Class and Split')
        ax.tick_params(axis='x', rotation=90)
        ax.legend(title='Visibility')

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'summary_avg_per_sample.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Summary avg-per-sample plot saved to {os.path.join(args.out_dir, 'summary_avg_per_sample.png')}")

    # Summary: Average Instances per Sample with separate y-axis per class
    n_classes = len(TARGET_CLASSES)
    fig_sep, axes_sep = plt.subplots(len(domains_list), n_classes, figsize=(4 * n_classes, 5 * len(domains_list)),
                                     squeeze=False)

    for i, domain in enumerate(domains_list):
        visibility_levels = sorted(VISIBILITY_MAP.keys())

        for k, cat in enumerate(TARGET_CLASSES):
            ax = axes_sep[i][k]
            x_pos = [0, 1]
            x_tick_labels = ['train', 'val']

            bottoms = [0.0, 0.0]
            for j, vis in enumerate(visibility_levels):
                values = []
                for split in ['train', 'val']:
                    n_samples = num_samples[domain][split]
                    count = domain_data[domain][f'simple_{split}'].get(cat, {}).get(vis, 0)
                    avg = (count / n_samples) if n_samples > 0 else 0
                    values.append(avg)
                ax.bar(x_pos, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j], width=0.6)
                bottoms = [b + v for b, v in zip(bottoms, values)]

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_tick_labels)
            if i == 0:
                ax.set_title(cat, fontsize=10)
            if k == 0:
                ax.set_ylabel(f'{domain}\nAvg / Sample', fontsize=8)
            # Only add legend to the first subplot
            if i == 0 and k == n_classes - 1:
                ax.legend(title='Visibility', fontsize=7, title_fontsize=8)

    fig_sep.suptitle('Avg Instances per Sample — Separate Scale per Class', fontsize=14, y=1.01)
    fig_sep.tight_layout()
    fig_sep.savefig(os.path.join(args.out_dir, 'summary_avg_per_sample_seperate.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_sep)

    print(f"Separate summary avg-per-sample plot saved to {os.path.join(args.out_dir, 'summary_avg_per_sample_seperate.png')}")

    # Summary plot
    fig, axes = plt.subplots(len(domains_list), 1, figsize=(20, 20))
    
    x_labels = [f'{cat}_{split}' for cat in TARGET_CLASSES for split in ['train', 'val']]
    
    for i, domain in enumerate(domains_list):
        ax = axes[i]
        
        visibility_levels = sorted(VISIBILITY_MAP.keys())
        bottoms = [0] * len(x_labels)
        
        for j, vis in enumerate(visibility_levels):
            values = []
            for cat in TARGET_CLASSES:
                for split in ['train', 'val']:
                    count = domain_data[domain][f'simple_{split}'].get(cat, {}).get(vis, 0)
                    total = total_ann_simple_train[domain] if split == 'train' else total_ann_simple_val[domain]
                    perc = (count / total * 100) if total > 0 else 0
                    values.append(perc)
            
            ax.bar(x_labels, values, bottom=bottoms, label=VISIBILITY_MAP[vis], color=colors[j])
            bottoms = [b + v for b, v in zip(bottoms, values)]
        
        ax.set_title(f'{domain} - Normalized Class and Visibility Distribution')
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Class and Split')
        ax.tick_params(axis='x', rotation=90)
        ax.legend(title='Visibility')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'simple_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simple summary plot saved to {os.path.join(args.out_dir, 'simple_summary.png')}")

if __name__ == '__main__':
    main()
