#!/usr/bin/env python3
"""
Data Poisoning Toolkit - CLI Interface
Command line interface for running attacks and detectors on CSV datasets
"""

import argparse
import sys
import pandas as pd
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich import box

if sys.platform == "win32":
    try:
        import colorama
        colorama.just_fix_windows_console()
    except ImportError:
        import os
        os.system("")

# Demo runner fonksiyonlarını import et
from datapoisontoolkit.demo_runner import (
    run_interactive_single_experiment,
    FlexibleDatasetLoader,
    AutoDatasetConfig,
    create_flexible_dataset_config,
    get_config,
    save_results,
    plot_results
)

from datapoisontoolkit.attacks import base_attack
from datapoisontoolkit.detectors import model_detectors, stat_detectors
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

# Rich console
console = Console(force_terminal=True, color_system="auto")

# ASCII Art Banner
BANNER = """
[bold cyan]
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                       ██████╗  ██████╗  ████████╗                          ║
║                       ██╔══██╗ ██╔══██╗ ╚══██╔══╝                          ║
║                       ██║  ██║ ██████╔╝    ██║                             ║
║                       ██║  ██║ ██╔═══╝     ██║                             ║
║                       ██████╔╝ ██║         ██║                             ║
║                       ╚═════╝  ╚═╝         ╚═╝                             ║
║                                                                            ║
║       [bold white]                Data Poisoning Toolkit[/bold white] [dim]v1.0[/dim]                          ║
║           [italic]          ML Security Research Framework[/italic]                         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
[/bold cyan]
"""


def print_banner():
    """Print the fancy banner"""
    console.print(BANNER)


def main():
    parser = argparse.ArgumentParser(
        description='Data Poisoning Toolkit - ML Security Research Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Interactive mode (recommended for first use)
  python -m datapoisontoolkit.cli interactive

  # Quick attack on a dataset
  python -m datapoisontoolkit.cli attack --data data/50Hertz.csv --type dspo --budget 10 --output results/

  # Run specific attack with custom parameters
  python -m datapoisontoolkit.cli attack --data data/Plant_1_Generation.csv --type random --budget 5 --strength 0.3

  # List available attacks and defenses
  python -m datapoisontoolkit.cli list

  # Analyze a dataset without running attacks
  python -m datapoisontoolkit.cli analyze --data data/traffic_index.csv

Modes:
  interactive    Full interactive mode with dataset selection
  attack        Run specific attack on a dataset
  list          Show available attacks and defenses
  analyze       Analyze dataset structure
        '''
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser(
        'interactive',
        help='Interactive mode - guided setup'
    )
    
    # Attack mode
    attack_parser = subparsers.add_parser(
        'attack',
        help='Run attack on dataset'
    )
    attack_parser.add_argument(
        '--data',
        required=True,
        type=str,
        help='Path to CSV dataset'
    )
    attack_parser.add_argument(
        '--type',
        choices=['random', 'greedy_residue', 'dspo', 'all'],
        default='dspo',
        help='Attack type (default: dspo)'
    )
    attack_parser.add_argument(
        '--target',
        type=str,
        help='Target column name (auto-detected if not specified)'
    )
    attack_parser.add_argument(
        '--budget',
        type=float,
        default=10.0,
        help='Poison budget as percentage (default: 10.0)'
    )
    attack_parser.add_argument(
        '--strength',
        type=float,
        default=0.5,
        help='Perturbation strength (default: 0.5)'
    )
    attack_parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='DSPO optimization iterations (default: 100)'
    )
    attack_parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='DSPO learning rate (default: 0.01)'
    )
    attack_parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    attack_parser.add_argument(
        '--defenses',
        nargs='+',
        choices=['lof', 'pca', 'kde', 'forest', 'median', 'quantile', 'dbscan', 'all'],
        default=['all'],
        help='Defense mechanisms to test against (default: all)'
    )
    
    # List mode
    list_parser = subparsers.add_parser(
        'list',
        help='List available attacks and defenses'
    )
    
    # Analyze mode
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze dataset structure'
    )
    analyze_parser.add_argument(
        '--data',
        required=True,
        type=str,
        help='Path to CSV dataset'
    )
    
    args = parser.parse_args()
    
    # Mode yoksa yardım göster
    if not args.mode:
        print_banner()
        parser.print_help()
        sys.exit(0)
    
    # Banner göster
    print_banner()
    
    # Mode'lara göre işlem yap
    if args.mode == 'interactive':
        run_interactive_mode()
    elif args.mode == 'attack':
        run_attack_mode(args)
    elif args.mode == 'list':
        list_components()
    elif args.mode == 'analyze':
        analyze_dataset(args.data)


def run_interactive_mode():
    """Run full interactive mode"""
    console.print(Panel.fit(
        "[bold yellow]INTERACTIVE MODE[/bold yellow]\n"
        "Guided setup for data poisoning experiments",
        border_style="yellow"
    ))
    console.print()
    
    run_interactive_single_experiment()


def run_attack_mode(args):
    """Run attack with command line arguments"""
    console.print(Panel.fit(
        "[bold magenta]ATTACK MODE[/bold magenta]\n"
        "Running data poisoning attack on dataset",
        border_style="magenta"
    ))
    console.print()
    
    # Dataset yükle
    csv_path = Path(args.data)
    if not csv_path.exists():
        console.print(f"[bold red]✗ Error:[/bold red] File not found: {csv_path}")
        sys.exit(1)
    
    console.print(f"[cyan]📁 Loading dataset:[/cyan] {csv_path}")
    
    # Dataset analizi ve preprocessing
    loader = FlexibleDatasetLoader()
    config = loader.analyze_dataset(str(csv_path), args.target)
    df_clean = loader.load_and_preprocess_dataset(config)
    
    console.print(f"[green]✓[/green] Dataset loaded: [bold]{df_clean.shape}[/bold]")
    console.print(f"  [dim]Target:[/dim] {config.target_column} [dim]({config.task_type})[/dim]")
    
    # Attack yapılandırması
    selected_attacks = ['random', 'greedy_residue', 'dspo'] if args.type == 'all' else [args.type]
    
    attack_config = {
        'selected_attacks': selected_attacks,
        'budget_pct': args.budget / 100.0,
        'perturb_strength': args.strength,
        'num_iterations': args.iterations,
        'learning_rate': args.lr,
        'batch_size': min(5000, len(df_clean)),
        'lambda_impact': 1.0,
        'lambda_stealth': 0.3,
        'lambda_defense': 0.7
    }
    
    # Defense seçimi
    if 'all' in args.defenses:
        active_defenses = ['lof', 'pca', 'kde', 'forest', 'median', 'quantile', 'dbscan']
    else:
        active_defenses = args.defenses
    
    defenses = {name: model_detectors.get_defense(name) for name in active_defenses}
    
    # Config panel
    config_text = f"""[cyan]Attacks:[/cyan] {', '.join(selected_attacks)}
[cyan]Budget:[/cyan] {args.budget}% ({int(len(df_clean) * attack_config['budget_pct'])} points)
[cyan]Strength:[/cyan] {args.strength}
[cyan]Defenses:[/cyan] {', '.join(active_defenses)}"""
    
    console.print(Panel(config_text, title="[bold]Attack Configuration[/bold]", border_style="cyan"))
    console.print()
    
    # Base model eğitimi
    X_clean = StandardScaler().fit_transform(df_clean.drop(columns=[config.target_column]))
    y_clean = df_clean[config.target_column]
    
    if config.task_type == 'classification':
        base_model = RandomForestClassifier(n_estimators=100, random_state=0)
    else:
        base_model = RandomForestRegressor(n_estimators=100, random_state=0)
    
    console.print("[cyan]🔧 Training base model...[/cyan]")
    base_model.fit(X_clean, y_clean)
    console.print("[green]✓[/green] Base model trained")
    console.print()
    
    # Attack çalıştırma
    all_results = []
    
    for attack_name in selected_attacks:
        console.rule(f"[bold magenta]Running {attack_name.upper()} Attack[/bold magenta]")
        console.print()
        
        try:
            # Attack instance oluştur
            if attack_name == 'random':
                attack_cfg = base_attack.BaseAttackConfig(
                    budget_pct=attack_config['budget_pct'],
                    perturb_strength=attack_config['perturb_strength']
                )
                attack_instance = base_attack.RandomAttack(
                    attack_cfg, df_clean, config.target_column, config.task_type
                )
            elif attack_name == 'greedy_residue':
                from datapoisontoolkit.attacks.greedy_residue_mock import GreedyResidueAttack
                attack_cfg = base_attack.BaseAttackConfig(
                    budget_pct=attack_config['budget_pct'],
                    perturb_strength=attack_config['perturb_strength']
                )
                attack_instance = GreedyResidueAttack(
                    attack_cfg, df_clean, config.target_column, config.task_type
                )
            elif attack_name == 'dspo':
                attack_instance = base_attack.get_attack(
                    "dspo", None, df_clean, config.target_column,
                    config.task_type, defenses, interactive_config=attack_config
                )
            
            # Poisoned dataset oluştur
            df_poisoned = attack_instance.craft(base_model)
            poison_indices = attack_instance.poison_indices
            
            console.print("[green]✓[/green] Poisoned dataset created")
            console.print()
            
            # Impact ve stealth değerlendirmesi
            impact_metrics = stat_detectors.evaluate_impact(
                df_clean, df_poisoned, config.target_column, config.task_type
            )
            stealth_metrics = stat_detectors.evaluate_stealth(
                df_clean[config.target_column],
                df_poisoned[config.target_column]
            )
            
            # Defense testleri - Progress bar ile
            console.print("[cyan]Testing defenses...[/cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Defense tests", total=len(defenses))
                
                for defense_name, defense_instance in defenses.items():
                    progress.update(task, description=f"[cyan]Testing {defense_name:>10}")
                    
                    try:
                        anomaly_scores = defense_instance.detect(df_poisoned[config.target_column])
                        
                        if isinstance(anomaly_scores, pd.Series):
                            threshold = anomaly_scores.quantile(1 - attack_config['budget_pct'])
                            detected_indices = anomaly_scores[anomaly_scores >= threshold].index
                        else:
                            detected_indices = anomaly_scores[anomaly_scores == -1].index
                        
                        defense_perf = stat_detectors.evaluate_defense_performance(
                            df_clean.index, poison_indices, detected_indices
                        )
                        
                        result_entry = {
                            "dataset": csv_path.stem,
                            "task": config.task_type,
                            "attack": attack_name,
                            "defense": defense_name,
                            "poison_budget": attack_config['budget_pct'],
                            **impact_metrics,
                            **defense_perf,
                            **stealth_metrics
                        }
                        all_results.append(result_entry)
                        
                        # Success/failure renklendirmesi
                        precision = defense_perf['precision']
                        recall = defense_perf['recall']
                        
                        if precision < 0.5 and recall < 0.5:
                            color = "green"  # Attack başarılı (defense başarısız)
                        elif precision > 0.7 or recall > 0.7:
                            color = "red"  # Defense başarılı (attack başarısız)
                        else:
                            color = "yellow"  # Orta
                        
                    except Exception as e:
                        console.print(f"  [{color}]✗ Error: {e}[/{color}]")
                    
                    progress.advance(task)
            
            console.print()
                    
        except Exception as e:
            console.print(f"[bold red]✗ Error running {attack_name}:[/bold red] {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Sonuçları kaydet
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        # Config oluştur (save_results için)
        cfg = get_config()
        for attack_name in selected_attacks:
            if attack_name == 'dspo':
                cfg.attacks[attack_name] = base_attack.DSPOConfig(
                    budget_pct=attack_config['budget_pct'],
                    perturb_strength=attack_config['perturb_strength'],
                    num_iterations=attack_config['num_iterations'],
                    learning_rate=attack_config['learning_rate']
                )
            else:
                cfg.attacks[attack_name] = base_attack.BaseAttackConfig(
                    budget_pct=attack_config['budget_pct'],
                    perturb_strength=attack_config['perturb_strength']
                )
        
        results_dir = Path(args.output)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        file_prefix = save_results(df_results, results_dir, csv_path.stem, cfg)
        
        try:
            plot_results(df_results, results_dir, csv_path.stem, cfg, file_prefix)
        except Exception as e:
            console.print(f"[yellow]⚠ Could not create plots:[/yellow] {e}")
        
        console.print()
        console.print(Panel.fit(
            f"[bold green]✓ ATTACK COMPLETED SUCCESSFULLY[/bold green]\n"
            f"Results saved to: [cyan]{results_dir}[/cyan]",
            border_style="green"
        ))
    else:
        console.print("[bold red]✗ No results generated[/bold red]")


def list_components():
    """List available attacks and defenses"""
    console.print(Panel.fit(
        "[bold blue]AVAILABLE COMPONENTS[/bold blue]\n"
        "Attacks and defenses in the toolkit",
        border_style="blue"
    ))
    console.print()
    
    # Attacks table
    attacks_table = Table(title="[bold magenta]Available Attacks[/bold magenta]", box=box.ROUNDED)
    attacks_table.add_column("Name", style="cyan", no_wrap=True)
    attacks_table.add_column("Description", style="white")
    
    attacks = {
        'random': 'Random baseline attack - random perturbations',
        'greedy_residue': 'Greedy attack targeting high-residue points',
        'dspo': 'D-SPO attack - sophisticated optimization-based attack'
    }
    for name, desc in attacks.items():
        attacks_table.add_row(name, desc)
    
    console.print(attacks_table)
    console.print()
    
    # Defenses table
    defenses_table = Table(title="[bold green]Available Defenses[/bold green]", box=box.ROUNDED)
    defenses_table.add_column("Name", style="cyan", no_wrap=True)
    defenses_table.add_column("Description", style="white")
    
    defenses = {
        'lof': 'Local Outlier Factor',
        'pca': 'PCA-based anomaly detection',
        'kde': 'Kernel Density Estimation',
        'forest': 'Isolation Forest',
        'median': 'Median Filter',
        'quantile': 'Quantile-based detection',
        'dbscan': 'DBSCAN clustering'
    }
    for name, desc in defenses.items():
        defenses_table.add_row(name, desc)
    
    console.print(defenses_table)
    console.print()


def analyze_dataset(data_path: str):
    """Analyze dataset structure without running attacks"""
    console.print(Panel.fit(
        "[bold blue]DATASET ANALYSIS[/bold blue]\n"
        "Analyzing dataset structure and properties",
        border_style="blue"
    ))
    console.print()
    
    csv_path = Path(data_path)
    if not csv_path.exists():
        console.print(f"[bold red]✗ Error:[/bold red] File not found: {csv_path}")
        sys.exit(1)
    
    console.print(f"[cyan]📁 Analyzing:[/cyan] {csv_path}")
    console.print()
    
    # Dataset yükle
    df = pd.read_csv(csv_path)
    
    # Basic info
    info_text = f"""[cyan]Shape:[/cyan] {df.shape[0]} rows × {df.shape[1]} columns
[cyan]Memory:[/cyan] {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"""
    console.print(Panel(info_text, title="[bold]Basic Information[/bold]", border_style="cyan"))
    console.print()
    
    # Kolon tipleri
    loader = FlexibleDatasetLoader()
    datetime_cols = loader._detect_datetime_columns(df)
    categorical_cols = loader._detect_categorical_columns(df)
    numeric_cols = loader._detect_numeric_columns(df)
    
    columns_text = f"""[cyan]Datetime columns:[/cyan] {len(datetime_cols)}
{('  ' + ', '.join(datetime_cols)) if datetime_cols else '  None'}

[cyan]Categorical columns:[/cyan] {len(categorical_cols)}
{('  ' + ', '.join(categorical_cols[:5]) + (' ...' if len(categorical_cols) > 5 else '')) if categorical_cols else '  None'}

[cyan]Numeric columns:[/cyan] {len(numeric_cols)}
{('  ' + ', '.join(numeric_cols[:5]) + (' ...' if len(numeric_cols) > 5 else '')) if numeric_cols else '  None'}"""
    
    console.print(Panel(columns_text, title="[bold]Column Types[/bold]", border_style="cyan"))
    console.print()
    
    # Target önerisi
    suggested_target = loader._auto_detect_target(df, numeric_cols)
    task_type = loader._determine_task_type(df, suggested_target)
    
    target_text = f"[cyan]Suggested Target:[/cyan] [bold]{suggested_target}[/bold]\n"
    target_text += f"[cyan]Task Type:[/cyan] {task_type}\n\n"
    
    if task_type == 'classification':
        target_text += f"[cyan]Unique values:[/cyan] {df[suggested_target].nunique()}\n"
        target_text += f"[cyan]Class distribution:[/cyan]\n"
        class_dist = df[suggested_target].value_counts().head()
        for val, count in class_dist.items():
            target_text += f"  {val}: {count} ({count/len(df)*100:.1f}%)\n"
    else:
        target_text += f"[cyan]Range:[/cyan] [{df[suggested_target].min():.2f}, {df[suggested_target].max():.2f}]\n"
        target_text += f"[cyan]Mean:[/cyan] {df[suggested_target].mean():.2f}\n"
        target_text += f"[cyan]Std:[/cyan] {df[suggested_target].std():.2f}"
    
    console.print(Panel(target_text, title="[bold]Target Column Analysis[/bold]", border_style="cyan"))
    console.print()
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_text = ""
        for col, count in missing[missing > 0].items():
            missing_text += f"{col}: {count} ({count/len(df)*100:.1f}%)\n"
        console.print(Panel(missing_text.rstrip(), title="[bold yellow]⚠ Missing Values[/bold yellow]", border_style="yellow"))
        console.print()
    
    # Ready status
    ready = len(numeric_cols) > 0
    status_text = f"[bold {'green' if ready else 'red'}]{'✓ Ready for attack' if ready else '✗ Preprocessing needed'}[/bold {'green' if ready else 'red'}]"
    console.print(Panel.fit(status_text, border_style="green" if ready else "red"))
    console.print()


if __name__ == '__main__':
    main()