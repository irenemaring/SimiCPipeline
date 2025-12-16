#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SimiC Visualization Class
Provides visualization methods for SimiC pipeline results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from pathlib import Path
from typing import Optional, List, Union, Dict
import warnings

# Safe import for the pipeline base. Provide a minimal fallback when missing.
try:
    from simicpipeline.core.main import SimiCPipeline as _BasePipeline  # type: ignore
except (ImportError, ModuleNotFoundError):
    class _BasePipeline(object):
        """
        Minimal stub to allow importing and testing SimiCVisualization without full runtime deps.
        Only provides attributes/methods required during visualization object construction.
        """
        def __init__(self, *args, **kwargs):
            workdir = kwargs.get("workdir", ".")
            self.output_path = Path(workdir)
            self.run_name = kwargs.get("run_name", "run")

        # Stubs that raise if visualization methods needing pipeline data are used without SimiCPipeline
        def load_results(self, *args, **kwargs):
            raise ImportError("SimiCPipeline is not available. Install full runtime deps to use this method.")

        def calculate_dissimilarity(self, *args, **kwargs):
            raise ImportError("SimiCPipeline is not available. Install full runtime deps to use this method.")

        def subset_label_specific_auc(self, *args, **kwargs):
            raise ImportError("SimiCPipeline is not available. Install full runtime deps to use this method.")

warnings.filterwarnings('ignore')

class SimiCVisualization(_BasePipeline):
    """
    Visualization class for SimiC results.
    Inherits from SimiCPipeline to access all pipeline functionality plus visualization methods.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize visualization pipeline."""
        super().__init__(*args, **kwargs)
        
        # Create figures directory
        self.figures_path = self.output_path / "figures"/ self.run_name
        self.figures_path.mkdir(parents=True, exist_ok=True)

        # Set default plot style
        # sns.set_style("whitegrid")
        self.default_figsize = (12, 6)
        
        # Label names mapping (numeric label -> custom name)
        self.label_names = {}
    
    def set_label_names(self, label_names: Dict[Union[int, str], str]):
        """
        Set custom names for phenotype labels.
        
        Args:
            label_names: Dictionary mapping numeric labels to custom names
                        e.g., {0: 'Control', 1: 'Treatment1', 2: 'Treatment2'}
        
        Example:
            viz.set_label_names({0: 'Control', 1: 'PD-L1', 2: 'DAC', 3: 'Combination'})
        """
        self.label_names = {int(k): str(v) for k, v in label_names.items()}
        print(f"Label names set: {self.label_names}")
    
    def _get_label_name(self, label: Union[int, str]) -> str:
        """
        Get the display name for a label.
        
        Args:
            label: Numeric label
            
        Returns:
            Custom name if set, otherwise returns 'Label {label}'
        """
        label_int = int(label)
        if label_int in self.label_names:
            return self.label_names[label_int]
        return f'Label {label}'
        
    def plot_r2_distribution(self, labels: Optional[List[Union[int, str]]] = None, 
                            threshold: float = 0.7,
                            save: bool = True,
                            filename: Optional[str] = None):
        """
        Plot R² distribution histograms for target genes.
        
        Args:
            labels: List of phenotype labels to plot (default: all)
            threshold: R² threshold line to display
            save: Whether to save the figure
            filename: Custom filename for saved figure
        """
        print("\n" + "="*70)
        print("PLOTTING R² DISTRIBUTIONS")
        print("="*70 + "\n")
        
        # Load results
        results = self.load_results('Ws_filtered')
        adj_r2 = results['adjusted_r_squared']
        
        if labels is None:
            labels = list(adj_r2.keys())
        
        n_labels = len(labels)
        fig, axes = plt.subplots(1, n_labels, figsize=(6*n_labels, 5))
        if n_labels == 1:
            axes = [axes]
        
        for idx, label in enumerate(labels):
            r2_values = adj_r2[label]
            selected = np.sum(r2_values > threshold)
            mean_r2 = np.mean(r2_values[r2_values > threshold])
            
            label_name = self._get_label_name(label)
            
            axes[idx].hist(r2_values, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
            axes[idx].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
            axes[idx].set_xlabel('Adjusted R²', fontsize=12)
            axes[idx].set_ylabel('Frequency', fontsize=12)
            axes[idx].set_title(f'{label_name}\nTargets selected: {selected}, Mean R²: {mean_r2:.3f}', 
                               fontsize=12)
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fname = filename or f"{self.run_name}_R2_distribution.pdf"
            plt.savefig(self.figures_path / fname, bbox_inches='tight', dpi=300)
            print(f"✓ Saved to {self.figures_path / fname}")
        
        return fig

    def plot_tf_weights(self, tf_names: Union[str, List[str]],
                       labels: Optional[List[Union[int, str]]] = None,
                       top_n_targets: int = 50,
                       save: bool = True,
                       filename: Optional[str] = None):
        """
        Plot weight barplots for specific transcription factors.
        
        Args:
            tf_names: TF name(s) to plot (single string or list)
            labels: Phenotype labels to include (default: all)
            top_n_targets: Number of top targets to display
            save: Whether to save the figure
            filename: Custom filename for saved figure
        """
        print("\n" + "="*70)
        print(f"PLOTTING TF WEIGHT BARPLOTS")
        print("="*70 + "\n")
        
        if isinstance(tf_names, str):
            tf_names = [tf_names]
        
        # Load results
        try:
            results = self.load_results('Ws_filtered')
        except FileNotFoundError:
            print("Error: Filtered weights not found!")
            return None
        
        weight_dic = results['weight_dic']
        tf_ids = results['TF_ids']
        target_ids = results['query_targets']
        
        if labels is None:
            labels = list(weight_dic.keys())
        
        # Get unselected targets per label
        adj_r2 = results['adjusted_r_squared']
        unselected_targets = {}
        for label in labels:
            unselected_targets[label] = [target_ids[i] for i, r2 in enumerate(adj_r2[label]) if r2 < 0.7]
        
        # Filter valid TF names
        valid_tf_names = [tf for tf in tf_names if tf in tf_ids]
        if not valid_tf_names:
            print(f"Error: None of the specified TFs found in results: {tf_names}")
            return None
        
        n_tfs = len(valid_tf_names)
        fig, axes = plt.subplots(n_tfs, 1, figsize=(16, 5*n_tfs))
        if n_tfs == 1:
            axes = [axes]
        
        for tf_idx, tf_name in enumerate(valid_tf_names):
            if tf_name not in tf_ids:
                print(f"Warning: TF '{tf_name}' not found, skipping...")
                continue
            
            print(f"Processing {tf_name}...")
            
            # Get TF index
            tf_index = tf_ids.index(tf_name)
            
            # Collect weights for all labels
            plot_data = []
            for label in labels:
                try:
                    weights = weight_dic[label][tf_index, :]
                    for i, target in enumerate(target_ids):
                        if target not in unselected_targets[label] and weights[i] != 0:
                            plot_data.append({
                                'target': target,
                                'weight': weights[i],
                                'label': str(label),
                                'label_name': self._get_label_name(label),
                                'abs_weight': abs(weights[i])
                            })
                except Exception as e:
                    print(f"  Error processing label {label}: {e}")
                    continue
            
            df = pd.DataFrame(plot_data)
            
            if df.empty:
                print(f"  No non-zero weights found for {tf_name}")
                ax = axes[tf_idx]
                ax.text(0.5, 0.5, f'No non-zero weights\nfor {tf_name}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'TF: {tf_name}', fontsize=14, fontweight='bold')
                continue
            
            # Get top targets by absolute weight
            top_targets = df.groupby('target')['abs_weight'].max().nlargest(top_n_targets).index
            df_filtered = df[df['target'].isin(top_targets)]
            
            # Order targets by max absolute weight
            target_order = df_filtered.groupby('target')['abs_weight'].max().sort_values(ascending=False).index
            
            # Plot
            ax = axes[tf_idx]
            x_pos = np.arange(len(target_order))
            bar_width = 0.8 / len(labels)
            
            colors = ["steelblue", "orange", "green", "purple", "brown",]
            for label_idx, label in enumerate(labels):
                label_data = df_filtered[df_filtered['label'] == str(label)]
                weights_ordered = [label_data[label_data['target'] == t]['weight'].values[0] 
                                  if t in label_data['target'].values else 0 
                                  for t in target_order]
                
                label_name = self._get_label_name(label)
                
                ax.bar(x_pos + label_idx * bar_width, weights_ordered, 
                      bar_width, label=label_name, 
                      color=colors[label_idx], edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Target Genes', fontsize=10)
            ax.set_ylabel('Weight', fontsize=10)
            ax.set_title(f'TF: {tf_name}', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos + bar_width * (len(labels)-1) / 2)
            ax.set_xticklabels(target_order, rotation=45, ha='right', fontsize=7)
            ax.legend(loc='best')
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        
        if save:
            fname = filename or f"{self.run_name}_TF_weights.pdf"
            try:
                plt.savefig(self.figures_path / fname, bbox_inches='tight', dpi=300)
                print(f"✓ Saved to {self.figures_path / fname}")
            except Exception as e:
                print(f"Error saving figure: {e}")
        
        return fig

    def plot_target_weights(self, target_names: Union[str, List[str]],
                           labels: Optional[List[Union[int, str]]] = None,
                           save: bool = True,
                           filename: Optional[str] = None):
        """
        Plot weight barplots for specific target genes.
        
        Args:
            target_names: Target gene name(s) to plot
            labels: Phenotype labels to include (default: all)
            save: Whether to save the figure
            filename: Custom filename for saved figure
        """
        print("\n" + "="*70)
        print(f"PLOTTING TARGET WEIGHT BARPLOTS")
        print("="*70 + "\n")
        
        if isinstance(target_names, str):
            target_names = [target_names]
        
        # Load results
        try:
            results = self.load_results('Ws_filtered')
        except FileNotFoundError:
            print("Error: Filtered weights not found!")
            return None
        
        weight_dic = results['weight_dic']
        tf_ids = results['TF_ids']
        target_ids = results['query_targets']
        
        if labels is None:
            labels = list(weight_dic.keys())
        
        # Get unselected targets per label
        adj_r2 = results['adjusted_r_squared']
        unselected_targets = {}
        for label in labels:
            unselected_targets[label] = [target_ids[i] for i, r2 in enumerate(adj_r2[label]) if r2 < 0.7]
        
        # Filter valid target names
        valid_target_names = [t for t in target_names if t in target_ids]
        if not valid_target_names:
            print(f"Error: None of the specified targets found in results: {target_names}")
            return None
        
        n_targets = len(valid_target_names)
        fig, axes = plt.subplots(n_targets, 1, figsize=(16, 5*n_targets))
        if n_targets == 1:
            axes = [axes]
        
        for tgt_idx, target_name in enumerate(valid_target_names):
            if target_name not in target_ids:
                print(f"Warning: Target '{target_name}' not found, skipping...")
                continue
            
            print(f"Processing {target_name}...")
            
            # Get target index
            target_index = target_ids.index(target_name)
            
            # Collect weights for all labels
            plot_data = []
            for label in labels:
                if target_name in unselected_targets[label]:
                    continue
                
                try:
                    weights = weight_dic[label][:, target_index]
                    for i, tf in enumerate(tf_ids):
                        if weights[i] != 0:
                            plot_data.append({
                                'tf': tf,
                                'weight': weights[i],
                                'label': str(label),
                                'label_name': self._get_label_name(label),
                                'abs_weight': abs(weights[i])
                            })
                except Exception as e:
                    print(f"  Error processing label {label}: {e}")
                    continue
            
            df = pd.DataFrame(plot_data)
            
            if df.empty:
                print(f"  No non-zero weights found for {target_name}")
                ax = axes[tgt_idx]
                ax.text(0.5, 0.5, f'No non-zero weights\nfor {target_name}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Target: {target_name}', fontsize=14, fontweight='bold')
                continue
            
            # Order TFs by max absolute weight
            tf_order = df.groupby('tf')['abs_weight'].max().sort_values(ascending=False).index
            
            # Plot
            ax = axes[tgt_idx]
            x_pos = np.arange(len(tf_order))
            bar_width = 0.8 / len(labels)
            
            colors = ["steelblue", "orange", "green", "purple", "brown",]
            
            for label_idx, label in enumerate(labels):
                label_data = df[df['label'] == str(label)]
                weights_ordered = [label_data[label_data['tf'] == t]['weight'].values[0] 
                                  if t in label_data['tf'].values else 0 
                                  for t in tf_order]
                
                label_name = self._get_label_name(label)
                
                ax.bar(x_pos + label_idx * bar_width, weights_ordered, 
                      bar_width, label=label_name, 
                      color=colors[label_idx], edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Transcription Factors', fontsize=10)
            ax.set_ylabel('Weight', fontsize=10)
            ax.set_title(f'Target: {target_name}', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos + bar_width * (len(labels)-1) / 2)
            ax.set_xticklabels(tf_order, rotation=45, ha='right', fontsize=8)
            ax.legend(loc='best')
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        
        if save:
            fname = filename or f"{self.run_name}_target_weights.pdf"
            try:
                plt.savefig(self.figures_path / fname, bbox_inches='tight', dpi=300)
                print(f"✓ Saved to {self.figures_path / fname}")
            except Exception as e:
                print(f"Error saving figure: {e}")
        
        return fig

    def plot_auc_distributions(self, tf_names: Union[str, List[str]],
                              labels: Optional[List[Union[int, str]]] = None,
                              fill: bool = True,
                              alpha: float = 0.5,
                              bw_adjust: Union[str, float] = "scott",
                              save: bool = True,
                              filename: Optional[str] = None):
        """
        Plot AUC density distributions for specific TFs across phenotypes.
        
        Args:
            tf_names: TF name(s) to plot
            labels: Phenotype labels to compare
            fill: Whether to fill the density curves (default: True)
            alpha: Transparency level for filled curves (0-1, default: 0.6)
            bw_adjust:  Bandwidth adjustment for density smoothness (default: 0.5)
                      Can be 'scott', 'silverman', or a float value
                      Lower values (e.g., 0.2-0.5) = less smooth, more detail
                      Higher values (e.g., 1.0-2.0) = more smooth
            save: Whether to save the figure
            filename: Custom filename for saved figure
        """
        print("\n" + "="*70)
        print(f"PLOTTING AUC DISTRIBUTIONS")
        print("="*70 + "\n")
        
        if isinstance(tf_names, str):
            tf_names = [tf_names]
        
        # Load AUC results
        try:
            auc_data = self.load_results('auc_filtered')
        except FileNotFoundError:
            print("Error: AUC filtered results not found!")
            return None
        
        if labels is None:
            labels = list(auc_data.keys())
        
        # Calculate dissimilarity scores
        try:
            dissim_scores = self.calculate_dissimilarity(select_labels=labels, verbose=False)
        except Exception as e:
            print(f"Warning: Could not calculate dissimilarity scores: {e}")
            dissim_scores = pd.DataFrame()
        
        # Filter out TFs that have no data
        valid_tf_names = []
        for tf_name in tf_names:
            has_data = False
            for label in labels:
                try:
                    auc_subset = self.subset_label_specific_auc('auc_filtered', label)
                    if tf_name in auc_subset.columns:
                        values = auc_subset[tf_name].dropna()
                        if len(values) > 1:  # Need at least 2 points for density
                            has_data = True
                            break
                except Exception:
                    continue
            
            if has_data:
                valid_tf_names.append(tf_name)
            else:
                print(f"Warning: Skipping {tf_name} - insufficient data for plotting")
        
        if not valid_tf_names:
            print("Error: No valid TFs to plot!")
            return None
        
        n_tfs = len(valid_tf_names)
        n_cols = 2
        n_rows = (n_tfs + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
        axes = axes.flatten() if n_tfs > 1 else [axes]
        
        # colors = sns.color_palette("husl", len(labels))
        colors = ["steelblue", "orange", "green", "purple", "brown",]
        
        for tf_idx, tf_name in enumerate(valid_tf_names):
            print(f"Processing {tf_name}...")
            
            ax = axes[tf_idx]
            plotted_any = False
            
            for label_idx, label in enumerate(labels):
                try:
                    auc_subset = self.subset_label_specific_auc('auc_filtered', label)
                    
                    if tf_name not in auc_subset.columns:
                        print(f"  Warning: {tf_name} not found in label {label}")
                        continue
                    
                    values = auc_subset[tf_name].dropna()
                    
                    # Check if we have enough valid data points
                    if len(values) < 2:
                        print(f"  Warning: Insufficient data for {tf_name} in label {label} (n={len(values)})")
                        continue
                    
                    # Check if all values are the same (would cause density plot to fail)
                    if values.std() == 0:
                        print(f"  Warning: No variance in {tf_name} for label {label}, plotting as vertical line")
                        label_name = self._get_label_name(label)
                        ax.axvline(values.iloc[0], color=colors[label_idx], 
                                  linestyle='--', linewidth=2, label=label_name, alpha=alpha)
                        plotted_any = True
                        continue
                    
                    label_name = self._get_label_name(label)
                    
                    # Plot density with optional fill and bandwidth adjustment
                    try:
                        if fill:
                            values.plot.density(ax=ax, label=label_name, 
                                               color=colors[label_idx], alpha=alpha, linewidth=2,
                                               bw_method=bw_adjust)
                            # Fill under the curve
                            line = ax.get_lines()[-1]
                            x_data = line.get_xdata()
                            y_data = line.get_ydata()
                            ax.fill_between(x_data, y_data, alpha=alpha, color=colors[label_idx])
                        else:
                            values.plot.density(ax=ax, label=label_name, 
                                               color=colors[label_idx], alpha=1.0, linewidth=2,
                                               bw_method=bw_adjust)
                        plotted_any = True
                    except Exception as plot_error:
                        print(f"  Warning: Could not plot density for {tf_name}, label {label}: {plot_error}")
                        # Try histogram as fallback
                        try:
                            ax.hist(values, bins=20, density=True, alpha=alpha, 
                                   color=colors[label_idx], label=label_name, 
                                   edgecolor='black')
                            plotted_any = True
                        except Exception:
                            print(f"  Error: Could not plot histogram either")
                            continue
                    
                    # Add rug plot (optional, commented out as it can clutter)
                    # ax.scatter(values, np.zeros_like(values) - 0.05 * (label_idx + 1), 
                    #           alpha=0.3, s=1, color=colors[label_idx])
                
                except Exception as e:
                    print(f"  Error processing label {label}: {e}")
                    continue
            
            if not plotted_any:
                ax.text(0.5, 0.5, f'No data available\nfor {tf_name}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_xlim(0, 1)
            else:
                # Add dissimilarity score to title
                if tf_name in dissim_scores.index:
                    dissim_score = dissim_scores.loc[tf_name, 'MinMax_score']
                    ax.set_title(f'{tf_name}\nDissimilarity: {dissim_score:.4f}', 
                               fontsize=12, fontweight='bold')
                else:
                    ax.set_title(f'{tf_name}', fontsize=12, fontweight='bold')
                
                ax.set_xlabel('AUC Score', fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                ax.legend(loc='best')
                ax.grid(alpha=0.3)
                ax.set_xlim(0, 1)
        
        # Hide extra subplots
        for idx in range(n_tfs, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save:
            fname = filename or f"{self.run_name}_AUC_distributions.pdf"
            try:
                plt.savefig(self.figures_path / fname, bbox_inches='tight', dpi=300)
                print(f"✓ Saved to {self.figures_path / fname}")
            except Exception as e:
                print(f"Error saving figure: {e}")
        
        return fig

    def plot_dissimilarity_heatmap(self, labels: Optional[List[Union[int, str]]] = None,
                                   top_n_tfs: Optional[int] = None,
                                   save: bool = True,
                                   filename: Optional[str] = None):
        """
        Plot heatmap of regulatory dissimilarity scores.
        
        Args:
            labels: Phenotype labels to compare
            top_n_tfs: Number of top TFs to display (by dissimilarity)
            save: Whether to save the figure
            filename: Custom filename for saved figure
        """
        print("\n" + "="*70)
        print(f"PLOTTING DISSIMILARITY HEATMAP")
        print("="*70 + "\n")
        
        # Calculate dissimilarity scores
        dissim_df = self.calculate_dissimilarity(select_labels=labels)
        
        if top_n_tfs:
            dissim_df = dissim_df.head(top_n_tfs)
        
        # Create heatmap using matplotlib
        fig, ax = plt.subplots(figsize=(8, max(6, len(dissim_df) * 0.3)))
        
        # Create the heatmap
        im = ax.imshow(dissim_df.values, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(dissim_df.columns)))
        ax.set_yticks(np.arange(len(dissim_df.index)))
        ax.set_xticklabels(dissim_df.columns, fontsize=10)
        ax.set_yticklabels(dissim_df.index, fontsize=8)
        
        # Rotate the x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Dissimilarity Score', rotation=270, labelpad=20, fontsize=10)
        
        # Annotate cells with values
        for i in range(len(dissim_df.index)):
            for j in range(len(dissim_df.columns)):
                text = ax.text(j, i, f'{dissim_df.values[i, j]:.4f}',
                             ha="center", va="center", color="white", fontsize=6)
        
        # Add title and labels
        ax.set_title('Regulatory Dissimilarity Scores', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Transcription Factors', fontsize=12)
        
        # Add gridlines
        ax.set_xticks(np.arange(len(dissim_df.columns)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(dissim_df.index)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
        
        plt.tight_layout()
        
        if save:
            fname = filename or f"{self.run_name}_dissimilarity_heatmap.pdf"
            plt.savefig(self.figures_path / fname, bbox_inches='tight', dpi=300)
            print(f"✓ Saved to {self.figures_path / fname}")
        
        return fig

    def plot_umap_with_activity(self, umap_df: pd.DataFrame,
                                tf_names: Union[str, List[str]],
                                labels: Optional[List[Union[int, str]]] = None,
                                group_cols: Optional[List[str]] = None,
                                save: bool = True,
                                filename: Optional[str] = None):
        """
        Plot UMAP colored by TF activity scores.
        
        Args:
            umap_df: DataFrame with UMAP coordinates and cell identifiers
                    Must have columns: 'Cell', 'umap_1', 'umap_2'
            tf_names: TF name(s) to plot activity scores for
            labels: Phenotype labels to include
            group_cols: Additional grouping columns to facet by (e.g., ['condition', 'treatment'])
            save: Whether to save the figure
            filename: Custom filename for saved figure
        """
        print("\n" + "="*70)
        print(f"PLOTTING UMAP WITH ACTIVITY SCORES")
        print("="*70 + "\n")
        
        if isinstance(tf_names, str):
            tf_names = [tf_names]
        
        # Load AUC data
        auc_data = self.load_results('auc_filtered')
        
        if labels is None:
            labels = list(auc_data.keys())
        
        # Check required columns
        required_cols = ['Cell', 'umap_1', 'umap_2']
        if not all(col in umap_df.columns for col in required_cols):
            raise ValueError(f"umap_df must contain columns: {required_cols}")
        
        n_tfs = len(tf_names)
        fig, axes = plt.subplots(n_tfs, 1, figsize=(14, 5*n_tfs))
        if n_tfs == 1:
            axes = [axes]
        
        for tf_idx, tf_name in enumerate(tf_names):
            print(f"Processing {tf_name}...")
            
            # Collect AUC scores for this TF
            activity_data = []
            for label in labels:
                auc_subset = self.subset_label_specific_auc('auc_filtered', label)
                
                if tf_name not in auc_subset.columns:
                    print(f"  Warning: {tf_name} not found in label {label}")
                    continue
                
                scores = auc_subset[[tf_name]].copy()
                scores['Cell'] = scores.index
                scores.columns = ['activity_score', 'Cell']
                scores['label'] = label
                activity_data.append(scores)
            
            if not activity_data:
                print(f"  No data found for {tf_name}")
                continue
            
            activity_df = pd.concat(activity_data, ignore_index=True)
            
            # Merge with UMAP coordinates
            plot_df = umap_df.merge(activity_df, on='Cell', how='inner')
            
            # Create scatter plot
            ax = axes[tf_idx]
            
            if group_cols and all(col in plot_df.columns for col in group_cols):
                # Create facet-like subplots
                groups = plot_df.groupby(group_cols)
                n_groups = len(groups)
                
                # This is simplified - for true faceting, would need more complex layout
                scatter = ax.scatter(plot_df['umap_1'], plot_df['umap_2'], 
                                   c=plot_df['activity_score'], 
                                   cmap='inferno', s=10, alpha=0.6)
                ax.set_title(f'{tf_name} Activity Score', fontsize=12, fontweight='bold')
            else:
                scatter = ax.scatter(plot_df['umap_1'], plot_df['umap_2'], 
                                   c=plot_df['activity_score'], 
                                   cmap='inferno', s=10, alpha=0.6)
                ax.set_title(f'{tf_name} Activity Score', fontsize=12, fontweight='bold')
            
            ax.set_xlabel('UMAP 1', fontsize=10)
            ax.set_ylabel('UMAP 2', fontsize=10)
            plt.colorbar(scatter, ax=ax, label='Activity Score')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fname = filename or f"{self.run_name}_UMAP_activity.pdf"
            plt.savefig(self.figures_path / fname, bbox_inches='tight', dpi=300)
            print(f"✓ Saved to {self.figures_path / fname}")
        
        return fig
