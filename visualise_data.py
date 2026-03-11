import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import argparse
import warnings

# Suppress pandas/seaborn warnings for clean terminal output
warnings.filterwarnings("ignore", category=FutureWarning)

class FlyVisualizer:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        sns.set_theme(style="whitegrid", palette="muted")
        
        # 1. Load the raw data
        self.df = self._load_data()
        
        # 2. Pre-calculate all complex metrics once globally
        self._calculate_derived_metrics()
        
    def _load_data(self):
        print(f"Parsing JSON files in {self.input_dir}...")
        json_files = glob.glob(os.path.join(self.input_dir, "*_tracking.json"))
        
        all_frames = []
        for jf in json_files:
            video_name = os.path.basename(jf).replace("_tracking.json", "")
            with open(jf, 'r') as f:
                data = json.load(f)
                
            for chamber, track in data.items():
                if chamber == "Summary": continue
                
                c_num = int(chamber.split("_")[1])
                group = "Control" if c_num <= 3 else "Experimental"
                
                for frame_idx, state in track.items():
                    if frame_idx == "Summary": continue
                    
                    time_s = state.get('time_s', float(frame_idx) * 0.033)
                    
                    all_frames.append({
                        "Video": video_name,
                        "Chamber": chamber,
                        "Group": group,
                        "Replicate_ID": f"{video_name}_{chamber}",
                        "Time_s": time_s,
                        "x_mm": state['x_mm'],
                        "y_mm": state['y_mm'],
                        "r": state['dist_mm'],
                        "r2": state['dist_mm']**2,
                        "speed": state['speed_mm']
                    })
        
        df = pd.DataFrame(all_frames)
        
        # Dynamically split into First and Second Half per video
        max_times = df.groupby('Video')['Time_s'].transform('max')
        df['Time_Phase'] = np.where(df['Time_s'] <= max_times / 2, 'First_Half', 'Second_Half')
        
        print(f"Loaded {len(df)} data points from {len(df['Replicate_ID'].unique())} unique chambers.")
        return df

    def _calculate_derived_metrics(self):
        print("Pre-calculating all derived metrics (Freezing, Meandering, Center Entries)...")
        self.df = self.df.sort_values(['Replicate_ID', 'Time_s']).reset_index(drop=True)
        
        # --- ZONES & THIGMOTAXIS ---
        self.df['in_center'] = self.df['r'] <= 3.5
        self.df['in_edge'] = self.df['r'] > 3.5
        
        # Detect Center Entries (crossed from edge to center)
        shifted_center = self.df.groupby('Replicate_ID')['in_center'].shift(1).fillna(self.df['in_center'])
        self.df['center_entry_event'] = (self.df['in_center'] == True) & (shifted_center == False)

        # --- FREEZING & STOPS ---
        self.df['is_stopped'] = self.df['speed'] < 2.0
        shifted_stop = self.df.groupby('Replicate_ID')['is_stopped'].shift(1).fillna(self.df['is_stopped'])
        self.df['stop_event'] = (self.df['is_stopped'] == True) & (shifted_stop == False)
        
        # --- MEANDERING ---
        self.df['dx'] = self.df.groupby('Replicate_ID')['x_mm'].diff()
        self.df['dy'] = self.df.groupby('Replicate_ID')['y_mm'].diff()
        self.df['dist_step'] = np.sqrt(self.df['dx']**2 + self.df['dy']**2)
        
        self.df['theta'] = np.arctan2(self.df['dy'], self.df['dx'])
        theta_diff = self.df.groupby('Replicate_ID')['theta'].diff()
        self.df['d_theta_deg'] = np.degrees(np.abs(np.arctan2(np.sin(theta_diff), np.cos(theta_diff))))
        
        # Calculate meander only when moving
        self.df['meander'] = np.nan
        moving_mask = self.df['speed'] > 2.0
        self.df.loc[moving_mask, 'meander'] = self.df.loc[moving_mask, 'd_theta_deg'] / self.df.loc[moving_mask, 'dist_step']
        
        # Cap meandering outliers at 95th percentile to prevent micro-jitter explosions
        cap = self.df['meander'].quantile(0.95)
        self.df.loc[self.df['meander'] > cap, 'meander'] = np.nan

    def add_significance_asterisks(self, ax, data1, data2, x_pos1, x_pos2, y_max):
        data1 = data1.dropna()
        data2 = data2.dropna()
        if len(data1) < 2 or len(data2) < 2: return
        t_stat, p_val = ttest_ind(data1, data2, nan_policy='omit')
        if p_val < 0.05:
            h = y_max * 0.05
            ax.plot([x_pos1, x_pos1, x_pos2, x_pos2], [y_max, y_max+h, y_max+h, y_max], lw=1.5, color='k')
            stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            ax.text((x_pos1+x_pos2)*.5, y_max+h, stars, ha='center', va='bottom', color='k')

    # ==========================================
    # 1. BASE VISUALIZATIONS
    # ==========================================
    
    def plot_heatmaps(self):
        print("Generating Spatial Heatmaps...")
        for resolution, bins in [("Standard", 50), ("HighRes", 200)]:
            grid_limit = np.linspace(-15, 15, bins)
            for phase in ["Overall", "First_Half", "Second_Half"]:
                subset = self.df if phase == "Overall" else self.df[self.df['Time_Phase'] == phase]
                if subset.empty: continue
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f'Spatial Density ({resolution}) - {phase}')
                for i, group in enumerate(["Control", "Experimental"]):
                    group_data = subset[subset['Group'] == group]
                    axes[i].hist2d(group_data['x_mm'], group_data['y_mm'], bins=grid_limit, cmap='inferno')
                    axes[i].set_title(group)
                    axes[i].set_aspect('equal')
                    axes[i].set_xlabel("X (mm)")
                    axes[i].set_ylabel("Y (mm)")
                    
                plt.savefig(os.path.join(self.input_dir, f"Vis_Heatmap_{resolution}_{phase}.png"))
                plt.close()

    def plot_timeseries(self):
        print("Generating Time Series...")
        temp_df = self.df.copy()
        temp_df['Time_Bin'] = temp_df['Time_s'].round()
        agg_df = temp_df.groupby(['Time_Bin', 'Group'])[['r', 'r2']].mean().reset_index()
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for i, metric, limit, ylabel in [(0, 'r', 5, "Average r (mm)"), (1, 'r2', 25, "Average r^2 (mm^2)")]:
            sns.lineplot(data=agg_df, x='Time_Bin', y=metric, hue='Group', ax=axes[i], alpha=0.3, legend=False)
            for group in ["Control", "Experimental"]:
                group_data = agg_df[agg_df['Group'] == group].copy()
                group_data['Rolling'] = group_data[metric].rolling(window=30, min_periods=1).mean()
                color = sns.color_palette()[0] if group == "Control" else sns.color_palette()[1]
                axes[i].plot(group_data['Time_Bin'], group_data['Rolling'], label=f"{group} (30s Trend)", color=color, linewidth=2.5)
            
            axes[i].set_ylabel(ylabel)
            axes[i].set_ylim(0, limit)
            axes[i].legend()

        axes[0].set_title("Radial Distance Over Time")
        axes[1].set_xlabel("Time (seconds)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.input_dir, "Vis_TimeSeries.png"))
        plt.close()

    def plot_bar_graphs(self):
        print("Generating Overall Bar Graphs...")
        rep_phases = self.df.groupby(['Replicate_ID', 'Group', 'Time_Phase'])[['r', 'r2', 'speed']].mean().reset_index()
        rep_overall = self.df.groupby(['Replicate_ID', 'Group'])[['r', 'r2', 'speed']].mean().reset_index()
        rep_overall['Time_Phase'] = 'Overall'
        combined = pd.concat([rep_overall, rep_phases])
        
        for metric, ylabel in [('r', 'Mean r (mm)'), ('r2', 'Mean r^2 (mm^2)'), ('speed', 'Mean Speed (mm/s)')]:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=combined, x='Time_Phase', y=metric, hue='Group', order=['Overall', 'First_Half', 'Second_Half'], capsize=0.1, errorbar='se', ax=ax)
            for i, phase in enumerate(['Overall', 'First_Half', 'Second_Half']):
                ctrl = combined[(combined['Group'] == 'Control') & (combined['Time_Phase'] == phase)][metric]
                exp = combined[(combined['Group'] == 'Experimental') & (combined['Time_Phase'] == phase)][metric]
                y_max = max(ctrl.max(), exp.max()) * 1.1 if not ctrl.empty and not exp.empty else 1
                self.add_significance_asterisks(ax, ctrl, exp, i - 0.2, i + 0.2, y_max)
                    
            ax.set_ylabel(ylabel)
            ax.set_title(f"Overall {metric} Comparison")
            plt.savefig(os.path.join(self.input_dir, f"Vis_Bar_{metric}.png"))
            plt.close()

    def plot_density(self):
        print("Generating Density Plots...")
        phases = self.df.groupby(['Replicate_ID', 'Group', 'Time_Phase'])[['r', 'r2']].mean().reset_index()
        overall = self.df.groupby(['Replicate_ID', 'Group'])[['r', 'r2']].mean().reset_index()
        overall['Time_Phase'] = 'Overall'
        combined = pd.concat([overall, phases])
        
        for metric, limit in [('r', 5), ('r2', 25)]:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
            fig.suptitle(f'Density Plot: Average {metric}')
            for i, phase in enumerate(['Overall', 'First_Half', 'Second_Half']):
                subset = combined[combined['Time_Phase'] == phase]
                if not subset.empty:
                    sns.kdeplot(data=subset, x=metric, hue='Group', fill=True, ax=axes[i], common_norm=False)
                axes[i].set_title(phase)
                axes[i].set_xlim(0, limit)
            plt.tight_layout()
            plt.savefig(os.path.join(self.input_dir, f"Vis_Density_{metric}.png"))
            plt.close()

    def plot_thigmotaxis(self):
        print("Generating Thigmotaxis Plot...")
        phases = self.df.groupby(['Replicate_ID', 'Group', 'Time_Phase'])['in_edge'].mean().reset_index()
        overall = self.df.groupby(['Replicate_ID', 'Group'])['in_edge'].mean().reset_index()
        overall['Time_Phase'] = 'Overall'
        combined = pd.concat([overall, phases])
        combined['in_edge'] *= 100 
        
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = {"Control": "lightgray", "Experimental": "lightgray"}
        sns.boxplot(data=combined, x='Time_Phase', y='in_edge', hue='Group', order=['Overall', 'First_Half', 'Second_Half'], palette=palette, showfliers=False, ax=ax)
        sns.stripplot(data=combined, x='Time_Phase', y='in_edge', hue='Group', order=['Overall', 'First_Half', 'Second_Half'], dodge=True, alpha=0.7, ax=ax)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title="Group")
        ax.set_ylabel("% Time in Edge (r > 3.5 mm)")
        ax.set_title("Thigmotaxis Comparison")
        plt.savefig(os.path.join(self.input_dir, "Vis_Thigmotaxis.png"))
        plt.close()

    def plot_cumulative_distance(self):
        print("Generating Cumulative Distance...")
        temp_df = self.df.copy()
        temp_df['Dist_Step_Safe'] = temp_df['speed'] * 0.033
        temp_df['Cumulative_Dist'] = temp_df.groupby('Replicate_ID')['Dist_Step_Safe'].cumsum()
        temp_df['Time_Bin'] = temp_df['Time_s'].round()
        agg_dist = temp_df.groupby(['Time_Bin', 'Group', 'Replicate_ID'])['Cumulative_Dist'].max().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=agg_dist, x='Time_Bin', y='Cumulative_Dist', hue='Group', errorbar='se', ax=ax)
        ax.set_title("Cumulative Distance Traveled")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Total Distance (mm)")
        plt.savefig(os.path.join(self.input_dir, "Vis_Cumulative_Distance.png"))
        plt.close()

    def plot_speed_vs_r_scatter(self):
        print("Generating Scatterplots...")
        for phase in ["Overall", "First_Half", "Second_Half"]:
            subset = self.df if phase == "Overall" else self.df[self.df['Time_Phase'] == phase]
            if subset.empty: continue
            
            samples = []
            for grp, grp_data in subset.groupby('Group'):
                samples.append(grp_data.sample(n=min(len(grp_data), 25000)))
            if not samples: continue
            
            sample_df = pd.concat(samples, ignore_index=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=sample_df, x='r', y='speed', hue='Group', alpha=0.05, edgecolor=None, ax=ax)
            ax.set_xlim(0, 5)
            ax.set_title(f"Speed vs Radial Distance ({phase})")
            ax.set_xlabel("r (mm)")
            ax.set_ylabel("Speed (mm/s)")
            
            leg = ax.legend()
            for lh in leg.legend_handles: lh.set_alpha(1)
            plt.savefig(os.path.join(self.input_dir, f"Vis_Scatter_Speed_v_r_{phase}.png"))
            plt.close()

    # ==========================================
    # 2. MICRO-BEHAVIOR VISUALIZATIONS (OVERALL)
    # ==========================================

    def _plot_box_strip(self, df, y_col, ylabel, title, filename, show_overall=False):
        order = ['Overall', 'First_Half', 'Second_Half'] if show_overall else ['First_Half', 'Second_Half']
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = {"Control": "lightgray", "Experimental": "lightgray"}
        
        sns.boxplot(data=df, x='Time_Phase', y=y_col, hue='Group', order=order, palette=palette, showfliers=False, ax=ax)
        sns.stripplot(data=df, x='Time_Phase', y=y_col, hue='Group', order=order, dodge=True, alpha=0.7, ax=ax)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title="Group")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.savefig(os.path.join(self.input_dir, filename))
        plt.close()

    def plot_micro_behaviors(self):
        print("Generating Micro-Behavior Plots...")
        
        # Freezing
        freeze = self.df.groupby(['Replicate_ID', 'Group', 'Time_Phase'])['is_stopped'].mean().reset_index()
        freeze['is_stopped'] *= 100
        self._plot_box_strip(freeze, 'is_stopped', "% Time Freezing", "Freezing Behavior", "Vis_Freezing_Time.png")
        
        # Stop Count
        stops = self.df.groupby(['Replicate_ID', 'Group', 'Time_Phase'])['stop_event'].sum().reset_index()
        self._plot_box_strip(stops, 'stop_event', "Number of Stop Bouts", "Hesitation / Stop Frequency", "Vis_Stop_Frequency.png")
        
        # Meandering
        meander = self.df.dropna(subset=['meander']).groupby(['Replicate_ID', 'Group', 'Time_Phase'])['meander'].mean().reset_index()
        self._plot_box_strip(meander, 'meander', "Meandering (Degrees / mm)", "Path Tortuosity", "Vis_Meandering.png")

    # ==========================================
    # 3. CENTER-ZONE EXCLUSIVE VISUALIZATIONS
    # ==========================================

    def plot_center_metrics(self):
        print("Generating Center-Zone Exclusive Plots & Entry Counts...")
        center_df = self.df[self.df['in_center'] == True]
        
        # Center Entries (Calculated from full df, not center_df, to capture transitions)
        entries_phase = self.df.groupby(['Replicate_ID', 'Group', 'Time_Phase'])['center_entry_event'].sum().reset_index()
        entries_overall = self.df.groupby(['Replicate_ID', 'Group'])['center_entry_event'].sum().reset_index()
        entries_overall['Time_Phase'] = 'Overall'
        combined_entries = pd.concat([entries_overall, entries_phase])
        self._plot_box_strip(combined_entries, 'center_entry_event', "Total Center Entries", "Frequency of Entering the Center Zone", "Vis_Center_Entries.png", show_overall=True)

        if center_df.empty:
            print("No center-zone data found.")
            return

        # Center Speeds
        c_phases = center_df.groupby(['Replicate_ID', 'Group', 'Time_Phase'])[['r', 'speed']].mean().reset_index()
        c_overall = center_df.groupby(['Replicate_ID', 'Group'])[['r', 'speed']].mean().reset_index()
        c_overall['Time_Phase'] = 'Overall'
        c_combined = pd.concat([c_overall, c_phases])
        
        for metric, ylabel in [('r', 'Mean r in Center (mm)'), ('speed', 'Mean Speed in Center (mm/s)')]:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=c_combined, x='Time_Phase', y=metric, hue='Group', order=['Overall', 'First_Half', 'Second_Half'], capsize=0.1, errorbar='se', ax=ax)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{metric} Comparison (Center Zone)")
            plt.savefig(os.path.join(self.input_dir, f"Vis_Center_Bar_{metric}.png"))
            plt.close()

        # Center Freezing
        c_freeze = center_df.groupby(['Replicate_ID', 'Group', 'Time_Phase'])['is_stopped'].mean().reset_index()
        c_freeze['is_stopped'] *= 100
        self._plot_box_strip(c_freeze, 'is_stopped', "% Time Freezing in Center", "Center Zone Freezing", "Vis_Center_Freezing.png")
        
        # Center Stops
        c_stops = center_df.groupby(['Replicate_ID', 'Group', 'Time_Phase'])['stop_event'].sum().reset_index()
        self._plot_box_strip(c_stops, 'stop_event', "Number of Stops in Center", "Center Zone Hesitation", "Vis_Center_Stops.png")
        
        # Center Meandering
        c_meander = center_df.dropna(subset=['meander']).groupby(['Replicate_ID', 'Group', 'Time_Phase'])['meander'].mean().reset_index()
        self._plot_box_strip(c_meander, 'meander', "Center Meandering (Deg / mm)", "Center Path Tortuosity", "Vis_Center_Meandering.png")

    # ==========================================
    # 4. CSV EXPORT
    # ==========================================

    def export_summary_csv(self):
        print("Exporting master dataset to CSV...")
        
        # We process 'Overall' and phases for Replicates
        def build_metric(df, col, metric_name, agg_func):
            phases = df.groupby(['Replicate_ID', 'Group', 'Time_Phase'])[col].agg(agg_func).reset_index()
            overall = df.groupby(['Replicate_ID', 'Group'])[col].agg(agg_func).reset_index()
            overall['Time_Phase'] = 'Overall'
            combined = pd.concat([overall, phases]).rename(columns={col: metric_name})
            return combined

        # Overall Base Metrics
        m1 = build_metric(self.df, 'speed', 'Mean_Speed', 'mean')
        m2 = build_metric(self.df, 'r', 'Mean_r', 'mean')
        m3 = build_metric(self.df, 'in_edge', 'Pct_Thigmotaxis', lambda x: x.mean() * 100)
        m4 = build_metric(self.df, 'is_stopped', 'Pct_Freezing', lambda x: x.mean() * 100)
        m5 = build_metric(self.df, 'stop_event', 'Stop_Bout_Count', 'sum')
        m6 = build_metric(self.df.dropna(subset=['meander']), 'meander', 'Mean_Meandering', 'mean')
        m7 = build_metric(self.df, 'center_entry_event', 'Center_Entries', 'sum')

        # Center-Exclusive Metrics
        c_df = self.df[self.df['in_center'] == True]
        c1 = build_metric(c_df, 'speed', 'Center_Mean_Speed', 'mean') if not c_df.empty else pd.DataFrame()
        c2 = build_metric(c_df, 'is_stopped', 'Center_Pct_Freezing', lambda x: x.mean() * 100) if not c_df.empty else pd.DataFrame()
        c3 = build_metric(c_df, 'stop_event', 'Center_Stop_Bout_Count', 'sum') if not c_df.empty else pd.DataFrame()
        c4 = build_metric(c_df.dropna(subset=['meander']), 'meander', 'Center_Mean_Meandering', 'mean') if not c_df.empty else pd.DataFrame()

        # Merge them all
        summary_df = m1
        for df_to_merge in [m2, m3, m4, m5, m6, m7, c1, c2, c3, c4]:
            if not df_to_merge.empty:
                summary_df = pd.merge(summary_df, df_to_merge, on=['Replicate_ID', 'Group', 'Time_Phase'], how='outer')

        # Sort cleanly
        summary_df['Phase_Order'] = summary_df['Time_Phase'].map({'Overall': 1, 'First_Half': 2, 'Second_Half': 3})
        summary_df = summary_df.sort_values(['Group', 'Replicate_ID', 'Phase_Order']).drop(columns=['Phase_Order'])
        
        output_path = os.path.join(self.input_dir, "Summary_Metrics_Master_Export.csv")
        summary_df.to_csv(output_path, index=False)
        print(f"Summary CSV successfully saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Fly Data Visualizer")
    parser.add_argument("-i", "--input", required=True, help="Directory containing *_tracking.json files")
    args = parser.parse_args()
    
    vis = FlyVisualizer(args.input)
    vis.plot_heatmaps()
    vis.plot_timeseries()
    vis.plot_bar_graphs()
    vis.plot_density()
    vis.plot_thigmotaxis()
    vis.plot_cumulative_distance()
    vis.plot_speed_vs_r_scatter()
    
    # Consolidated micro-behaviors & Center metrics
    vis.plot_micro_behaviors()
    vis.plot_center_metrics()
    
    # Export
    vis.export_summary_csv()
    
    print("All visualizations and master CSV data saved to your input directory.")

if __name__ == "__main__":
    main()