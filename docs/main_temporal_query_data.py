import pandas as pd
import sys
from pathlib import Path
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Query temporal data from MLNet tensor.")
    parser.add_argument("--query_name", type=str, required=True, help="Nome della query della quale si vuole creare l'immagine della distribuzione nel tempo.")
    parser.add_argument("--do_single_alert", action="store_true", help="Whether to plot single alert temporal distribution.")
    parser.add_argument("--not_do_table", action="store_true", help="Whether to skip the table in the multiple alerts plot.")
    return parser.parse_args()

def main_multiple_alerts(args):
    query_name = args.query_name

    # Load df from data folder of query results
    print(Path(str(Path(__file__).resolve().parent / 'data' / f"{query_name}.csv")))
    if Path(str(Path(__file__).resolve().parent / 'data' / f"{query_name}.csv")).exists():
        data = pd.read_csv(f"{str(Path(__file__).resolve().parent)}/data/{query_name}.csv", sep=";")
        data["time_rawdate"] = pd.to_datetime(data["time_rawdate"], unit = "ms")
    else:
        raise FileNotFoundError(f"File data/{query_name}.csv non trovato.")
    
    


    print(data.head())
    # Group by both 'name' and 'time_rawdate' and count occurrences
    data_grouped = data.groupby(['name', 'time_rawdate','risk']).size().reset_index(name='count')
    data_grouped_per_pie = data.groupby(['name']).size().reset_index(name='count')

    # Get top 5 frequent names for pie chart
    top_5_names = data_grouped_per_pie.nlargest(5, 'count')
    other_count = data_grouped_per_pie[~data_grouped_per_pie['name'].isin(top_5_names['name'])]['count'].sum()
    
    # Create pie chart data with top 5 + other
    pie_data = top_5_names.copy()
    if other_count > 0:
        other_row = pd.DataFrame({'name': ['Other'], 'count': [other_count]})
        pie_data = pd.concat([pie_data, other_row], ignore_index=True)

    # Create pie chart with tab10 colors
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    colors = plt.cm.tab10.colors[:len(pie_data)]
    plt.pie(pie_data['count'], labels=pie_data['name'], autopct='%1.1f%%', startangle=140, colors=colors)
    plt.show()
    fig.savefig(f"{str(Path(__file__).resolve().parent)}/media/{query_name}_top5_names_piechart.png", bbox_inches='tight', dpi=300)

    print(data_grouped.head())
    # filter by keep the top 5 frequent names
    top_names = data["name"].value_counts().nlargest(5).index
    top_frequent_data = data_grouped[data_grouped['name'].isin(top_names)]
    # filter by keep the top 5 risky ones
    # Get the highest risk value for each name, then select top 5 names
    name_max_risk = data_grouped.groupby('name')['risk'].max().sort_values(ascending=False)
    risky_names = name_max_risk.head(5).index.tolist()  # Get names with the top 5 highest risk values
    risky_data = data_grouped[data_grouped['name'].isin(risky_names)]
    
    # Group risky data by date and sum counts
    risky_data_by_date = risky_data.groupby(['name', 'time_rawdate']).agg({
        'count': 'sum',
        'risk': 'max'  # Take max risk for each name-date combination
    }).reset_index()
    
    print(risky_data_by_date.head())
    print(risky_data_by_date["name"].value_counts())    
    
    # Create a figure with custom layout that includes both left subplot and facet grid
    fig = plt.figure(figsize=(18, len(top_names) * 2.5))

    # Create a grid: left subplot + right side for individual time series plots
    if not args.not_do_table:
        gs = fig.add_gridspec(len(top_names), 2, width_ratios=[2, 3], hspace=0.4, wspace=0.3)
        
        # Add the left subplot that spans full height
        ax_left = fig.add_subplot(gs[:, 0])  # Spans all rows in the left column
        
        # Create summary table data for the left subplot
        table_data = []
        for name in top_names:
            name_data = top_frequent_data[top_frequent_data['name'] == name]
            total_count = name_data['count'].sum()
            num_dates = name_data['time_rawdate'].nunique()
            avg_count = name_data['count'].mean()
            max_count = name_data['count'].max()
            table_data.append([name, total_count, num_dates, f"{avg_count:.1f}", max_count])
        
        # Create table headers
        headers = ['Name', 'Total Count', 'Num Date', 'Avg Count\n per data', 'Max Count\n in una data']
        
        # Create the table
        table = ax_left.table(cellText=table_data,
                            colLabels=headers,
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.5, 0.2, 0.2, 0.2, 0.2])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, len(top_names) * 0.8)  # Width slightly wider, height to match right plots
        
        # Style header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#E6E6FA')
            table[(0, i)].set_text_props(weight='bold')
        
        # Remove axis elements since we only want the table
        ax_left.axis('off')
    
    else:
        gs = fig.add_gridspec(len(top_names), 1, hspace=0.4)
    
    # Create individual subplots for each name on the right side
    for i, name in enumerate(top_names):
        if not args.not_do_table:
            ax_right = fig.add_subplot(gs[i, 1])
        else:
            ax_right = fig.add_subplot(gs[i, 0])
        
        # Filter data for this specific name
        name_data = top_frequent_data[top_frequent_data['name'] == name]
        
        # Create a complete date range with finer resolution
        min_date = name_data['time_rawdate'].min()
        max_date = name_data['time_rawdate'].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')  # Daily frequency
        
        # Create a complete dataframe with all dates, filling missing with 0
        complete_data = pd.DataFrame({'time_rawdate': date_range})
        complete_data = complete_data.merge(name_data[['time_rawdate', 'count']], 
                                          on='time_rawdate', how='left')
        complete_data['count'] = complete_data['count'].fillna(0)
        
        # Plot the line for this name with complete temporal grid
        sns.lineplot(data=complete_data, x="time_rawdate", y="count", ax=ax_right, marker='o', markersize=3)
        
        # Remove spines
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['bottom'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
        ax_right.spines['right'].set_visible(True)
        ax_right.spines['right'].set_linewidth(0.8)
        
        # Move y-axis ticks and label to the right side
        ax_right.yaxis.set_label_position("right")
        ax_right.yaxis.tick_right()
        
        # Format y-axis ticks as integers
        from matplotlib.ticker import MaxNLocator, FuncFormatter
        ax_right.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_right.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
        
        # Set title and labels
        ax_right.set_title(name, fontsize=14)
        ax_right.set_xlabel("Time [y%-m%]" if i == len(top_names)-1 else "", fontsize=14)
        ylabel_text = "Count" if i == len(top_names)//2 else ""
        ax_right.set_ylabel(ylabel_text, fontsize=12)
    
    # Save the unified figure
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    if args.not_do_table:
        fig.savefig(f"{str(Path(__file__).resolve().parent)}/media/{query_name}_temporal_distribution_by_name_unified_notable.png", bbox_inches='tight', dpi=300)
    else:
        fig.savefig(f"{str(Path(__file__).resolve().parent)}/media/{query_name}_temporal_distribution_by_name_unified.png", bbox_inches='tight', dpi=300)
    
    plt.show()

     # Create a figure with custom layout that includes both left subplot and facet grid
    fig = plt.figure(figsize=(18, len(risky_names) * 2.5))
    
    # Create a grid: left subplot + right side for individual time series plots
    if not args.not_do_table:
        gs = fig.add_gridspec(len(risky_names), 2, width_ratios=[2.5, 3], hspace=0.4, wspace=0.3)
        
        # Add the left subplot that spans full height
        ax_left = fig.add_subplot(gs[:, 0])  # Spans all rows in the left column
        
        # Create summary table data for the left subplot
        table_data = []
        for name in risky_names:
            risk = risky_data_by_date[risky_data_by_date['name'] == name]['risk'].max()
            name_data = risky_data_by_date[risky_data_by_date['name'] == name]
            total_count = name_data['count'].sum()
            num_dates = name_data['time_rawdate'].nunique()
            avg_count = name_data['count'].mean()
            max_count = name_data['count'].max()
            table_data.append([f"{name} ({risk})", total_count, num_dates, f"{avg_count:.1f}", max_count])

        # Create table headers
        headers = ['Name (risk)', 'Total Count', 'Num Date', 'Avg Count\n per data', 'Max Count\n in una data']
        
        # Create the table
        table = ax_left.table(cellText=table_data,
                            colLabels=headers,
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.5, 0.2, 0.2, 0.2, 0.2])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, len(top_names) * 0.8)  # Width slightly wider, height to match right plots
        
        # Style header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#E6E6FA')
            table[(0, i)].set_text_props(weight='bold')
        
        # Remove axis elements since we only want the table
        ax_left.axis('off')
    else:
        gs = fig.add_gridspec(len(risky_names), 1, hspace=0.4)

    # Create individual subplots for each name on the right side
    for i, name in enumerate(risky_names):
        if not args.not_do_table:
            ax_right = fig.add_subplot(gs[i, 1])
        else:
            ax_right = fig.add_subplot(gs[i, 0])
        
        # Filter data for this specific name
        name_data = risky_data_by_date[risky_data_by_date['name'] == name]
        
        # Create a complete date range with finer resolution
        min_date = name_data['time_rawdate'].min()
        max_date = name_data['time_rawdate'].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')  # Daily frequency
        
        # Create a complete dataframe with all dates, filling missing with 0
        complete_data = pd.DataFrame({'time_rawdate': date_range})
        complete_data = complete_data.merge(name_data[['time_rawdate', 'count']], 
                                          on='time_rawdate', how='left')
        complete_data['count'] = complete_data['count'].fillna(0)
        
        # Plot the line for this name with complete temporal grid
        sns.lineplot(data=complete_data, x="time_rawdate", y="count", ax=ax_right, marker='o', markersize=3)
        
        # Remove spines
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['bottom'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
        ax_right.spines['right'].set_visible(True)
        ax_right.spines['right'].set_linewidth(0.8)
        
        # Move y-axis ticks and label to the right side
        ax_right.yaxis.set_label_position("right")
        ax_right.yaxis.tick_right()
        
        # Format y-axis ticks as integers
        from matplotlib.ticker import MaxNLocator, FuncFormatter
        ax_right.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_right.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
        
        # Set title and labels
        ax_right.set_title(name, fontsize=14)
        ax_right.set_xlabel("Time [y%-m%]" if i == len(risky_names)-1 else "", fontsize=14)
        ylabel_text = "Count" if i == len(risky_names)//2 else ""
        ax_right.set_ylabel(ylabel_text, fontsize=12)

    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    if args.not_do_table:
        fig.savefig(f"{str(Path(__file__).resolve().parent)}/media/{query_name}_temporal_distribution_by_risky_name_notable.png", bbox_inches='tight', dpi=300)
    else:
        fig.savefig(f"{str(Path(__file__).resolve().parent)}/media/{query_name}_temporal_distribution_by_risky_name.png", bbox_inches='tight', dpi=300)
    plt.show()


def main(args):
    query_name = args.query_name

    # Load df from data folder of query results
    print(Path(str(Path(__file__).resolve().parent / 'data' / f"{query_name}.csv")))
    if Path(str(Path(__file__).resolve().parent / 'data' / f"{query_name}.csv")).exists():
        data = pd.read_csv(f"{str(Path(__file__).resolve().parent)}/data/{query_name}.csv", sep=";")
        data["time_rawdate"] = pd.to_datetime(data["time_rawdate"], unit = "ms")
    else:
        raise FileNotFoundError(f"File data/{query_name}.csv non trovato.")
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.lineplot(data=data, x="time_rawdate", y="count", ax=ax)
    
    # Remove axis lines (spines) but keep ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xlabel("Time [y%-m%]", fontsize=14)
    
    fig.savefig(f"{str(Path(__file__).resolve().parent)}/media/{query_name}_temporal_distribution.png", bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    args = parse_args()
    if args.do_single_alert:
        main(args)
    else:
        main_multiple_alerts(args)