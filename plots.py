import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta

def reports_by_type():
    reports = pd.read_csv('my_data/combined_reports.csv')
    reports.dropna(subset=['operatingCenter'], inplace=True)
    # The new reports has some very granular new types, but not enough to try to plot
    shared_events = {'Hazard Identification': 'red',
             'Material Release': 'orange',
             'Near Miss': 'blue',
             'Property Damage': 'purple',
             'Security': 'yellow',
             'Verification': 'teal'}
    type_dummies = pd.DataFrame([reports.operatingCenter]).T
    for event in shared_events.keys():
        type_dummies[event] = reports.eventType.apply(lambda s: event in s).astype(int)
    dates = pd.to_datetime(reports.serverCreatedDate)
    months = dates.apply(lambda d: datetime(d.year, d.month, 1))
    type_dummies['month'] = months
    type_counts = type_dummies.groupby(['operatingCenter', 'month']).sum()
    min_month = type_dummies.month.min()
    max_month = type_dummies.month.max()
    active_sites = ['Wamsutter', 'East Texas', 'Farmington', 'Anadarko', 'Durango',
       'Arkoma']
    f, axes = plt.subplots(len(active_sites), 1, figsize=(12, 16))
    for index, group in enumerate(active_sites):
        ax = axes[index]
        tc = type_counts.loc[group]
        N = tc.shape[0]
        bottom = np.zeros(N)
        for t in shared_events:
            ax.bar(tc.index, tc[t], bottom=bottom, label=t, width=20, color=shared_events[t])
            bottom += tc[t]
        ax.set_title(group)
        ax.set_xlim(min_month, max_month)
        ax.set_ylim(0, 1300)
        if index <= 4:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], title='Line', loc='upper left')
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], title='Line', loc='upper right')
    plt.tight_layout()
    plt.savefig('plots/types_over_time.png')

if __name__ == '__main__':
    print('Im a plot-making machine.')
