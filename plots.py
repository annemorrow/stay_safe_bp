import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from datetime import datetime, timedelta
from scorer import get_pipeline
from nltk.corpus import stopwords

def reports_by_type():
    '''
    Make a stacked bar graph for each operating center showing how many reports
    came in of each time during each month.
    '''
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

def time_plots():
    '''
    Save a set of histograms related to the time columns of reports.
    '''
    reports = pd.read_csv('my_data/combined_reports.csv')
    reports['serverCreatedDate'] = pd.to_datetime(reports.serverCreatedDate)
    reports['serverModifiedDate'] = pd.to_datetime(reports.serverModifiedDate)
    reports['adapterProcessedDate'] = pd.to_datetime(reports.adapterProcessedDate)
    days_until_modified = \
        (reports.serverModifiedDate - reports.serverCreatedDate).apply(lambda d: d.days)
    days_until_adapted = \
        (reports.adapterProcessedDate - reports.serverCreatedDate).apply(lambda d: d.days)
    days_until_modified.hist()
    plt.title('Days From Created To Modified')
    plt.xlabel('Days')
    plt.ylabel('Number of Reports')
    plt.savefig('plots/created_to_modified_hist.png')
    plt.close()
    days_until_modified_two_weeks = days_until_modified[days_until_modified <= 14]
    days_until_modified_two_weeks.hist()
    plt.title('Days From Created To Modified--Two Weeks')
    plt.xlabel('Days')
    plt.ylabel('Number of Reports')
    plt.savefig('plots/created_to_modified_hist_first_two_weeks.png')
    plt.close()
    days_until_adapted.hist()
    plt.title('Days From Created Until Put In Database')
    plt.xlabel('Days')
    plt.ylabel('Number of Reports')
    plt.savefig('plots/days_until_database.png')
    plt.close()
    reports.adapterProcessedDate.hist()
    plt.title('Adapter Processed Date')
    plt.xlabel('Days')
    plt.ylabel('Number of Reports')
    plt.savefig('plots/adapter_processed.png')
    plt.close()

def scatter_and_legend(point_size=1, x=3, y=1):
    '''
    Plot the incident descriptions in the x-y plane using coordinates from pipe_SVD
    where Stop the Job is in red, Further Action Necessary is in green,
    Action Completed Onsite is in orange, and No Action Necessary is in blue.
    '''
    pipe_SVD = get_pipeline()
    reports = pd.read_csv('my_data/combined_reports.csv').set_index('seq')
    reports.dropna(subset=['immediateActionsTaken', 'incidentDescription'], inplace=True)
    coordinates = pipe_SVD.transform(reports.incidentDescription)[:, [x, y]]
    x_coords, y_coords = coordinates[:,0], coordinates[:,1]
    actions = ['No Action Necessary', 'Action Completed Onsite',
               'Further Action Necessary', 'Stop the Job']
    color_map = {
                'Stop the Job': 'red',
                'Further Action Necessary': 'green',
                'Action Completed Onsite': 'orange',
                'No Action Necessary': 'blue'
                }
    plt.figure(figsize=(10, 10))
    for action in actions:
        plt.scatter(x_coords[reports.immediateActionsTaken == action],
                    y_coords[reports.immediateActionsTaken == action],
                    s=point_size,
                    c=color_map[action],
                    label=action)
    plt.axis('equal')
    plt.axis('off')
    #plt.legend(loc=(.6, .25), markerscale=10)

def words_around_edges():
    '''
    Put words from the svd components around the edges to get a sense of (if not
    an actual interpretation of) how descriptions are clustered.
    '''
    pipe_SVD = get_pipeline()
    x = pipe_SVD.named_steps['decomp'].components_[3,:]
    y = pipe_SVD.named_steps['decomp'].components_[1,:]
    vocab = pipe_SVD.named_steps['tfidf'].vocabulary_
    inverse_vocab = {vocab[key]:key for key in vocab}
    x_positive = [inverse_vocab[index] for index in x.argsort()[::-1][:30]]
    x_negative = [inverse_vocab[index] for index in x.argsort()[:30]]
    y_positive = [inverse_vocab[index] for index in y.argsort()[::-1][:30]]
    y_negative = [inverse_vocab[index] for index in y.argsort()[:30]]
    stop_words = set(stopwords.words('english'))
    x_positive = [word for word in x_positive if word not in stop_words]
    x_negative = [word for word in x_negative if word not in stop_words]
    y_positive = [word for word in y_positive if word not in stop_words]
    y_negative = [word for word in y_negative if word not in stop_words]
    for i, word in enumerate(x_positive):
        plt.text(.55, .2-i/40.0, word)
    for i, word in enumerate(x_negative):
        plt.text(-.32, .25 - i/40.0, word)
    for i, word in enumerate(y_positive):
        plt.text(-.25+i/30.0, .39 - (i%3)*.02, word)
    for i, word in enumerate(y_negative):
        plt.text(-.25+i/25.0, -.37 - (i%3)*.02, word)

def words_in_corners():
    '''
    Near the three main clusters/streamers, put a short interpretation
    of what sort of descriptions can be found there.
    '''
    plt.text(-.32, .4, 'The Wells Themselves', size='large')
    plt.text(.5, .1, 'Landscaping \n and Roads', size='large')
    plt.text(-.1, -.37, 'Conversations and Forms', size='large')

def plot_test_data(tag, x=3, y=1):
    graded = pd.read_csv('my_data/graded.csv')
    graded = graded[graded.immediateActionsTaken == tag]
    pipe_SVD = get_pipeline()
    coordinates = pipe_SVD.transform(graded.incidentDescription.astype(str))[:, [x, y]]
    x, y = coordinates[:,0], coordinates[:,1]
    tag_colors = {
                  'No Action Necessary':'blue',
                  'Action Completed Onsite': 'orange'
                 }
    plt.scatter(x[graded.grade == 0], y[graded.grade == 0],
                s=30, edgecolor='black', linewidth=3, c=tag_colors[tag],
                label='Not Important--{}'.format(tag))
    plt.scatter(x[graded.grade == 1], y[graded.grade == 1],
                s=30, edgecolor='red', linewidth=3, c=tag_colors[tag],
                label='Important--{}'.format(tag))
    l = plt.legend(loc=(.6, .5), markerscale=1.5)

def plot_data_and_test(tag, x=3, y=1):
    scatter_and_legend(.1, x, y)
    plot_test_data(tag, x, y)
    #plt.show()

def plot_3d():
    pipe_SVD = get_pipeline()
    reports = pd.read_csv('my_data/combined_reports.csv').set_index('seq')
    reports.dropna(subset=['immediateActionsTaken', 'incidentDescription'], inplace=True)
    coordinates = pipe_SVD.transform(reports.incidentDescription)[:, [1, 2, 3]]
    colors = list(reports.immediateActionsTaken.map({
                'Stop the Job': 'red',
                'Further Action Necessary': 'green',
                'Action Completed Onsite': 'orange',
                'No Action Necessary': 'blue'
                }))
    x, y, z = coordinates[:,0], coordinates[:,1], coordinates[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=colors, s=.01)
    graded = pd.read_csv('my_data/graded.csv')
    graded_tag = graded[graded.immediateActionsTaken == 'Action Completed Onsite']
    coordinates = pipe_SVD.transform(graded_tag.incidentDescription.astype(str))[:, [1, 2, 3]]
    x, y, z = coordinates[:,0], coordinates[:,1], coordinates[:,2]
    grade_colors = list(graded_tag.grade.map({0: 'black', 1: 'red'}))
    ax.scatter(x, y, z, c=grade_colors, s=3)
    plt.show()


if __name__ == '__main__':
    print('I am a plot-making machine.')
    plot_data_and_test('Action Completed Onsite', 2, 3)
