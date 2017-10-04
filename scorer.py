import numpy as np
import pandas as pd
import os.path
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pickle

def get_pipeline():
    '''
    Either load or create the pipeline that converts the incident descriptions
    into cartesian coordinates in 4 dimensional TruncatedSVD space (dimensions 3
    and 1 are the important ones for no action necessary and further action necessary,
    2 and 3 are the important ones for action completed onsite)
    '''
    if not os.path.isfile('SVD_pipe.pkl'):
        reports = pd.read_csv('my_data/combined_reports.csv')
        reports.dropna(subset=['immediateActionsTaken', 'incidentDescription'], inplace=True)
        comments = reports.incidentDescription.values
        pipe = Pipeline([
                            ('tfidf', TfidfVectorizer()),
                            ('decomp', TruncatedSVD(n_components=4))
                        ])
        pipe.fit(comments)
        with open('SVD_pipe.pkl', 'wb') as f:
            pickle.dump(pipe, f)
    else:
        with open('SVD_pipe.pkl', 'rb') as f:
            pipe = pickle.load(f)
    return pipe

'''
ACTION COMPLETED ONSITE

The dimensions in the SVD that have been most useful for no action necessary and
futher action necessary does not seperate action completed onsite into useful and
not useful.  However, in another pair of dimensions from the same SVD, actions completed
onsite seperates into two strands from the center, and in one of those strands,
the reports are very unlikely to be useful.  For reports tagged actions completed onsite,
rank the reports based on how close they are to the strand where some of the reports
are useful
'''
def get_coefficients_of_line():
    graded = pd.read_csv('my_data/graded.csv', index_col=0)
    completed_useful = graded[(graded.immediateActionsTaken == 'Action Completed Onsite') &
                              (graded.grade == 1)]
    pipe = get_pipeline()
    coords = pipe.transform(completed_useful.incidentDescription.dropna())
    x = coords[:, 2]
    y = coords[:, 3]
    line = LinearRegression(fit_intercept=True)
    line.fit(x.reshape(-1, 1), y)
    intercept = line.intercept_
    slope = line.coef_[0]
    return (intercept, slope)

def distance_to_line(line_intercept, line_slope, point_x, point_y):
    a = line_slope
    b = -1
    c = line_intercept
    return np.abs(a * point_x + b * point_y + c) / np.sqrt(a**2 + b**2)

def count_meaningful_event_types(eventTypes_column):
    '''
    The event types of most reports are either empty or vague.  I want to prioritize
    reports that have meaningful event types.
    '''
    meaningful_types =  {'Fire/Explosion',
                         'Injury/Illness',
                         'Material Release',
                         'Near Miss',
                         'Property Damage',
                         'Security'}
    return eventTypes_column.apply(lambda s: sum([typ in s for typ in meaningful_types]))

'''
NO ACTION NECESSARY

No Action Necessary (and probably Further Action Necessary) can be ranked based on proximity
to the Stop the Job cluster.
'''

def find_stop_job_center():
    pipe = get_pipeline()
    reports = pd.read_csv('my_data/combined_reports.csv')
    comments = reports[reports.immediateActionsTaken == 'Stop the Job'].incidentDescription.astype(str)
    coordinates = pipe.transform(comments)
    x, y = coordinates[:, 3], coordinates[:, 1]
    x_center, y_center = x.mean(), y.mean()
    return (x_center, y_center)


class ReportSorter(object):

    def __init__(self):
        self.pipe_SVD = get_pipeline()
        self.completed_intercept, self.completed_slope = get_coefficients_of_line()
        self.stop_job_x, self.stop_job_y = find_stop_job_center()

    def score_no_action_necessary(self, comments):
        coordinates = self.pipe_SVD.transform(comments)
        x, y = coordinates[:, 3], coordinates[:, 1]
        return np.sqrt((self.stop_job_x - x)**2 + (self.stop_job_y - y)**2)

    def score_further_action_necessary(self, comments):
        coordinates = self.pipe_SVD.transform(comments)
        x, y = coordinates[:, 3], coordinates[:, 1]
        return np.sqrt((self.stop_job_x - x)**2 + (self.stop_job_y - y)**2)

    def score_action_completed_onsite(self, comments):
        coordinates = self.pipe_SVD.transform(comments)
        x, y = coordinates[:, 2], coordinates[:, 3]
        return distance_to_line(self.completed_intercept, self.completed_slope, x, y)

    def sort_reports(self, reports):
        '''
        Sort first by whether the report is labeled 'Stop the Job', 'Further Action Necessary',
        'Action Completed Onsite', or 'No Action Necessary'.
        Then sort by the number of meaningful types in the eventTypes column.
        Finally, sort by the Morrow Metric Score.
        '''
        pipe = get_pipeline()
        df = reports[['immediateActionsTaken', 'eventType', 'incidentDescription']].copy()
        df['typeCount'] = count_meaningful_event_types(df.eventType)
        df['flag_number'] = df.immediateActionsTaken.map({
                                                        'Stop the Job': 3,
                                                        'Further Action Necessary': 2,
                                                        'Action Completed Onsite': 1,
                                                        'No Action Necessary': 0
                                                        })
        stop_the_job = df[df.immediateActionsTaken == 'Stop the Job'].copy()
        stop_the_job['score'] = np.full(len(stop_the_job), 2)
        further_action_necessary = df[df.immediateActionsTaken == 'Further Action Necessary'].copy()
        further_action_necessary['score'] = self.score_further_action_necessary(
                        further_action_necessary.incidentDescription.astype(str)
                        )
        action_completed_onsite = df[df.immediateActionsTaken == 'Action Completed Onsite'].copy()
        action_completed_onsite['score'] = self.score_action_completed_onsite(
                        action_completed_onsite.incidentDescription.astype(str)
                        )
        no_action_necessary = df[df.immediateActionsTaken == 'No Action Necessary'].copy()
        no_action_necessary['score'] = self.score_no_action_necessary(
                        no_action_necessary.incidentDescription.astype(str)
                        )
        df_scored = pd.concat([stop_the_job,
                               further_action_necessary,
                               action_completed_onsite,
                               no_action_necessary])
        df_scored.sort_values(['flag_number', 'typeCount', 'score'], ascending=[False, False, True],
                        inplace=True)
        return df_scored[['immediateActionsTaken', 'eventType', 'score', 'incidentDescription']]


if __name__ == '__main__':
    reports = pd.read_csv('my_data/combined_reports.csv')
    reports.dropna(subset=['immediateActionsTaken', 'incidentDescription'], inplace=True)
    sample = reports.sample(20)
    comments = sample.incidentDescription.astype(str)
    rs = ReportSorter()
    df = rs.sort_reports(sample)
