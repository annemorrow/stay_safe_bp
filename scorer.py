import numpy as np
import pandas as pd
import os.path
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pickle

def get_pipeline():
    '''
    Either load or create the pipeline that converts the incident descriptions
    into cartesian coordinates in 4 dimensional TruncatedSVD space (dimensions 3
    and 1 are the important ones)
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

def score_comments(comments, pipe=None):
    '''
    The Morrow Metric:  with the pipeline's dimension 3 as x and dimention 1 as y,
    return y - x.
    '''
    if pipe == None:
        pipe = get_pipeline()
    coordinates = pipe.transform(comments)[:, [3, 1]]
    return coordinates[:,1] - coordinates[:,0]

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

def sort_reports(reports):
    '''
    Sort first by whether the report is labeled 'Stop the Job', 'Further Action Necessary',
    'Action Completed Onsite', or 'No Action Necessary'.
    Then sort by the number of meaningful types in the eventTypes column.
    Finally, sort by the Morrow Metric Score.
    '''
    pipe = get_pipeline()
    df = reports[['immediateActionsTaken', 'eventType', 'incidentDescription']].copy()
    df['flag_number'] = df.immediateActionsTaken.map({
                                                    'Stop the Job': 3,
                                                    'Further Action Necessary': 2,
                                                    'Action Completed Onsite': 1,
                                                    'No Action Necessary': 1
                                                    })
    df['typeCount'] = count_meaningful_event_types(df.eventType)
    df['score'] = score_comments(df.incidentDescription, pipe=pipe)
    df.sort_values(['flag_number', 'typeCount', 'score'], ascending=False, inplace=True)
    return df[['immediateActionsTaken', 'eventType', 'score', 'incidentDescription']]


if __name__ == '__main__':
    reports = pd.read_csv('my_data/combined_reports.csv')
    reports.dropna(subset=['immediateActionsTaken', 'incidentDescription'], inplace=True)
    sample = reports.sample(20)
    print(sort_reports(sample))
