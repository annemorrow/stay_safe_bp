import numpy as np
import pandas as pd
import pickle

def daysBeforeModification(reports):
    created = pd.to_datetime(reports['serverCreatedDate'])
    modified = pd.to_datetime(reports['serverModifiedDate'])
    delta = modified - created
    days = delta.apply(lambda dt: dt.days)
    return pd.DataFrame(days.astype(int), columns=['daysBeforeModification'])

def transfered(reports):
    df =  pd.DataFrame(reports['modifiedBy'] != reports['createdBy']).astype(int)
    df.columns = ['transfered']
    return df

def jobType(reports):
    jobTypeObserved_dict = dict()
    for col in reports.jobTypeObserved.dropna().unique():
        if 'operations' in col.lower():
            jobTypeObserved_dict[col] = 'Operations'
        elif 'maintenance' in col.lower():
            jobTypeObserved_dict[col] = 'Maintenance'
        elif 'wells' in col.lower():
            jobTypeObserved_dict[col] = "Well Intervention"
        elif 'construction' in col.lower():
            jobTypeObserved_dict[col] = 'Construction'
        else:
            jobTypeObserved_dict[col] = col
        jobTypeObserved_dict['Produced Fluid Management'] = 'Other'
        jobTypeObserved_dict['Rig Move'] = 'Other'
    jobType = reports['jobTypeObserved'].map(jobTypeObserved_dict)
    d =  pd.get_dummies(jobType)
    wanted_cols = ['Automation',
                    'Completions',
                    'Construction',
                    'Drilling',
                    'Maintenance',
                    'Operations',
                    'Well Intervention']
    for col in wanted_cols:
        if col not in d.columns:
            d[col] = np.zeros(len(d))
    d = d[wanted_cols]
    return d

def eventType(reports):
    shared_events = {'Fire/Explosion',
                     'Injury/Illness',
                     'Material Release',
                     'Near Miss',
                     'Property Damage',
                     'Security'}
    event_dummies = pd.DataFrame()
    for col in shared_events:
        event_dummies[col] = reports.eventType.apply(lambda s: col in s).astype(int)
    return event_dummies

from string import whitespace
def count_non_whitespace(s):
    for c in whitespace:
        s.replace(c, '')
    return len(s)

def incidentDescriptionLength(reports):
    l = pd.DataFrame(reports.incidentDescription.astype(str).apply(count_non_whitespace))
    l.columns=['commentLength']
    return l

def incidentDescription(reports):
    with open('SVD_pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
    coords = pd.DataFrame(index=reports.index)
    coords_arrays = pipe.transform(reports.incidentDescription.astype(str))
    coords['description_x'] = coords_arrays[:,3]
    coords['description_y'] = coords_arrays[:,1]
    return coords

def isBP(reports):
    df =  pd.DataFrame((reports['companyInvolved'] == 'BP').astype(int))
    df.columns = ['isBP']
    return df


def immediateActionsTaken(reports):
    return pd.get_dummies(reports.immediateActionsTaken)

def assetType(reports):
    return pd.get_dummies(reports.assetType)[['Well', 'Facility']]

def replicateGroup(reports):
    return pd.get_dummies(reports.replicateGroup)

def serverCreatedDate(reports):
    return pd.to_datetime(reports['serverCreatedDate'])

def latlon(reports):
    return reports[['latitude', 'longitude']].fillna(0)

def event(reports):
    return pd.DataFrame(reports.event.map({'Observation': 0, 'Incident': 1}))

def concat(reports):
    df = pd.DataFrame(index=reports.index)
    my_cols = ['serverCreatedDate',
               'daysBeforeModification',
               'incidentDescription',
               'immediateActionsTaken',
               'transfered',
               'assetType',
               'jobType',
               'eventType',
               'isBP',
               'replicateGroup',
               'latlon']
    for col in my_cols:
        df = pd.concat([df, globals()[col](reports)], axis=1)
    return df

def selective_concat(reports):
    return pd.concat([event(reports),
                   jobType(reports),
                   incidentDescription(reports),
                   incidentDescriptionLength(reports)], axis=1)

if __name__ == '__main__':
    reports = pd.read_csv('my_data/combined_reports.csv').set_index('seq')
    numeric_reports = concat(reports)
    numeric_reports.to_csv('my_data/numeric_reports.csv')
