import pandas as pd
import numpy as np
from load_original_data import old_reports, new_reports


'''
I'm trying to keep as much information as possible from the new reports, but
in some cases, the column didn't have useful information, or that information
could not be extracted from the old reports.  Therefore, in order to have one
clean dataset, I'm only keeping columns that either exist or can be derived
from the old data.
'''

# keep if the column already exists in both datasets
shared = list(set(old_reports.columns) & set(new_reports.columns))
# the following column names don't exist in the old dataset but can be derived
derived = ['assetType',
           'incidentDescription',
           'operationOrDevelopment',
           'eventClassification',
           'jobTypeObserved',
           'event',
           'eventType',
           'latitude',
           'longitude',
           'name']

# The new data has an assetType column that I love, but which doesn't exist in
# the old data.  However, the old data has an id (spread across two columns)
# which is indicative of assetType
def get_id(row):
    if not pd.isnull(row.locationWellSite):
        return row.locationWellSite
    if not pd.isnull(row.assetId):
        return row.assetId
    return np.nan

def well_or_fac(id_string):
    if id_string == 'Other':
        return "Other"
    if 'PAD' in str(id_string):
        return "Well"  #Confirm with Ben because he knows what PADs are
    if 'CDP' in str(id_string):
        return "Well"
    if 'FAC' in str(id_string):
        return "Facility"
    len_dict = {16: "Well",
                38: "Well",
                10: "Facility",
                17: "Facility",
                18: "Facility",
                19: "Other"}
    if pd.isnull(id_string):
        return np.nan
    if len(id_string) in len_dict:
        return len_dict[len(id_string)]
    return "Well"

def old_assetType():
    return old_reports.apply(lambda row: well_or_fac(get_id(row)), axis=1)


# The old data doesn't have longitude or latitude, but that's pretty much determined
# by replicateGroup.

def get_lon(string):
    try:
        return new_reports.groupby('replicateGroup')['longitude'].mean()[string]
    except:
        return np.nan

def get_lat(string):
    try:
        return new_reports.groupby('replicateGroup')['latitude'].mean()[string]
    except:
        return np.nan

# There's only one comment column in the new data, but multiple free-text columns
# in the old data, which I've put together into one string.
def combined_comments():
    key_words = ['other', 'description', 'detail', 'comment', 'notlisted']
    free_text_cols = [col for col in old_reports.columns for word in key_words\
                      if word in col.lower()]
    comments = old_reports[free_text_cols].fillna('')
    combined_comments = comments.apply(\
                      lambda row: reduce(lambda x, y: x + ' ' + str(y), row), axis=1)
    return combined_comments

def prepare_old_data():
    # start with what's already there
    df = old_reports[shared]
    # derive from old data
    df['assetType'] = old_assetType()
    df['assetId'] = old_reports.apply(lambda row: get_id(row), axis=1)
    df['incidentDescription'] = combined_comments()
    df['operationOrDevelopment'] = old_reports['userFunction']
    df['latitude'] = old_reports.replicateGroup.apply(get_lat)
    df['longitude'] = old_reports.replicateGroup.apply(get_lon)
    df['name'] = old_reports.locationWellSiteName
    df['eventType'] = old_reports.eventTitle
    df['jobTypeObserved'] = old_reports.jobGroup
    df['eventClassification'] = old_reports.eventTitle.apply(lambda s: "Verification" if\
                       "Verification" in s else "Unknown")
    df['event'] = old_reports.actualConsequences.apply(lambda s: "Observation" if\
                     s == '[]' else "Incident")
    return df[shared + derived] # make sure order is consistent

def prepare_new_data():
    return new_reports[shared + derived]

def concatenate_data():
    cleaned_old = prepare_old_data()
    cleaned_new = prepare_new_data()
    return pd.concat([cleaned_old, cleaned_new])

if __name__ == '__main__':
    combined_df = concatenate_data()
    combined_df.to_csv('my_data/combined_reports.csv', index=False)
