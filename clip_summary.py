# clip summary down to 70 days

import pickle
import pandas as pd
# import numpy as np
from userid_map import create_map
import datetime as dt
import argparse
import os
import glob

pd.set_option('display.max_rows', 1000)

id_mapping = create_map('./data/mperf_ids.txt')

# ------------- functions -------------

def get_index_of_first_valid_day(null_marker_series):
    
    # increment index until the first False is found
    # (False indicates that not that all marker values were NaN on that day)
    valid_day_index=0
    while null_marker_series[valid_day_index] is True:
        valid_day_index = valid_day_index+1
    
    return valid_day_index

def trim_data(subj_df, idx, days=70):
    return subj_df.iloc[idx:idx+70].copy()

def main(args):
    # data_file = args.data_file
    data_directory = args.data_directory
    days = args.days
    verbose = args.verbose

    # find latest summary file in directory
    summary_files = []
    os.chdir(data_directory)
    for file in glob.glob("master_summary_dataframe*"):
        summary_files.append(file)
  
    summary_files.sort()

    print("found files:")
    for s in summary_files:
        print(s)

    data_file = summary_files[-1]

    print("\nusing latest summary: {}".format(data_file))

    out = pickle.load( open(data_file, "rb" ) )
    df=out["dataframe"]
    md=out["metadata"]

    if verbose is None:
        verbose = False

    df_out = pd.DataFrame()

    qual_cols = [c for c in df.columns if 'qualtrics' in c]

    ids = []
    for id in df.index.get_level_values('Participant').unique():
        ids.append(id)

    for id in ids:
        
        if verbose:
            print(id)
        
        mperf_id = id_mapping[id]
        
        # get one subject's data
        subj_df = df.loc[id]
                
        # drop all non-qualtrics days
        subj_df.dropna(axis=0, subset=qual_cols, how='all', inplace=True)
        
        # get marker data only, no qualtrics columns
        subj_marker_df = subj_df.drop(qual_cols, axis=1)
        
        # indicate where markers are fully absent (everywhere .isnull().all() == True)
        null_marker_series = subj_marker_df.isnull().all(axis=1)
        
        # check for subject with no data
        if (null_marker_series is None) or (null_marker_series.shape[0] < 1):
            if null_marker_series is None:
                print("problem found in data for subject {}.  nan check returned None.".format(id))
            else:
                print("problem found in data for subject {}.  nan check length: {}.".format(id, len(null_marker_series)))
            continue
        
        # find index of first valid day
        idx = get_index_of_first_valid_day(null_marker_series)

        # trim df to prescribed number of days from first valid day
        subj_df = trim_data(subj_df, idx, days=days)
        
        # add subject-level index back in
        subj_df = pd.concat([subj_df], keys=[id], names=['Participant'])
        
        if verbose:
            print("first valid date: {}".format(subj_df.index.get_level_values("Date")[idx]))
            print("valid days: {}".format(subj_df.shape[0]))
            print("last valid date: {}\n".format(subj_df.index.get_level_values("Date")[subj_df.shape[0] - 1]))
        
        df_out = pd.concat([df_out, subj_df])
        
    print(df_out.shape)

    # save out file for next stage
    out["dataframe"] = df_out
    with open('clipped_summary.pkl', 'wb+') as f:
        pickle.dump(out, f, protocol=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="clips summary to correct number of qualtrics days per participant")
    # parser.add_argument("--data-file", help="name of the data file in master_data_file/")
    parser.add_argument("--data-directory", help="directory with the master summary file to trim")
    parser.add_argument("--days", default=70, help="number of qualtrics days per participant to clip down to")
    parser.add_argument("--verbose", action='store_const', const=True, help="add --verbose flag to see output")
    args = parser.parse_args()

    main(args)