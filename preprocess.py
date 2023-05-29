import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def index_with_approach(orig_df):
    # 'pid' column 초기값 설정
    pid = 0
    pid_list = []

    # 'behavior' column의 값이 '물리적거리'일 때마다 pid 값을 1씩 증가시킴
    for behavior in orig_df['behavior']:
        if behavior == '물리적거리':
            pid_list.append(pid)
            pid += 1
        else:
            pid_list.append(pid_list[-1])

    # 'pid' column을 DataFrame에 추가
    orig_df['pid'] = pid_list
    
    print(orig_df.head(20))

    return orig_df


def index_with_start(df):
    pids = []
    current_pid = 0

    for i in range(len(df)):
        if i == 0:
            pids.append(current_pid)
            continue
        
        prev_row = df.iloc[i - 1]
        current_row = df.iloc[i]

        if prev_row['appearance'] == current_row['appearance'] and (current_row['start'] - (prev_row['start'] + prev_row['duration'])) < 2000:
            pids.append(current_pid)
        else:
            current_pid += 1
            pids.append(current_pid)

    df['pid'] = pids
    return df


def reindex(orig_df, columns):
    orig_df = orig_df.reindex(columns=columns)
    return orig_df


def cal_time_ratio(orig_df):
    grouped = orig_df.groupby('pid')
    total_duration = grouped['duration'].transform('sum')
    orig_df['time_ratio'] = orig_df['duration'] / total_duration
    return orig_df


def map_code_score(orig_df, scores):
    """
    scores = {'Pass': 1, 'Follow': 2, 'Avoid': 2, 'Approach': 3, 
              'None': 4, 'Touch': 5, 'Gesture': 5}
    """
    orig_df['code_score'] = orig_df['code'].map({ 'Pass': scores['Pass'], 'Approach': scores['Approach'], 'Follow': scores['Follow'], 'Avoid': scores['Avoid'],
                                        'None': scores['None'], 'Touch': scores['Touch'], 'Gesture': scores['Gesture']})
    return orig_df


def normalize(orig_df, col, scale=100):
    min_score = orig_df[col].min()
    max_score = orig_df[col].max()
    orig_df['norm_engagement_score'] = (orig_df[col] - min_score) / (max_score - min_score) * scale
    
    return orig_df


def make_data(df):
    df = df.groupby('pid').agg({'code': list, 'duration': list, 'time_ratio': list, 'code_score': list, 'A/C': 'first', 'M/F': 'first', 'appearance': 'first'})
    df = df.drop(df[df['code'].apply(lambda x: len(x) <= 1)].index)
    df['code_count'] = [len(x) for x in df['code']]
    
    return df


def cal_engagement_score(df):
    engagement_scores = []
    for i in df.index:
        engagement_score = []
        for code_score, time_ratio in zip(df['code_score'][i], df['time_ratio'][i]):
            engagement_score.append(code_score * time_ratio)
        engagement_scores.append(sum(engagement_score))

    df['engagement_score'] = engagement_scores

    return df


def normalize(df, col_name, scale=100):
    min_score = df[col_name].min()
    max_score = df[col_name].max()
    df['norm_engagement_score'] = (df[col_name] - min_score) / (max_score - min_score) * scale
    
    return df


def classify_level(df, threshold=50):
    engagement_levels = []
    for i in df.index:
        if df['norm_engagement_score'][i] > threshold:
            engagement_levels.append('high')
        else:
            engagement_levels.append('low')
            
    df['engagement_level'] = engagement_levels
    
    return df


def draw_dist_score(df):
    sns.scatterplot(x='pid', y='norm_engagement_score', data=df)
    plt.title('Distribution of Normalized Engagement Scores')
    plt.show()


def draw_dist_level(df):
    palette = {"high": "#4878d0", "low": "#ffa07a"}
    sns.scatterplot(x='pid', y='norm_engagement_score', hue='engagement_level', palette=palette, data=df)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title("Distribution of Engagement Level")
    plt.show()


def store_df(df, path):
    df.to_csv(path)


if __name__ == "__main__":
    # load dataset
    orig_df = pd.read_csv('./data/exhibition_behavior_preprocessed.csv', delimiter=',', index_col=False)
    # orig_df = orig_df.drop(['pid', 'time_ratio', 'engagement_score', 'norm_engagement_score', 'engagement_level'], axis=1)
    
    # make pid 
    orig_df = preprocess.indexing(orig_df)
    orig_df = preprocess.reindex(orig_df)

    # calculate time ratio for each code among the same pid
    orig_df = preprocess.cal_time_ratio(orig_df)

    # map code score
    scores = {'Pass': 1, 'Follow': 2, 'Avoid': 2, 'Approach': 3, 
              'None': 4, 'Touch': 5, 'Gesture': 5}
    orig_df = preprocess.map_code_score(orig_df, scores)

    # make data for decision tree
    df = make_data(orig_df)

    # calculate engagement score
    df = cal_engagement_score(df)

    # normalize engagement score
    df = normalize(df, 'engagement_score')

    # classify engagement level
    df_50 = classify_level(df)
    df_70 = classify_level(df, threshold=70)

    # draw scatter plot for the distribution of engagement level
    draw_dist_level(df_50)
    draw_dist_level(df_70)