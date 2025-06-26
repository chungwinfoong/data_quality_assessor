import operator
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd

source_dict = {
    "truth": {
        "label": "label_start_time"  # any valid column name with nan representing unmatch event
    },
    "analysed": {
        "proprietary": "proprietary_start_time",
        "modular": "modular_start_time",
    },
}

comparison_matrix_dict = {
    "match_duration": {
        "thresholds": np.linspace(-50, 200, 26),
        "match_operator": operator.ge,
    },
    "match_percentage_wrt_truth_duration": {
        "thresholds": np.linspace(0, 1, 26),
        "match_operator": operator.ge,
    },
    "match_percentage_wrt_analysed_duration": {
        "thresholds": np.linspace(0, 1, 26),
        "match_operator": operator.ge,
    },
    "match_percentage_wrt_total_duration": {
        "thresholds": np.linspace(0, 1, 26),
        "match_operator": operator.ge,
    },
    "delta_mid_time": {
        "thresholds": np.linspace(0, 250, 26),
        "match_operator": operator.lt,
    },
}

# aggregate_filtre_list ={
#     'shift_id_list':[],
#     'equipment_id_list':[],
#     'label_list':['Dump','Load'],
#     'source_list':[],
#     'tolerance_list':{
#         'match_duration': [0],
#         'match_percentage_wrt_truth_duration': [0.04, 0.2, 0.4],
#         'match_percentage_wrt_analysed_duration': [0.04, 0.2, 0.4],
#         'match_percentage_wrt_total_duration': [0.04, 0.2, 0.4],
#         'delta_mid_time':[250]
#     }
# }

# Calculate match data comparison matrix
# Output: dict match_dict
#   {(mid, sid, eid, label): DF with (sources)_ + start_time, mid_time, stop_time, duration, delta, match pecentages}

params = ["start_time", "mid_time", "stop_time"]
match_dict_pkl_dir = "match_dict.pkl"
with open(match_dict_pkl_dir, "rb") as f:
    match_dict = pickle.load(
        f
    )  # {(mid, sid, eid, label): DF with (sources)_ + start_time, mid_time, stop_time}


def calculate_comparison_matrix(df, params, source_dict):
    truth_source = list(source_dict["truth"].keys())[0]
    analysed_sources = source_dict["analysed"].keys()
    for source in analysed_sources:
        for param in params:
            df = calculate_abs_delta(df, param, truth_source, source)
        df = calculate_durations_and_match_percentage(df, truth_source, source)

    return df


def calculate_abs_delta(df, param, truth_source, analysed_source):
    required_columns = {truth_source + "_" + param, analysed_source + "_" + param}
    missing = required_columns.difference(set(df))
    assert not missing, f"DF is missing columns: {missing}"

    df[analysed_source + "_delta_" + param] = (
        df.loc[:, required_columns].diff(axis=1).abs().iloc[:, 1]
    )

    return df


def calculate_durations_and_match_percentage(df, truth_source, analysed_source):
    required_columns = {
        source + suffix
        for suffix in ["_start_time", "_stop_time"]
        for source in [truth_source, analysed_source]
    }
    missing = required_columns.difference(set(df))
    assert not missing, f"DF is missing columns: {missing}"

    match_mask = (
        df.loc[:, truth_source + "_stop_time"].notna()
        & df.loc[:, analysed_source + "_stop_time"].notna()
    )
    df[truth_source + "_duration"] = (
        df.loc[match_mask, truth_source + "_stop_time"]
        - df.loc[match_mask, truth_source + "_start_time"]
    )
    df[analysed_source + "_duration"] = (
        df.loc[match_mask, analysed_source + "_stop_time"]
        - df.loc[match_mask, analysed_source + "_start_time"]
    )
    df[analysed_source + "_match_duration"] = df.loc[
        match_mask, [truth_source + "_stop_time", analysed_source + "_stop_time"]
    ].min(axis=1) - df.loc[
        match_mask, [truth_source + "_start_time", analysed_source + "_start_time"]
    ].max(
        axis=1
    )
    # df.loc[df.loc[:,analysed_source+'_match_duration']<0, analysed_source+'_match_duration'] = 0
    df[analysed_source + "_total_duration"] = df.loc[
        match_mask, [truth_source + "_stop_time", analysed_source + "_stop_time"]
    ].max(axis=1) - df.loc[
        match_mask, [truth_source + "_start_time", analysed_source + "_start_time"]
    ].min(
        axis=1
    )

    df[analysed_source + "_match_percentage_wrt_truth_duration"] = (
        df[analysed_source + "_match_duration"] / df[truth_source + "_duration"]
    )
    df[analysed_source + "_match_percentage_wrt_analysed_duration"] = (
        df[analysed_source + "_match_duration"] / df[analysed_source + "_duration"]
    )
    df[analysed_source + "_match_percentage_wrt_total_duration"] = (
        df[analysed_source + "_match_duration"]
        / df[analysed_source + "_total_duration"]
    )
    # df.loc[df.loc[:, 'matched_percentage_wrt_total'] < 0, 'matched_percentage_wrt_'] = 0

    return df


for key, match_df in match_dict.items():
    match_dict[key] = calculate_comparison_matrix(match_df, params, source_dict)

# Evaluate matches at varying matrix thresholds
# Output: dict match_bool_dict
#   {(mid, sid, eid, label, comparison_matrix): {analysed_source: bool DF with comparison_matrix threshold vs events}}


def evaluate_comparison_matrix(df, comparison_matrix, thresholds, match_operator):

    required_columns = {comparison_matrix}
    missing = required_columns.difference(set(df))
    assert not missing, f"DF is missing columns: {missing}"

    eval_df = pd.concat(
        [match_operator(df[comparison_matrix], th).rename(th) for th in thresholds],
        axis=1,
    )

    return eval_df


match_bool_dict = defaultdict(
    dict
)  # {(mid, sid, eid, label, comparison_matrix): {analysed_source: DF}}
for key, match_df in match_dict.items():
    mid, sid, eid, label = key
    for (
        comparison_matrix,
        comparison_matrix_properties,
    ) in comparison_matrix_dict.items():
        for analysed_source in source_dict["analysed"].keys():
            key = (mid, sid, eid, label, comparison_matrix)
            match_bool_dict[key][analysed_source] = evaluate_comparison_matrix(
                match_df,
                analysed_source + "_" + comparison_matrix,
                # **comparison_matrix_list[comparison_matrix],
                comparison_matrix_properties["thresholds"],
                comparison_matrix_properties["match_operator"],
            )

# Calculate precision and recall at varying matrix thresholds
# Output: dict precision_recall_dict
#   {(mid, sid, eid, label, comparison_matrix): {analysed_source: DF with TP, FP, FN, precision, recall,...
#   vs threshold}}


def calculate_precision_recall(
    match_df, match_bool_df, any_truth_source_col, any_analysed_source_col
):
    required_columns = {any_truth_source_col, any_analysed_source_col}
    missing = required_columns.difference(set(match_df))
    assert not missing, f"DF is missing columns: {missing}"

    precision_recall_df = pd.DataFrame()

    precision_recall_df["true_positive"] = match_bool_df.sum(axis=0)
    precision_recall_df["total_positives_truth"] = (
        match_df.loc[:, any_truth_source_col].notna().sum()
    )
    precision_recall_df["total_positives_analysed"] = (
        match_df.loc[:, any_analysed_source_col].notna().sum()
    )

    precision_recall_df["precision"] = (
        precision_recall_df["true_positive"]
        / precision_recall_df["total_positives_analysed"]
    )
    precision_recall_df["recall"] = (
        precision_recall_df["true_positive"]
        / precision_recall_df["total_positives_truth"]
    )
    # unmatched data contributes to both false positive and false negative
    precision_recall_df["false_positive"] = (
        precision_recall_df["total_positives_analysed"]
        - precision_recall_df["true_positive"]
    )
    precision_recall_df["false_negative"] = (
        precision_recall_df["total_positives_truth"]
        - precision_recall_df["true_positive"]
    )
    precision_recall_df["event_count"] = (
        precision_recall_df.loc[:, ["true_positive", "false_positive"]]
        .sum(axis=1)
        .astype(int)
    )

    return precision_recall_df


def calculate_both_false_label(match_df, match_bool_dfs, any_truth_source_col):
    required_columns = {any_truth_source_col}
    missing = required_columns.difference(set(match_df))
    assert not missing, f"DF is missing columns: {missing}"

    both_false_label_df = pd.DataFrame(index=match_bool_dfs[0].columns)
    both_false_label_df["both_false_positive"] = 0
    both_false_label_df["both_false_negative"] = 0

    unmatch_mask_df = ~match_bool_dfs[0] & ~match_bool_dfs[1]

    for tolerance in unmatch_mask_df.columns:
        for event_id in unmatch_mask_df.index:
            if (
                unmatch_mask_df.loc[event_id, tolerance]
                & match_df.isna().loc[event_id, any_truth_source_col]
            ):
                both_false_label_df.loc[tolerance, "both_false_positive"] += 1
            elif (
                unmatch_mask_df.loc[event_id, tolerance]
                & match_df.notna().loc[event_id, any_truth_source_col]
            ):
                both_false_label_df.loc[tolerance, "both_false_negative"] += 1

    both_false_label_df = both_false_label_df.astype(int)

    return both_false_label_df


precision_recall_dict = defaultdict(
    dict
)  # {(mid, sid, eid, label, comparison_matrix): {analysed_source: DF}}
for key, match_df in match_dict.items():
    mid, sid, eid, label = key
    for comparison_matrix in comparison_matrix_dict.keys():
        for analysed_source in source_dict["analysed"].keys():
            sub_key = (mid, sid, eid, label, comparison_matrix)
            precision_recall_dict[sub_key][analysed_source] = (
                calculate_precision_recall(
                    match_df,
                    match_bool_dict[sub_key][analysed_source],
                    source_dict["truth"]["label"],
                    source_dict["analysed"][analysed_source],
                )
            )
        both_false_label_df = calculate_both_false_label(
            match_df,
            [
                match_bool_dict[(mid, sid, eid, label, comparison_matrix)][
                    analysed_source
                ]
                for analysed_source in source_dict["analysed"].keys()
            ],
            source_dict["truth"]["label"],
        )
        for analysed_source in source_dict["analysed"].keys():
            key = (mid, sid, eid, label, comparison_matrix)
            precision_recall_dict[key][analysed_source] = pd.concat(
                [precision_recall_dict[key][analysed_source], both_false_label_df],
                axis=1,
            )

# Merge precision recall data for different analysed sources to similar index
# Output: dict merged_precision_recall_dict
#    {(mid, sid, eid, label, comparison_matrix):
#       DF with sources_TP, sources_FP, sources_FN, sources_eventcount, both_FP, both_FN, sources_precision,
#       sources_recall vs threshold}

sources = source_dict["analysed"].keys()


def merge_precision_recall_sources(precision_recall_dfs):
    df = pd.concat(
        [
            df.drop(columns=["both_false_positive", "both_false_negative"]).add_prefix(
                source + "_"
            )
            for source, df in precision_recall_dfs.items()
        ],
        axis=1,
    )
    df = pd.concat(
        [
            df,
            list(precision_recall_dfs.items())[0][1].loc[
                :, ["both_false_positive", "both_false_negative"]
            ],
        ],
        axis=1,
    )
    return df


# merged_precision_recall_dict = defaultdict(dict) # {(mid, sid, eid, label, comparison_matrix): DF}
merged_precision_recall_dict = {}
for key, dfs in precision_recall_dict.items():
    mid, sid, eid, label, comparison_matrix = key
    merged_precision_recall_dict[key] = merge_precision_recall_sources(dfs)

with open("match_dict.pkl", "wb") as f:
    pickle.dump(match_dict, f)
    print(f"Saved dict match_dict as match_dict.pkl")
with open("precision_recall_dict.pkl", "wb") as f:
    pickle.dump(precision_recall_dict, f)
    print(f"Saved dict precision_recall_dict as precision_recall_dict.pkl")
with open("merged_precision_recall_dict.pkl", "wb") as f:
    pickle.dump(merged_precision_recall_dict, f)
    print(
        f"Saved dict merged_precision_recall_dict as merged_precision_recall_dict.pkl"
    )
