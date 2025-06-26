import pickle
from collections import defaultdict
from itertools import product

import pandas as pd

from ressys_dp.utils.array import make_proximity_groups, get_overall_least_change_group
from tools.modular.adapters import (
    ADSAdapter,
    ModularTSVAdapterMMTelfer,
    LabelCSVAdapter,
)

minesites = ["macmahon-telfer"]
shifts = ["202202150", "201809030"]
sources = ["proprietary", "modular", "label"]

# Collect data
# Output: dict raw_dict
#   {(mid, sid, eid, label): {source: DF with start_time, time, mid_time, stop_time}}

spec_list = list(product(minesites, shifts, sources))
# TODO: Remove - DEBUG
spec = spec_list[0]
split_by = ["equipment_id", "label"]
params = ["start_time", "mid_time", "stop_time"]


def collect_data(spec, split_by, params):

    try:
        adapters
    except NameError:
        proprietary_adapter = ADSAdapter()
        modular_adapter = ModularTSVAdapterMMTelfer(
            "/home/chungfoong/Work/data/modular"
        )
        label_adapter = LabelCSVAdapter(
            "/home/chungfoong/Work/data-processing-labeled-data/LabelledData"
        )
        adapters = {
            "proprietary": proprietary_adapter,
            "modular": modular_adapter,
            # 'minestar': minestar_adapter,
            "label": label_adapter,
        }

    mid, sid, source = spec
    adapter = adapters[source]
    df = adapter.get_data(mid, sid)

    sdfs = {key: sdf[params] for key, sdf in df.groupby(split_by)}

    return sdfs


raw_dict = defaultdict(dict)  # {(mid, sid, eid, label): {source: DF}}
for spec in spec_list:
    mid, sid, source = spec
    dfs = collect_data(spec, split_by, params)
    for sub_key, df in dfs.items():  # sub_key: eid, label
        eid, label = sub_key
        key = (mid, sid, eid, label)
        raw_dict[key][source] = df

# Match/merge (using broad tolerance) data event to common index
# Output: dict initial_match_dict
#   {(mid, sid, eid, label): DF with (sources)_ + start_time, mid_time, stop_time}

initial_tolerance = 300
spec_list = raw_dict.keys()  # (mid, sid, eid, label)


def match_event_id(dfs, initial_tolerance):

    df = pd.concat([df.assign(source=source) for source, df in dfs.items()])
    df.sort_values(by=["start_time"], inplace=True)

    start_time_proximity_group = make_proximity_groups(
        df.loc[:, "start_time"], initial_tolerance
    )
    mid_time_proximity_group = make_proximity_groups(
        df.loc[:, "mid_time"], initial_tolerance
    )
    stop_time_proximity_group = make_proximity_groups(
        df.loc[:, "stop_time"], initial_tolerance
    )

    df["event_id"] = get_overall_least_change_group(
        start_time_proximity_group, mid_time_proximity_group, stop_time_proximity_group
    )

    return df


def match_common_index(df, sources, params):

    params = [params] if isinstance(params, str) else params
    required_columns = set(["source", "event_id"] + params)
    missing = required_columns.difference(set(df))
    assert not missing, f"DF is missing columns: {missing}"

    common_index_df = pd.DataFrame()

    for param in params:
        common_index_single_param_df = match_common_index_single_param(
            df, sources, param
        )
        common_index_single_param_df.rename(
            columns={source: source + "_" + param for source in sources}, inplace=True
        )
        common_index_df = pd.concat(
            [common_index_df, common_index_single_param_df], axis=1
        )

    return common_index_df


def match_common_index_single_param(df, sources, param):

    required_columns = {"source", param, "event_id"}
    missing = required_columns.difference(set(df))
    assert not missing, f"DF is missing columns: {missing}"

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # TODO: Implement better way to deal with repeating label - Maybe add to list to be classify as false positive
    # Remove repeating labels within tolerance by merging label instance and averaging their time
    df = df.groupby(["event_id", "source"]).mean().reset_index()
    common_index_single_param_df = pd.DataFrame(columns=sources)
    common_index_single_param_df = common_index_single_param_df.append(
        df.pivot(index=["event_id"], columns=["source"], values=param)
    )
    return common_index_single_param_df


# match_dict = defaultdict(dict) # {(mid, sid, eid, label): DF}
match_dict = {}
for spec in spec_list:
    df = match_event_id(raw_dict[spec], initial_tolerance)
    if set(df.loc[:, "source"]) == set(sources):
        match_dict[spec] = match_common_index(df, sources, params)

with open("match_dict.pkl", "wb") as f:
    pickle.dump(match_dict, f)
    print(f"Saved dict match_dict as match_dict.pkl")
