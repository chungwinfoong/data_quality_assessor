import pandas as pd
import argparse
import pickle

from ressys_dp.tools import Minesite
from ressys_dp.utils.array import make_proximity_groups, get_overall_least_change_group
from tools.modular.adapters import (
    ADSAdapter,
    ModularTSVAdapter,
    LabelCSVAdapter,
    ModularTSVAdapterMMTelfer,
    ModularTSVAdapterMMTropicana,
)
from tools.modular.comparison import (
    merge_data_attributes_to_similar_event_index,
    calculate_abs_delta,
    merge_multiple_data_attributes_to_similar_event_index,
)
from tools.modular.presentation import calculate_durations_and_match_percentage


def main(minesite_id, shift_id, source1, source2, initial_tolerance=300):

    source_list = [source1, source2]
    baseline = ["label"]
    raw_data_attribute_list = ["start_time", "mid_time", "stop_time"]

    # Initialise adapters
    proprietary_adapter = ADSAdapter()
    modular_adapter = ModularTSVAdapterMMTelfer("/home/chungfoong/Work/data/modular")
    label_adapter = LabelCSVAdapter(
        "/home/chungfoong/Work/data-processing-labeled-data/LabelledData"
    )
    adapters = {
        "proprietary": proprietary_adapter,
        "modular": modular_adapter,
        # 'minestar': minestar_adapter,
        "label": label_adapter,
    }

    # Collect data from sources
    data = {
        source: adapters[source].get_data(minesite_id, shift_id)
        for source in source_list + baseline
    }
    combined_data = pd.concat(data.values()).sort_values(by=["equipment_id", "time"])

    equipment_list = list(set(combined_data.loc[:, "equipment_id"]))
    equipment_list = (
        equipment_list if isinstance(equipment_list, list) else [equipment_list]
    )
    label_list = list(set(combined_data.loc[:, "label"]))

    for equipment_id in equipment_list:
        index_mask = combined_data.loc[:, "equipment_id"] == equipment_id
        combined_data.loc[index_mask, "stop_time"] -= combined_data.loc[
            index_mask, "start_time"
        ].min()
        combined_data.loc[index_mask, "mid_time"] -= combined_data.loc[
            index_mask, "start_time"
        ].min()
        combined_data.loc[index_mask, "start_time"] -= combined_data.loc[
            index_mask, "start_time"
        ].min()

    # Sort combined data using multiIndex - equipment_id and label
    dict_data = {
        equipment_id: {
            label: combined_data.loc[
                (combined_data.loc[:, "equipment_id"] == equipment_id)
                & (combined_data.loc[:, "label"] == label),
                ["source"] + raw_data_attribute_list,
            ]
            for label in label_list
        }
        for equipment_id in equipment_list
    }
    #
    sorted_data = combined_data.set_index(["equipment_id", "label"]).sort_index()

    # Initial match of events judging from either start_time, mid_time or stop_time within high tolerance
    for equipment_id in equipment_list:
        for label in label_list:
            dict_data[equipment_id][label]["match_id_start_time"] = (
                make_proximity_groups(
                    dict_data[equipment_id][label].loc[:, "start_time"],
                    initial_tolerance,
                )
            )
            dict_data[equipment_id][label]["match_id_mid_time"] = make_proximity_groups(
                dict_data[equipment_id][label].loc[:, "mid_time"], initial_tolerance
            )
            dict_data[equipment_id][label]["match_id_stop_time"] = (
                make_proximity_groups(
                    dict_data[equipment_id][label].loc[:, "stop_time"],
                    initial_tolerance,
                )
            )
            dict_data[equipment_id][label].reset_index(drop=True, inplace=True)
            dict_data[equipment_id][label]["event_id"] = ""
            for idx in dict_data[equipment_id][label].index:
                dict_data[equipment_id][label].loc[idx, "event_id"] = (
                    dict_data[equipment_id][label]
                    .loc[
                        idx,
                        [
                            "match_id_start_time",
                            "match_id_mid_time",
                            "match_id_stop_time",
                        ],
                    ]
                    .min()
                )
                for match_id in [
                    "match_id_start_time",
                    "match_id_mid_time",
                    "match_id_stop_time",
                ]:
                    if (
                        dict_data[equipment_id][label].loc[idx, "event_id"]
                        < dict_data[equipment_id][label].loc[idx, match_id]
                    ):
                        dict_data[equipment_id][label].loc[
                            idx : dict_data[equipment_id][label].index[-1], match_id
                        ] -= 1
    equipment_list = list(set([x[0] for x in sorted_data.index]))
    label_list = list(set([x[1] for x in sorted_data.index]))
    #
    sorted_data["event_id"] = ""
    start_time_proximity_group = sorted_data.groupby(
        sorted_data.index
    ).start_time.apply(make_proximity_groups, tolerance=initial_tolerance)
    mid_time_proximity_group = sorted_data.groupby(sorted_data.index).mid_time.apply(
        make_proximity_groups, tolerance=initial_tolerance
    )
    stop_time_proximity_group = sorted_data.groupby(sorted_data.index).stop_time.apply(
        make_proximity_groups, tolerance=initial_tolerance
    )
    for index in start_time_proximity_group.index:
        sorted_data.loc[index, "event_id"] = get_overall_least_change_group(
            start_time_proximity_group[index],
            mid_time_proximity_group[index],
            stop_time_proximity_group[index],
        )

    # TODO: Remove Time - DEBUG
    import time

    t = time.time()
    elapsed_dict = time.time() - t
    t = time.time()
    elapsed_df = time.time() - t
    # Merge events/label instances into similar events/df index with high tolerance
    merged_data = {
        equipment_id: {
            label: merge_multiple_data_attributes_to_similar_event_index(
                dict_data[equipment_id][label],
                source_list + baseline,
                raw_data_attribute_list,
            )
            for label in label_list
        }
        for equipment_id in equipment_list
    }

    sorted_group = sorted_data.groupby(sorted_data.index)
    merged_data_v2 = pd.DataFrame()
    for group in sorted_group.groups.keys():
        pd.concat(
            [
                merged_data_v2,
                merge_multiple_data_attributes_to_similar_event_index(
                    sorted_group.get_group(group),
                    source_list + baseline,
                    raw_data_attribute_list,
                ),
            ]
        )
    # merged_group = sorted_data.groupby(sorted_data.index).apply(merge_multiple_data_attributes_to_similar_event_index, source_list=source_list+baseline, merging_column_list=raw_data_attribute_list)

    # Calculate duration, delta, match percentage of merged data

    for source in source_list:
        for equipment_id in equipment_list:
            for label in label_list:
                for raw_data_attribute in raw_data_attribute_list:
                    merged_data[equipment_id][label] = calculate_abs_delta(
                        merged_data[equipment_id][label],
                        raw_data_attribute,
                        baseline[0],
                        source,
                    )
                merged_data[equipment_id][label] = (
                    calculate_durations_and_match_percentage(
                        merged_data[equipment_id][label], baseline[0], source
                    )
                )

    return combined_data, merged_data


# TODO: Remove - DEBUG
minesite_id = "macmahon-telfer"
source1 = "proprietary"
source2 = "modular"
tolerance = 300
# shift_id = '202202090' #-202110031'
# shift_list = shift_id.split('-')
# if len(shift_list) == 1:
#     None
# elif len(shift_list) == 2:
#     shift_list = Minesite.from_central_resources(minesite_id).get_shift_list_for_shifts(shift_list[0], shift_list[1])
# else:
#     raise AssertionError('Invalid shift_id input')

# tropicana
# shift_list = [202012130, 202012201]

# telfer
# shift_list = [202202150, 202110141, 201904011, 201904010, 201903010, 201809030, 201809031, 201809020, 201809021, 201809010, 201809011]
shift_list = [201809030]

merged_data = {}
combined_data = {}
for shift_id in shift_list:
    shift_combined_data, shift_merged_data = main(
        minesite_id, shift_id, source1, source2, tolerance
    )
    merged_data[shift_id] = shift_merged_data
    combined_data[shift_id] = shift_combined_data
    print(f"Finished importing Shift {shift_id}")

with open("combined_data.pkl", "wb") as f:
    pickle.dump(combined_data, f)
    print(f"Saved combined_data DataFrame as combined_data.pkl")

with open("merged_data.pkl", "wb") as f:
    pickle.dump(merged_data, f)
    print(f"Saved merged_data DataFrame as merged_data.pkl")

# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='Collect and group event labels from 2 sources within broad tolerance')
#     parser.add_argument('-m', '--minesite_id', type=str, metavar='', required=True,
#                         help='Minesite ID, e.g. macmahon-telfer')
#     parser.add_argument('-s', '--shift_id', type=str, metavar='', required=True, help='Shift ID, e.g. 202110021 or 202110021-202110101')
#     parser.add_argument('-s1', '--source1', type=str, metavar='', required=True, help='Name of first data source')
#     parser.add_argument('-s2', '--source2', type=str, metavar='', required=True, help='Name of second data source')
#     parser.add_argument('-t', '--tolerance', type=int, metavar='', required=False,
#                         help='Tolerance (s) to classify event match')
#     args = parser.parse_args()
#
#     shift_list = args.shift_id.split('-')
#     if len(shift_list) == 1:
#         None
#     elif len(shift_list) == 2:
#         shift_list = Minesite.from_central_resources(args.minesite_id).get_shift_list_for_shifts(shift_list[0],
#                                                                                             shift_list[1])
#     else:
#         raise AssertionError('Invalid shift_id input')
#
#     for shift_id in shift_list:
#         shift_combined_data, shift_merged_data = main(args.minesite_id, shift_id, args.source1, source2, args.tolerance)
#         merged_data = {
#             shift_id:
#                 shift_merged_data
#         }
#         combined_data = {
#             shift_id:
#                 shift_combined_data
#         }
#
#     with open("combined_data.pkl", "wb") as f:
#         pickle.dump(combined_data, f)
#
#     with open("merged_data.pkl", "wb") as f:
#         pickle.dump(merged_data, f)
