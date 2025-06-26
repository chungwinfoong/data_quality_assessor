import pandas as pd
import argparse
import pickle

from ressys_dp.tools import Minesite
from ressys_dp.utils.array import make_proximity_groups
from tools.modular.adapters import ADSAdapter, ModularLoadingEventsAdapter
from tools.modular.comparison import (
    merge_data_attributes_to_similar_event_index,
    calculate_abs_delta,
)


def main(minesite_id, shift_id, source1, source2, initial_tolerance=300):

    source_list = [source1, source2]
    # if not initial_tolerance:
    #     initial_tolerance = 300

    # Initialise adapters
    proprietary_adapter = ADSAdapter()
    modular_adapter = ModularLoadingEventsAdapter()
    # minestar_adapter =
    # truth_set_adapter =
    adapters = {
        "proprietary": proprietary_adapter,
        "modular": modular_adapter,
        # 'minestar': minestar_adapter,
        # 'truth_set': trust_set_adapter
    }

    # Collect data from sources
    data = {
        source: adapters[source].get_data(minesite_id, shift_id)
        for source in source_list
    }
    combined_data = pd.concat(data.values()).sort_values(by=["equipment_id", "time"])

    equipment_list = list(set(combined_data.loc[:, "equipment_id"]))
    equipment_list = (
        equipment_list if isinstance(equipment_list, list) else [equipment_list]
    )
    label_list = list(set(combined_data.loc[:, "label"]))

    # Split combined data from 2 sources into 2 level dict sorted by equipments and labels
    data = {
        equipment_id: {
            label: combined_data.loc[
                (combined_data.loc[:, "equipment_id"] == equipment_id)
                & (combined_data.loc[:, "label"] == label),
                ["source", "time"],
            ]
        }
        for equipment_id in equipment_list
        for label in label_list
    }

    # Initial match of events with high tolerance
    for equipment_id in equipment_list:
        for label in label_list:
            data[equipment_id][label]["event_id"] = make_proximity_groups(
                data[equipment_id][label].loc[:, "time"], initial_tolerance
            )

    # Merge events/label instances into similar events/df index with high tolerance
    merged_data = {
        equipment_id: {
            label: merge_data_attributes_to_similar_event_index(
                data[equipment_id][label], source_list, "time"
            )
            for label in label_list
        }
        for equipment_id in equipment_list
    }

    # Calculate delta of merged data
    for equipment_id in equipment_list:
        for label in label_list:
            merged_data[equipment_id][label] = calculate_abs_delta(
                merged_data[equipment_id][label], "time", source_list
            )

    return merged_data


# # TODO: Remove - DEBUG
# minesite_id = 'macmahon-telfer'
# shift_id = '202110021-202110031'
# source1 =  'proprietary'
# source2 = 'modular'
# tolerance = 300
#
# shift_list = shift_id.split('-')
# if len(shift_list) == 1:
#     None
# elif len(shift_list) == 2:
#     shift_list = Minesite.from_central_resources(minesite_id).get_shift_list_for_shifts(shift_list[0], shift_list[1])
# else:
#     raise AssertionError('Invalid shift_id input')
#
# merged_data = {
#     shift_id:
#         main(minesite_id, shift_id, source1, source2, tolerance)
#     for shift_id in shift_list
# }
#
# with open("merged_data.pkl", "wb") as f:
#     pickle.dump(merged_data, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Collect and group event labels from 2 sources within broad tolerance"
    )
    parser.add_argument(
        "-m",
        "--minesite_id",
        type=str,
        metavar="",
        required=True,
        help="Minesite ID, e.g. macmahon-telfer",
    )
    parser.add_argument(
        "-s",
        "--shift_id",
        type=str,
        metavar="",
        required=True,
        help="Shift ID, e.g. 202110021 or 202110021-202110101",
    )
    parser.add_argument(
        "-s1",
        "--source1",
        type=str,
        metavar="",
        required=True,
        help="Name of first data source",
    )
    parser.add_argument(
        "-s2",
        "--source2",
        type=str,
        metavar="",
        required=True,
        help="Name of second data source",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=int,
        metavar="",
        required=False,
        help="Tolerance (s) to classify event match",
    )
    args = parser.parse_args()

    shift_list = args.shift_id.split("-")
    if len(shift_list) == 1:
        None
    elif len(shift_list) == 2:
        shift_list = Minesite.from_central_resources(
            args.minesite_id
        ).get_shift_list_for_shifts(shift_list[0], shift_list[1])
    else:
        raise AssertionError("Invalid shift_id input")

    merged_data = {
        shift_id: main(
            args.minesite_id, shift_id, args.source1, args.source2, args.tolerance
        )
        for shift_id in shift_list
    }

    with open("merged_data.pkl", "wb") as f:
        pickle.dump(merged_data, f)
