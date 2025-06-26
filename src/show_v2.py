import argparse
import operator
import pickle
import pandas as pd
import numpy as np

from bokeh.io import output_file, save
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Legend, HoverTool
from bokeh.layouts import layout
from bokeh.palettes import Bokeh5

from tools.modular.presentation import (
    calculate_precision_recall,
    create_raw_shift_event_label_visualisation,
    calculate_precision_recall_aggregate,
    calculate_match,
    calculate_both_false_label,
    create_total_precision_recall_table,
    create_filtred_precision_recall_table,
    calculate_precision_recall_aggregate_average,
    create_precision_recall_heatmap_from_result_dict,
)


def main(
    combined_data,
    merged_data,
    minesite_id,
    source_info,
    comparison_matrix_list,
    aggregate_filtre_list,
):

    truth_source = list(source_info["truth"].keys())[0]

    match_data = {
        shift_id: {
            comparison_matrix: {
                equipment_id: {
                    label: {
                        analysed_source: calculate_match(
                            merged_data[shift_id][equipment_id][label],
                            analysed_source + "_" + comparison_matrix,
                            # **comparison_matrix_list[comparison_matrix],
                            comparison_matrix_list[comparison_matrix]["tolerance"],
                            comparison_matrix_list[comparison_matrix]["match_operator"],
                        )
                        for analysed_source in source_info["analysed"].keys()
                    }
                    for label in merged_data[shift_id][equipment_id].keys()
                }
                for equipment_id in merged_data[shift_id].keys()
            }
            for comparison_matrix in comparison_matrix_list.keys()
        }
        for shift_id in merged_data.keys()
    }

    print("Finished matching events")

    precision_recall_data = {
        shift_id: {
            comparison_matrix: {
                equipment_id: {
                    label: {
                        analysed_source: calculate_precision_recall(
                            merged_data[shift_id][equipment_id][label],
                            match_data[shift_id][comparison_matrix][equipment_id][
                                label
                            ][analysed_source],
                            source_info["truth"][truth_source],
                            source_info["analysed"][analysed_source],
                        )
                        for analysed_source in source_info["analysed"].keys()
                    }
                    for label in merged_data[shift_id][equipment_id].keys()
                }
                for equipment_id in merged_data[shift_id].keys()
            }
            for comparison_matrix in comparison_matrix_list.keys()
        }
        for shift_id in merged_data.keys()
    }

    print("Finished part 1/2 of precision and recall calculation")

    precision_recall_data = {
        shift_id: {
            comparison_matrix: {
                equipment_id: {
                    label: {
                        analysed_source: calculate_both_false_label(
                            precision_recall_data[shift_id][comparison_matrix][
                                equipment_id
                            ][label][analysed_source],
                            merged_data[shift_id][equipment_id][label],
                            match_data[shift_id][comparison_matrix][equipment_id][
                                label
                            ],
                            source_info["truth"][truth_source],
                        )
                        for analysed_source in source_info["analysed"].keys()
                    }
                    for label in merged_data[shift_id][equipment_id].keys()
                }
                for equipment_id in merged_data[shift_id].keys()
            }
            for comparison_matrix in comparison_matrix_list.keys()
        }
        for shift_id in merged_data.keys()
    }

    print("Finished precision and recall calculation")

    with open("precision_recall_result.pkl", "wb") as f:
        pickle.dump(precision_recall_data, f)

    with open("precision_recall_result.pkl", "rb") as f:
        precision_recall_data = pickle.load(f)

    aggregate, merged_aggregate = calculate_precision_recall_aggregate(
        precision_recall_data, aggregate_filtre_list, source_info
    )

    plot_raw = []
    for shift_id in merged_data.keys():
        plot_raw.append(
            create_raw_shift_event_label_visualisation(
                combined_data[shift_id], minesite_id, shift_id
            )
        )
    output_file(filename="time_series.html")
    file_path = save(layout(plot_raw))
    print(file_path)

    plot_hm = []
    for shift_id in precision_recall_data.keys():
        plot_hm.append(
            [
                create_precision_recall_heatmap_from_result_dict(
                    precision_recall_data[shift_id][comparison_matrix],
                    comparison_matrix,
                    shift_id,
                    "Load",
                )
                for comparison_matrix in precision_recall_data[shift_id].keys()
            ]
        )
    output_file(filename="heatmap.html")
    file_path = save(layout(plot_hm))
    print(file_path)

    plot_total_table = create_total_precision_recall_table(merged_aggregate)
    output_file(filename="complete_aggregate.html")
    file_path = save(layout(plot_total_table))
    print(file_path)

    plot_filtred_table = create_filtred_precision_recall_table(
        merged_aggregate, aggregate_filtre_list
    )
    output_file(filename="filtred_aggregate.html")
    file_path = save(layout(plot_filtred_table))
    print(file_path)

    return precision_recall_data, aggregate, merged_aggregate


# TODO: Remove - DEBUG

combined_data_pkl_data_file_path = "combined_data.pkl"
merged_data_pkl_data_file_path = "merged_data.pkl"
minesite_id = "macmahon-telfer"

source_info = {
    "truth": {
        "label": "label_start_time"  # any valid column name with nan representing unmatch event
    },
    "analysed": {
        "proprietary": "proprietary_start_time",
        "modular": "modular_start_time",
    },
}

comparison_matrix_list = {
    "match_duration": {
        "tolerance": np.linspace(-50, 200, 26),
        "match_operator": operator.ge,
    },
    "match_percentage_wrt_truth_duration": {
        "tolerance": np.linspace(0, 1, 26),
        "match_operator": operator.ge,
    },
    "match_percentage_wrt_analysed_duration": {
        "tolerance": np.linspace(0, 1, 26),
        "match_operator": operator.ge,
    },
    "match_percentage_wrt_total_duration": {
        "tolerance": np.linspace(0, 1, 26),
        "match_operator": operator.ge,
    },
    "delta_mid_time": {
        "tolerance": np.linspace(0, 250, 26),
        "match_operator": operator.lt,
    },
}

aggregate_filtre_list = {
    "shift_id_list": [],
    "equipment_id_list": [],
    "label_list": ["Dump", "Load"],
    "source_list": [],
    "tolerance_list": {
        "match_duration": [0],
        "match_percentage_wrt_truth_duration": [0.04, 0.2, 0.4],
        "match_percentage_wrt_analysed_duration": [0.04, 0.2, 0.4],
        "match_percentage_wrt_total_duration": [0.04, 0.2, 0.4],
        "delta_mid_time": [250],
    },
}

with open(combined_data_pkl_data_file_path, "rb") as f:
    combined_data = pickle.load(f)

with open(merged_data_pkl_data_file_path, "rb") as f:
    merged_data = pickle.load(f)

precision_recall_data, aggregate, merged_aggregate = main(
    combined_data,
    merged_data,
    minesite_id,
    source_info,
    comparison_matrix_list,
    aggregate_filtre_list,
)

# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='Calculate precision and recall result')
#     parser.add_argument('-cpkl', '--combined_data_pkl_data_file_path', type=str, metavar='', required=True,
#                         help='File path to .pkl combined data')
#     parser.add_argument('-mpkl', '--merged_data_pkl_data_file_path', type=str, metavar='', required=True,
#                         help='File path to .pkl merged data')
#     parser.add_argument('-m', '--minesite_id', type=str, metavar='', required=True,
#                         help='Minesite ID, e.g. macmahon-telfer')
#     parser.add_argument('-ts', '--any_truth_source_col', type=str, metavar='', required=True,
#                         help='Source name which data is taken as the truth set/baseline')
#     parser.add_argument('-as', '--any_analysed_source_col', type=str, metavar='', required=True,
#                         help='Source name which data is being compared against the truth set')
#     args = parser.parse_args()
#
#     comparison_matrix_list = {
#         'match_duration': {
#             'tolerance': np.linspace(-100, 400, 101),
#             'match_operator': operator.ge
#         },
#         'match_percentage_wrt_proprietary': {
#             'tolerance': np.linspace(0, 1, 101),
#             'match_operator': operator.ge
#         },
#         'match_percentage_wrt_modular': {
#             'tolerance': np.linspace(0, 1, 101),
#             'match_operator': operator.ge
#         },
#         'match_percentage_wrt_total': {
#             'tolerance': np.linspace(0, 1, 101),
#             'match_operator': operator.ge
#         },
#         'delta_mid_time': {
#             'tolerance': np.linspace(0, 300, 101),
#             'match_operator': operator.lt
#         }
#     }
#
# with open(args.combined_data_pkl_data_file_path, "rb") as f:
#     combined_data = pickle.load(f)
#
# with open(args.merged_data_pkl_data_file_path, "rb") as f:
#     merged_data = pickle.load(f)
#
#     main(combined_data, merged_data, args.minesite_id, args.any_truth_source_col, args.any_analysed_source_col, comparison_matrix_list)
