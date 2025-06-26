import argparse
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
    create_recall_and_precision_heatmap_from_result_dict,
)


def main(data, truth_source, analysed_source, tolerance):

    precision_recall_data = {
        shift_id: {
            equipment_id: {
                label: calculate_precision_recall(
                    data[shift_id][equipment_id][label],
                    truth_source,
                    analysed_source,
                    "delta_time",
                    tolerance,
                    True,
                )
                for label in data[shift_id][equipment_id].keys()
            }
            for equipment_id in data[shift_id].keys()
        }
        for shift_id in data.keys()
    }

    with open("precision_recall_result.pkl", "wb") as f:
        pickle.dump(precision_recall_data, f)

    plot_hm = [
        list(
            create_recall_and_precision_heatmap_from_result_dict(
                precision_recall_data[shift_id], shift_id, "Load"
            )
        )
        for shift_id in precision_recall_data.keys()
    ]

    output_file(filename="dashboard_v2.html")
    file_path = save(layout([plot_hm]))
    print(file_path)

    return None


# # TODO: Remove - DEBUG
#
# pkl_data_file_path = 'merged_data.pkl'
# truth_source = 'proprietary'
# analysed_source = 'modular'
# tolerance_range = '10-300'
# tolerance_interval = 5
#
# with open(pkl_data_file_path, "rb") as f:
#     data = pickle.load(f)
#
# tolerance = np.arange(int(tolerance_range.split('-')[0]), int(tolerance_range.split('-')[1]),
#                       tolerance_interval)
#
# precision_recall_data = main(data, truth_source, analysed_source, tolerance)
#
# with open("precision_recall_result.pkl", "wb") as f:
#     pickle.dump(precision_recall_data, f)
#
#
# plot_hm = [list(create_recall_and_precision_heatmap_from_result_dict(precision_recall_data[shift_id], shift_id, 'Load'))
#     for shift_id in precision_recall_data.keys()]
#
# output_file(filename="dashboard_v2.html")
# file_path = save(layout([
#     plot_hm
# ]))
# print(file_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calculate precision and recall result"
    )
    parser.add_argument(
        "-pkl",
        "--pkl_data_file_path",
        type=str,
        metavar="",
        required=True,
        help="File path to .pkl data",
    )
    parser.add_argument(
        "-ts",
        "--truth_source",
        type=str,
        metavar="",
        required=True,
        help="Source name which data is taken as the truth set/baseline",
    )
    parser.add_argument(
        "-as",
        "--analysed_source",
        type=str,
        metavar="",
        required=True,
        help="Source name which data is being compared against the truth set",
    )
    parser.add_argument(
        "-tr",
        "--tolerance_range",
        type=str,
        metavar="",
        required=True,
        help="Min and max of tolerance (s) separated by (-), e.g. 10-300",
    )
    parser.add_argument(
        "-ti",
        "--tolerance_interval",
        type=int,
        metavar="",
        required=False,
        help="Interval of tolerance (s)",
    )
    args = parser.parse_args()

    with open(args.pkl_data_file_path, "rb") as f:
        data = pickle.load(f)

    tolerance = np.arange(
        int(args.tolerance_range.split("-")[0]),
        int(args.tolerance_range.split("-")[1]),
        args.tolerance_interval,
    )

    main(data, args.truth_source, args.analysed_source, tolerance)
