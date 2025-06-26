import pandas as pd
import numpy as np

from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource,
    Legend,
    HoverTool,
    GroupFilter,
    CDSView,
    CustomJS,
    RangeSlider,
    Div,
    Select,
    SingleIntervalTicker,
    DataTable,
    TableColumn,
    HTMLTemplateFormatter,
)
from bokeh.palettes import Bokeh5, RdYlGn10, Spectral6
from bokeh.transform import factor_cmap

from ressys_dp.tools import Minesite


def calculate_match(merged_data, comparison_matrix, tolerance, match_operator):
    """Classify initially broadly matched events (using initial tolerance) as matches using comparison matrix and
    tolerance

    Input pd.DataFrame merged_data: must have columns:
        [float source1: source1 epoch time (s), float source2: source1 epoch time (s),
            float time_attribute_name: time diff between 2 matched events (s)]

        np.nan in source1 or source2 column indicates unmatched events

    @param pd.DataFrame merged_data: data with label times and sources of similar events combined into the same index
    @param str comparison_matrix: name of comparison_matrix/column to be used for calculation
    @param array[int] tolerance: array of tolerance, in seconds, classifying events from different sources
                                 as matching event
    @param function match_operator: to determine if labels are matches, i.e. greater than/less than operator
    @return pd.DataFrame: Matches events according to comparison_matrix and various tolerances
    """

    required_columns = {comparison_matrix}
    missing = required_columns.difference(set(merged_data))
    assert not missing, f"DF is missing columns: {missing}"

    match_df = pd.concat(
        [
            match_operator(merged_data[comparison_matrix], tol).rename(tol)
            for tol in tolerance
        ],
        axis=1,
    )

    return match_df


def calculate_precision_recall(
    merged_data, match_df, any_truth_source_col, any_analysed_source_col
):
    """Calculate precision and recall of matched event labels from 2 sources taking one data source as
    the truth set and the other as being analysed

    Input pd.DataFrame merged_data: must have columns:
        [float source1: source1 epoch time (s), float source2: source1 epoch time (s),
            float time_attribute_name: time diff between 2 matched events (s)]

        np.nan in source1 or source2 column indicates unmatched events

    Precision and recall of one source are the recall and precision of the other, respectively.

    Output pd.DataFrame: will have columns:
        [int index: Tolerance (s), float precision: Precision, float recall: Recall]

    @param pd.DataFrame merged_data: data with label times and sources of similar events combined into the same index
    @param pd.DataFrame matched_df: Matches events according to comparison_matrix and various tolerances
    @param str any_truth_source_col: source name which data is taken as the truth set/baseline
    @param str any_analysed_source_col: source name which data is being compared against the truth set
    @return pd.DataFrame: precision and recall between 2 data sources at varying determined array of tolerances
    """

    required_columns = {any_truth_source_col, any_analysed_source_col}
    missing = required_columns.difference(set(merged_data))
    assert not missing, f"DF is missing columns: {missing}"

    precision_recall_df = pd.DataFrame()

    precision_recall_df["true_positive"] = match_df.sum(axis=0)
    precision_recall_df["total_positives_truth"] = (
        merged_data.loc[:, any_truth_source_col].notna().sum()
    )
    precision_recall_df["total_positives_analysed"] = (
        merged_data.loc[:, any_analysed_source_col].notna().sum()
    )

    precision_recall_df["precision"] = (
        precision_recall_df["true_positive"]
        / precision_recall_df["total_positives_analysed"]
    )
    precision_recall_df["recall"] = (
        precision_recall_df["true_positive"]
        / precision_recall_df["total_positives_truth"]
    )

    precision_recall_df["false_positive"] = (
        precision_recall_df["total_positives_analysed"]
        - precision_recall_df["true_positive"]
    )
    precision_recall_df["false_negative"] = (
        precision_recall_df["total_positives_truth"]
        - precision_recall_df["true_positive"]
    )

    return precision_recall_df


def calculate_both_false_label(
    precision_recall_data, merged_data, match_df, any_truth_source_col
):

    required_columns = {any_truth_source_col}
    missing = required_columns.difference(set(merged_data))
    assert not missing, f"DF is missing columns: {missing}"

    source_list = list(match_df.keys())

    both_false_label_df = pd.DataFrame(index=match_df[source_list[0]].columns)
    both_false_label_df["both_false_positive"] = 0
    both_false_label_df["both_false_negative"] = 0

    unmatch_mask_df = ~match_df[source_list[0]] & ~match_df[source_list[1]]

    for tolerance in unmatch_mask_df.columns:
        for event_id in unmatch_mask_df.index:
            if (
                unmatch_mask_df.loc[event_id, tolerance]
                & merged_data.isna().loc[event_id, any_truth_source_col]
            ):
                both_false_label_df.loc[tolerance, "both_false_positive"] += 1
            elif (
                unmatch_mask_df.loc[event_id, tolerance]
                & merged_data.notna().loc[event_id, any_truth_source_col]
            ):
                both_false_label_df.loc[tolerance, "both_false_negative"] += 1

    both_false_label_df = both_false_label_df.astype(int)

    precision_recall_data = pd.concat(
        [precision_recall_data, both_false_label_df], axis=1
    )

    return precision_recall_data


def calculate_durations_and_match_percentage(
    merged_data, truth_source, analysed_source
):
    """Calculate percentage of duration match with respect to total duration and duration of either sources

    Negative match_duration represents duration between ending and starting of non-overlapping matched events

    Input pd.DataFrame data: must have columns:
        [float source[0]+'_start_time': start_time of first source,
            float source[0}+'_stop_time': start_time of first source,
            float source[1]+'_start_time': start_time of second source,
            float source[1}+'_stop_time': start_time of second source]

    :param pd.DataFrame merged_data: data with label times and sources of similar events combined into the same index
    :param list(str) source_list: list of data source names
    :return pd.DataFrame: merged_data with added column containing percentage match of matched events
    """

    required_columns = {
        source + suffix
        for suffix in ["_start_time", "_stop_time"]
        for source in [truth_source, analysed_source]
    }
    missing = required_columns.difference(set(merged_data))
    assert not missing, f"DF is missing columns: {missing}"

    match_mask = (
        merged_data.loc[:, truth_source + "_stop_time"].notna()
        & merged_data.loc[:, analysed_source + "_stop_time"].notna()
    )
    merged_data[truth_source + "_duration"] = (
        merged_data.loc[match_mask, truth_source + "_stop_time"]
        - merged_data.loc[match_mask, truth_source + "_start_time"]
    )
    merged_data[analysed_source + "_duration"] = (
        merged_data.loc[match_mask, analysed_source + "_stop_time"]
        - merged_data.loc[match_mask, analysed_source + "_start_time"]
    )
    merged_data[analysed_source + "_match_duration"] = merged_data.loc[
        match_mask, [truth_source + "_stop_time", analysed_source + "_stop_time"]
    ].min(axis=1) - merged_data.loc[
        match_mask, [truth_source + "_start_time", analysed_source + "_start_time"]
    ].max(
        axis=1
    )
    # merged_data.loc[merged_data.loc[:,analysed_source+'_match_duration']<0, analysed_source+'_match_duration'] = 0
    merged_data[analysed_source + "_total_duration"] = merged_data.loc[
        match_mask, [truth_source + "_stop_time", analysed_source + "_stop_time"]
    ].max(axis=1) - merged_data.loc[
        match_mask, [truth_source + "_start_time", analysed_source + "_start_time"]
    ].min(
        axis=1
    )

    merged_data[analysed_source + "_match_percentage_wrt_truth_duration"] = (
        merged_data[analysed_source + "_match_duration"]
        / merged_data[truth_source + "_duration"]
    )
    merged_data[analysed_source + "_match_percentage_wrt_analysed_duration"] = (
        merged_data[analysed_source + "_match_duration"]
        / merged_data[analysed_source + "_duration"]
    )
    merged_data[analysed_source + "_match_percentage_wrt_total_duration"] = (
        merged_data[analysed_source + "_match_duration"]
        / merged_data[analysed_source + "_total_duration"]
    )
    # merged_data.loc[merged_data.loc[:, 'matched_percentage_wrt_total'] < 0, 'matched_percentage_wrt_'] = 0

    return merged_data


def calculate_precision_recall_aggregate(
    precision_recall_data, aggregate_filtre_list, source_info
):

    aggregate = pd.DataFrame(
        columns=[
            "shift_id",
            "comparison_matrix",
            "equipment_id",
            "label",
            "source",
            "tolerance",
            "true_positive",
            "total_positives_truth",
            "total_positives_analysed",
        ]
    )

    if aggregate_filtre_list["shift_id_list"] == []:
        shift_id_iter = precision_recall_data.keys()
    else:
        shift_id_iter = (
            aggregate_filtre_list["shift_id_list"]
            if isinstance(aggregate_filtre_list["shift_id_list"], list)
            else [aggregate_filtre_list["shift_id_list"]]
        )
    for shift_id in shift_id_iter:

        if aggregate_filtre_list["tolerance_list"].keys() == []:
            comparison_matrix_iter = precision_recall_data[shift_id].keys()
        else:
            comparison_matrix_iter = (
                aggregate_filtre_list["tolerance_list"].keys()
                if isinstance(aggregate_filtre_list["tolerance_list"].keys(), list)
                else list(aggregate_filtre_list["tolerance_list"].keys())
            )
        for comparison_matrix in comparison_matrix_iter:

            if aggregate_filtre_list["equipment_id_list"] == []:
                equipment_id_iter = precision_recall_data[shift_id][
                    comparison_matrix
                ].keys()
            else:
                equipment_id_iter = (
                    aggregate_filtre_list["equipment_id_list"]
                    if isinstance(aggregate_filtre_list["equipment_id_list"], list)
                    else [aggregate_filtre_list["equipment_id_list"]]
                )
            for equipment_id in equipment_id_iter:

                if aggregate_filtre_list["label_list"] == []:
                    label_iter = precision_recall_data[shift_id][comparison_matrix][
                        equipment_id
                    ].keys()
                else:
                    label_iter = (
                        aggregate_filtre_list["label_list"]
                        if isinstance(aggregate_filtre_list["label_list"], list)
                        else [aggregate_filtre_list["label_list"]]
                    )
                for label in label_iter:

                    if aggregate_filtre_list["source_list"] == []:
                        source_iter = precision_recall_data[shift_id][
                            comparison_matrix
                        ][equipment_id][label].keys()
                    else:
                        source_iter = (
                            aggregate_filtre_list["source_list"]
                            if isinstance(aggregate_filtre_list["source_list"], list)
                            else [aggregate_filtre_list["source_list"]]
                        )
                    for source in source_iter:

                        if (
                            aggregate_filtre_list["tolerance_list"][comparison_matrix]
                            == []
                        ):
                            tolerance_iter = precision_recall_data[shift_id][
                                comparison_matrix
                            ][equipment_id][label].index
                        else:
                            tolerance_iter = (
                                aggregate_filtre_list["tolerance_list"][
                                    comparison_matrix
                                ]
                                if isinstance(
                                    aggregate_filtre_list["tolerance_list"][
                                        comparison_matrix
                                    ],
                                    list,
                                )
                                else [
                                    aggregate_filtre_list["tolerance_list"][
                                        comparison_matrix
                                    ]
                                ]
                            )
                        for tolerance in tolerance_iter:

                            if all(
                                precision_recall_data[shift_id][comparison_matrix][
                                    equipment_id
                                ][label][source]
                                .loc[:, "precision"]
                                .notna()
                            ) & all(
                                precision_recall_data[shift_id][comparison_matrix][
                                    equipment_id
                                ][label][source]
                                .loc[:, "recall"]
                                .notna()
                            ):
                                aggregate = pd.concat(
                                    [
                                        aggregate,
                                        pd.DataFrame(
                                            {
                                                "shift_id": shift_id,
                                                "comparison_matrix": comparison_matrix,
                                                "equipment_id": equipment_id,
                                                "label": label,
                                                "source": source,
                                                "tolerance": tolerance,
                                                "true_positive": precision_recall_data[
                                                    shift_id
                                                ][comparison_matrix][equipment_id][
                                                    label
                                                ][
                                                    source
                                                ].loc[
                                                    tolerance, "true_positive"
                                                ],
                                                "total_positives_truth": precision_recall_data[
                                                    shift_id
                                                ][
                                                    comparison_matrix
                                                ][
                                                    equipment_id
                                                ][
                                                    label
                                                ][
                                                    source
                                                ].loc[
                                                    tolerance, "total_positives_truth"
                                                ],
                                                "total_positives_analysed": precision_recall_data[
                                                    shift_id
                                                ][
                                                    comparison_matrix
                                                ][
                                                    equipment_id
                                                ][
                                                    label
                                                ][
                                                    source
                                                ].loc[
                                                    tolerance,
                                                    "total_positives_analysed",
                                                ],
                                                "precision": precision_recall_data[
                                                    shift_id
                                                ][comparison_matrix][equipment_id][
                                                    label
                                                ][
                                                    source
                                                ].loc[
                                                    tolerance, "precision"
                                                ],
                                                "recall": precision_recall_data[
                                                    shift_id
                                                ][comparison_matrix][equipment_id][
                                                    label
                                                ][
                                                    source
                                                ].loc[
                                                    tolerance, "recall"
                                                ],
                                                "both_false_positive": precision_recall_data[
                                                    shift_id
                                                ][
                                                    comparison_matrix
                                                ][
                                                    equipment_id
                                                ][
                                                    label
                                                ][
                                                    source
                                                ].loc[
                                                    tolerance, "both_false_positive"
                                                ],
                                                "both_false_negative": precision_recall_data[
                                                    shift_id
                                                ][
                                                    comparison_matrix
                                                ][
                                                    equipment_id
                                                ][
                                                    label
                                                ][
                                                    source
                                                ].loc[
                                                    tolerance, "both_false_negative"
                                                ],
                                            },
                                            index=[0],
                                        ),
                                    ]
                                )

    # unmatched data contributes to both false positive and false negative
    aggregate["false_negative"] = (
        aggregate["total_positives_truth"] - aggregate["true_positive"]
    )
    aggregate["false_positive"] = (
        aggregate["total_positives_analysed"] - aggregate["true_positive"]
    )
    aggregate["event_count"] = (
        aggregate.loc[:, ["true_positive", "false_positive"]].sum(axis=1).astype(int)
    )

    merged_aggregate = aggregate.pivot(
        index=["shift_id", "comparison_matrix", "equipment_id", "label", "tolerance"],
        columns=["source"],
        values=[
            "true_positive",
            "false_positive",
            "false_negative",
            "event_count",
            "precision",
            "recall",
            "both_false_positive",
            "both_false_negative",
        ],
    ).reset_index()

    merged_aggregate.columns = [
        "_".join(col).strip() for col in merged_aggregate.columns.swaplevel().values
    ]

    source_list = list(source_info["analysed"].keys())

    merged_aggregate.rename(
        columns={
            "_shift_id": "shift_id",
            "_equipment_id": "equipment_id",
            "_label": "label",
            "_comparison_matrix": "comparison_matrix",
            "_tolerance": "tolerance",
            source_list[0] + "_both_false_positive": "both_false_positive",
            source_list[0] + "_both_false_negative": "both_false_negative",
        },
        inplace=True,
    )
    merged_aggregate.drop(
        columns=[
            source_list[1] + "_both_false_positive",
            source_list[1] + "_both_false_negative",
        ],
        inplace=True,
    )

    merged_aggregate["shift_id"] = merged_aggregate["shift_id"].astype(str)

    return aggregate, merged_aggregate


def calculate_precision_recall_aggregate_average(merged_aggregate):

    column_list = set(merged_aggregate.columns)
    unwanted_columns = [
        "shift_id",
        "equipment_id",
        "label",
        "comparison_matrix",
        "tolerance",
    ]

    for col in unwanted_columns:
        if col in column_list:
            merged_aggregate.drop(columns=col, inplace=True)

    average_aggregate = merged_aggregate.sum(axis=0).to_frame().T

    precision_recall_column_dict = {}
    precision_recall_column_list = [
        b
        for a, b in zip(
            [n.endswith(("recall", "precision")) for n in average_aggregate.columns],
            average_aggregate.columns,
        )
        if a
    ]

    for col in precision_recall_column_list:
        source, data = col.split("_")
        if data == "precision":
            average_aggregate[col] = average_aggregate[source + "_true_positive"] / (
                average_aggregate[source + "_true_positive"]
                + average_aggregate[source + "_false_positive"]
            )
        if data == "recall":
            average_aggregate[col] = average_aggregate[source + "_true_positive"] / (
                average_aggregate[source + "_true_positive"]
                + average_aggregate[source + "_false_negative"]
            )

    return average_aggregate


def create_raw_shift_event_label_visualisation(combined_data, minesite_id, shift_id):

    activity_list = ["Out of Bounds", "Load", "Dump"]
    source_list = sorted(list(set(combined_data["source"])))

    combined_data["width"] = (
        combined_data.loc[:, ["start_time", "stop_time"]].diff(axis=1).abs().iloc[:, 1]
    ) / 60
    combined_data["mid_time"] = (combined_data["mid_time"]) / 60
    combined_data.reset_index(drop=True, inplace=True)
    combined_data.reset_index(drop=False, inplace=True)

    source = ColumnDataSource(combined_data)

    # Dashboard Title
    minesite = Minesite.from_central_resources(minesite_id)
    shift_item = minesite.get_shift_item_for_shift_id(shift_id)
    shift_date = shift_item.day_date.strftime("%Y-%m-%d")
    shift_type = shift_item.get_shift_in_day_details()["name"]
    div = Div(
        text=f"""
    <hr />
    <div>Minesite ID: {minesite_id}\n<div>
    <div>Shift Date: {shift_date}</div>
    <div>Shift Type: {shift_type}</div>
    """
    )

    # Dashboard Comparison with respect to Time
    equipment_list = list(set(source.data["equipment_id"]))
    equipment_list.sort()

    equipment_filtre = GroupFilter(column_name="equipment_id", group=equipment_list[0])

    r_view = CDSView(
        source=source,
        filters=[
            equipment_filtre,
            # BooleanFilter(
            #     [True if str(label) != 'nan' else False for label in list(source_raw.data[y])
        ],
    )
    r_view.filters[0].js_on_change(
        "group",
        CustomJS(args=dict(view=r_view), code="view.properties.filters.change.emit();"),
    )
    equipment_select = Select(title="Select Equipment:", options=equipment_list)
    equipment_select.js_link("value", equipment_filtre, "group")

    p = figure(
        plot_width=1800,
        plot_height=300,
        title="Raw Data Visualisation",
        x_range=(-10, source.data["mid_time"].max() + 10),
        y_range=source_list,
        tools=["xwheel_pan", "hover"],
        x_axis_label="Time (min)",
    )
    p.xaxis.ticker = SingleIntervalTicker(interval=15)

    r = p.rect(
        x="mid_time",
        y="source",
        width="width",
        height=1,
        line_color="black",
        line_width=2,
        fill_color=factor_cmap("label", palette=Spectral6, factors=activity_list),
        fill_alpha=0.8,
        source=source,
        view=r_view,
    )
    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [
        ("Mid Time", "@mid_time"),
        ("Label", "@label"),
        ("Count", "@index"),
    ]
    rs = RangeSlider(
        title="Adjust x-axis range",  # a title to display above the slider
        start=int(source.data["mid_time"].min())
        - 10,  # set the minimum value for the slider
        end=int(source.data["mid_time"].max())
        + 10,  # set the maximum value for the slider
        step=1,  # increments for the slider
        value=(p.x_range.start, p.x_range.end),  # initial values for slider
        sizing_mode="stretch_width",
    )
    rs.js_link("value", p.x_range, "start", attr_selector=0)
    rs.js_link("value", p.x_range, "end", attr_selector=1)
    # def change_xaxis_tick():
    #     p.xaxis.ticker = SingleIntervalTicker(interval=rs.value[1]-rs.value[0])
    # p.x_range.js_on_change("start", change_xaxis_tick())

    return [[[div], [equipment_select, rs], [p]]]


def create_precision_recall_heatmap_from_result_dict(
    data, comparison_matrix, shift_id, label
):
    # TODO: Update
    """

    :param dict result: dictionary with 2 levels (level 0 = equipment_id, level 1 = label) containing precision and
                        recall result df
    :return bolek.figure: Precision heatmap
    :return bokeh.figure: Recall heatmap
    """

    equipment_list = list(data.keys())
    analysed_source_list = list(data[equipment_list[0]][label].keys())

    bokeh_layout = []

    for analysed_source in analysed_source_list:
        tolerance = data[equipment_list[0]][label][analysed_source].index

        comparison_matrix_name = {
            "match_duration": "Match duration (s) to consider event as match - Greater than is match",
            "match_percentage_wrt_truth_duration": "Match percentage wrt Truth Source (matched_duration/truth_duration)"
            " to consider event as match - Greater than is match",
            "match_percentage_wrt_analysed_duration": "Match percentage wrt Analysed Source (matched_duration/analysed_duration"
            " to consider event as match - Greater than is match",
            "match_percentage_wrt_total_duration": "Match duration percentage wrt combined (matched_duration/combined_duration"
            " to consider event as match - Greater than is match",
            "delta_mid_time": "Delta between mid time of matched events (s)"
            " to consider event as match - Less than is match",
        }

        equip_vs_tol_hm = {}
        hm_df = pd.DataFrame()

        for hm_data in [
            "precision",
            "recall",
            "true_positive",
            "total_positives_analysed",
            "total_positives_truth",
        ]:
            equip_vs_tol_hm[hm_data] = pd.DataFrame(
                columns=equipment_list, index=tolerance
            )

            for equipment_id in equipment_list:
                equip_vs_tol_hm[hm_data][equipment_id] = list(
                    data[equipment_id][label][analysed_source].loc[:, hm_data]
                )
                if all(equip_vs_tol_hm[hm_data].loc[:, equipment_id] == 0):
                    equip_vs_tol_hm[hm_data].drop(columns=equipment_id, inplace=True)

            equip_vs_tol_hm[hm_data] = equip_vs_tol_hm[hm_data].transpose()

            hm_df[hm_data] = equip_vs_tol_hm[hm_data].stack()

        hm_df.fillna(0, inplace=True)

        # Heatmap colour mapper
        i = pd.IntervalIndex.from_tuples(
            [(-1, 0), (0, 50), (50, 75), (75, 90), (90, 99), (99, 100), (100, 200)]
        )
        mapper = pd.Series(["#DCDCDC"] + list(Bokeh5) + ["black"], index=i)
        # i = pd.IntervalIndex.from_tuples([(-1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50)])
        # mapper = pd.Series(RdYlGn10[::-1], index=i)
        hm_df["precision_color"] = [mapper[n * 100] for n in hm_df["precision"]]
        hm_df["recall_color"] = [mapper[n * 100] for n in hm_df["recall"]]

        hm_df = hm_df.reset_index().rename(
            columns={"level_0": "equipment_id", "level_1": "tolerance"}
        )
        hm_source = ColumnDataSource(hm_df)

        # mapper=LinearColorMapper(palette='Bokeh5', low=0,high=100)
        y_range = sorted(list(set(hm_df["equipment_id"])), reverse=True)

        p_precision_hm = figure(
            plot_width=800,
            plot_height=len(y_range) * 25 + 70,
            title=f"{analysed_source} Precision Heat Map | Shift {shift_id} | Comparison Matrix {comparison_matrix}",
            # f" (true_positive / total_positives_analysed)",
            x_range=(
                hm_df["tolerance"].min()
                - (hm_source.data["tolerance"][1] - hm_source.data["tolerance"][0]),
                hm_df["tolerance"].max()
                + hm_source.data["tolerance"][1]
                - hm_source.data["tolerance"][0],
            ),
            y_range=y_range,
            tools="hover",
            x_axis_label=comparison_matrix_name[comparison_matrix],
            y_axis_label="Equipment ID",
        )
        r_precision_hm = p_precision_hm.rect(
            x="tolerance",
            y="equipment_id",
            width=hm_source.data["tolerance"][1] - hm_source.data["tolerance"][0],
            height=1,
            line_color="white",
            line_width=2,
            fill_color="precision_color",
            source=hm_source,
        )
        # legend = Legend(
        #     items=[('test1', [r_precision_hm]), ('test2', [r_precision_hm])]
        # )
        # p_precision_hm.add_layout(legend)
        hover = p_precision_hm.select(dict(type=HoverTool))
        hover.tooltips = [
            ("Equipment ID", "@equipment_id"),
            ("Tolerance", "@tolerance"),
            ("Precision", "@precision"),
            ("Total True Positive", "@true_positive"),
            ("Total Positive in Analysed Source", "@total_positives_analysed"),
        ]

        p_recall_hm = figure(
            plot_width=800,
            plot_height=len(y_range) * 25 + 70,
            title=f"{analysed_source} Recall Heat Map | Shift {shift_id} | Comparison Matrix {comparison_matrix}",
            # f" (true_positive / total_positives_analysed)",
            x_range=(
                hm_df["tolerance"].min()
                - (hm_source.data["tolerance"][1] - hm_source.data["tolerance"][0]),
                hm_df["tolerance"].max()
                + hm_source.data["tolerance"][1]
                - hm_source.data["tolerance"][0],
            ),
            y_range=y_range,
            tools="hover",
            x_axis_label=comparison_matrix_name[comparison_matrix],
            y_axis_label="Equipment ID",
        )
        r_recall_hm = p_recall_hm.rect(
            x="tolerance",
            y="equipment_id",
            width=hm_source.data["tolerance"][1] - hm_source.data["tolerance"][0],
            height=1,
            line_color="white",
            line_width=2,
            fill_color="recall_color",
            source=hm_source,
        )
        hover = p_recall_hm.select(dict(type=HoverTool))
        hover.tooltips = [
            ("Equipment ID", "@equipment_id"),
            ("Tolerance", "@tolerance"),
            ("Recall", "@recall"),
            ("Total True Positive", "@true_positive"),
            ("Total Positive in Truth Source", "@total_positives_truth"),
        ]

        bokeh_layout.append([p_precision_hm, p_recall_hm])

    return bokeh_layout


def create_total_precision_recall_table(merged_complete_aggregate):

    source = ColumnDataSource(merged_complete_aggregate)

    bokeh_table_total = DataTable(
        columns=[
            TableColumn(field=col, title=col)
            for col in merged_complete_aggregate.columns
        ],
        source=source,
        height=800,
        width=1800,
        index_position=None,
    )

    return bokeh_table_total


def create_filtred_precision_recall_table(
    merged_complete_aggregate, aggregate_filtre_list
):

    bokeh_table_list = []
    merged_aggregate_dict = {}

    for comparison_matrix in aggregate_filtre_list["tolerance_list"].keys():
        merged_aggregate_dict[comparison_matrix] = {}
        for tolerance in aggregate_filtre_list["tolerance_list"][comparison_matrix]:
            for label in aggregate_filtre_list["label_list"]:

                div = Div(
                    text=f"""
                    <div id="line">&nbsp;</div>
                    <div>Comparison Matrix: {comparison_matrix}\n<div>
                    <div>Threshold: {tolerance}</div>
                    <div>Label: {label}\n<div>
                    """
                )

                comparison_matrix_mask = (
                    (
                        merged_complete_aggregate.loc[:, "comparison_matrix"]
                        == comparison_matrix
                    )
                    & (merged_complete_aggregate.loc[:, "tolerance"] == tolerance)
                    & (merged_complete_aggregate.loc[:, "label"] == label)
                )

                filtred_aggregate = merged_complete_aggregate.loc[
                    comparison_matrix_mask, :
                ]

                column_list = filtred_aggregate.columns

                filtred_source = ColumnDataSource(filtred_aggregate)
                percentage_column_list = [
                    b
                    for a, b in zip(
                        [n.endswith(("recall", "precision")) for n in column_list],
                        column_list,
                    )
                    if a
                ]
                remaining_column_list = list(
                    filtred_aggregate.columns.drop(
                        ["comparison_matrix", "tolerance"] + percentage_column_list
                    )
                )

                def get_html_formatter(my_col):
                    template = """
                        <div style="background:<%=
                            (function colorfromint(){
                                if(result_col < 0.7){
                                    return('#ff5c5c')}
                                else if (result_col < 0.8)
                                    {return('#ffb85c')}
                                else if (result_col < 0.9)
                                    {return('#ffde5c')}
                                else if (result_col < 0.95)
                                    {return('#c1ff5c')}
                                else if (result_col <= 1)
                                    {return('#5cff6c')}
                                }()) %>;
                            color: black">
                        <%= value %>
                        </div>
                    """.replace(
                        "result_col", my_col
                    )

                    return HTMLTemplateFormatter(template=template)

                filtred_columns = [
                    TableColumn(field=col, title=col.replace("_", " "))
                    for col in remaining_column_list
                ] + [
                    TableColumn(
                        field=col,
                        title=col.replace("_", " "),
                        formatter=get_html_formatter(col),
                    )
                    for col in percentage_column_list
                ]

                filtred_table = DataTable(
                    columns=filtred_columns,
                    source=filtred_source,
                    index_position=None,
                    fit_columns=True,
                    height=600,
                    width=1800,
                )

                average_merged_aggregate = calculate_precision_recall_aggregate_average(
                    filtred_aggregate
                )
                average_source = ColumnDataSource(average_merged_aggregate)

                remaining_column_list = list(
                    average_merged_aggregate.columns.drop(percentage_column_list)
                )

                average_columns = [
                    TableColumn(field=col, title=col.replace("_", " "))
                    for col in remaining_column_list
                ] + [
                    TableColumn(
                        field=col,
                        title=col.replace("_", " "),
                        formatter=get_html_formatter(col),
                    )
                    for col in percentage_column_list
                ]

                average_table = DataTable(
                    columns=average_columns,
                    source=average_source,
                    index_position=None,
                    fit_columns=True,
                    height=50,
                    width=1800,
                )

                bokeh_table_list.append([div])
                bokeh_table_list.append([average_table])
                bokeh_table_list.append([filtred_table])

    return bokeh_table_list


# def dashboard(matched_data, combined_data, aggregation, result, tolerance_in_seconds):
#
#     activity_list = ['Return', 'Load', 'Haul', 'Dump', 'Transit']
#
#     # TODO: Remove - DEBUG
#     MINESITE_ID = 'macmahon-telfer'
#     SHIFT_ID = 202110021
#     minesite_id = MINESITE_ID
#     shift_id = SHIFT_ID
#
#     # Dashboard Title
#     minesite = Minesite.from_central_resources(minesite_id)
#     shift_item = minesite.get_shift_item_for_shift_id(shift_id)
#     shift_date = shift_item.day_date.strftime('%Y-%m-%d')
#     shift_type = shift_item.get_shift_in_day_details()['name']
#     div = Div(text=f"""
#     <div>Minesite ID: {minesite_id}\n<div>
#     <div>Shift Date: {shift_date}</div>
#     <div>Shift Type: {shift_type}</div>
#     """)
#
#     # Dashboard Comparison with respect to Label Instances
#     source_comparison = ColumnDataSource(matched_data)
#     x_range = ["".join(str(i)) for i in list(source_comparison.data[source_comparison.column_names[0]])]
#     y_range = [
#         b for a, b in
#         zip(
#             list(
#                 (np.array(source_comparison.column_names) == 'match_label')
#                 | (np.array(source_comparison.column_names) == 'proprietary')
#                 | (np.array(source_comparison.column_names) == 'modular')
#                 | (np.array(source_comparison.column_names) == 'proprietary')
#             ),
#             source_comparison.column_names
#         ) if a
#     ]
#
#     source_comparison.add(x_range, 'x_range')
#
#     for y in y_range:
#         source_comparison.add([y] * len(x_range), y + '_range')
#
#     p1 = figure(
#         plot_width=1800,
#         plot_height=350,
#         title="Data Comparison",
#         x_range=x_range,
#         y_range=y_range,
#         tools="xwheel_pan",
#         x_axis_label='Equipment ID/Label Instances'
#     )
#     for y in y_range:
#         r1 = p1.rect(
#             x='x_range',
#             y=y + '_range',
#             width=1,
#             height=1,
#             line_color=factor_cmap('match_label', palette=['red'] * len(activity_list), factors=activity_list,
#                                    nan_color='white'),
#             line_width=2,
#             fill_color=factor_cmap(y, palette=Spectral6, factors=activity_list, nan_color='white'),
#             source=source_comparison,
#             legend_field=y
#         )
#     # legend = Legend(items=[LegendItem(label=label, renderers=[r1], index=idx) for idx, label in enumerate(activity_list)])
#     # p1.add_layout(legend)
#
#     rs1 = RangeSlider(
#         title="Adjust x-axis range",  # a title to display above the slider
#         start=0,  # set the minimum value for the slider
#         end=575,  # set the maximum value for the slider
#         step=1,  # increments for the slider
#         value=(0, len(x_range)),
#         # value=(0, int(len(x_range)/10)),  # initial values for slider
#         sizing_mode='stretch_width'
#     )
#     rs1.js_link("value", p1.x_range, "start", attr_selector=0)
#     rs1.js_link("value", p1.x_range, "end", attr_selector=1)
#
#     # Dashboard Comparison with respect to Time
#     source_comparison.data['time'] = (
#         (source_comparison.data['time']
#          - source_comparison.data['time'].min()) / 60
#     )
#     source_comparison.data['equipment_id'], source_comparison.data['event_id'] = (
#         zip(*source_comparison.data['equipment_id_event_id'])
#     )
#
#     equipment_list = list(set(source_comparison.data['equipment_id']))
#     equipment_list.sort()
#
#     equipment_filtre = GroupFilter(
#         column_name='equipment_id',
#         group=equipment_list[0]
#     )
#
#     r2_view = [
#         CDSView(
#             source=source_comparison,
#             filters=[
#                 equipment_filtre,
#                 BooleanFilter(
#                     [True if str(label) != 'nan' else False for label in list(source_comparison.data[y])]
#                 )
#             ]
#         ) for y in y_range
#     ]
#     for view in r2_view:
#         view.filters[0].js_on_change(
#             'group',
#              CustomJS(args=dict(view=view), code="view.properties.filters.change.emit();")
#         )
#     equipment_select = Select(title="Select Equipment:", options=equipment_list)
#     equipment_select.js_link('value', equipment_filtre, 'group')
#
#     # def handle_change(attr, old, new):
#     #     return True
#     #
#     # view.filters[0].on_change('group', handle_change)
#
#     p2 = figure(
#         plot_width=1800,
#         plot_height=350,
#         title='Data Comparison',
#         x_range=(source_comparison.data['time'].min(), source_comparison.data['time'].max()),
#         y_range=y_range,
#         tools='xwheel_pan',
#         x_axis_label='Time (min)'
#     )
#     for idx, y in enumerate(y_range):
#         r2_mid_point = p2.rect(
#             x='time',
#             y=y + '_range',
#             width=2,
#             height=1,
#             line_color=factor_cmap(
#                 'match_label',
#                 palette=['red'] * len(activity_list),
#                 factors=activity_list,
#                 nan_color='white'
#             ),
#             line_width=2,
#             fill_color=factor_cmap(
#                 y,
#                 palette=Spectral6,
#                 factors=activity_list
#             ),
#             fill_alpha=0.8,
#             source=source_comparison,
#             view=r2_view[idx],
#             # legend_field=y
#         )
#         r2_tolerance_window = p2.rect(
#             x='time',
#             y=y + '_range',
#             width=tolerance_in_seconds / 60,
#             height=1,
#             line_color=factor_cmap(
#                 y,
#                 palette=['black'] * len(activity_list),
#                 factors=activity_list),
#             line_width=2,
#             fill_alpha=0,
#             source=source_comparison,
#             view=r2_view[idx],
#         )
#
#     rs2 = RangeSlider(
#         title='Adjust x-axis range',  # a title to display above the slider
#         start=int(source_comparison.data['time'].min()),  # set the minimum value for the slider
#         end=int(source_comparison.data['time'].max()),  # set the maximum value for the slider
#         step=1,  # increments for the slider
#         value=(p2.x_range.start, p2.x_range.end),  # initial values for slider
#         sizing_mode='stretch_width'
#     )
#     rs2.js_link("value", p2.x_range, "start", attr_selector=0)
#     rs2.js_link("value", p2.x_range, "end", attr_selector=1)
#
#     # Dashboard Aggregate of Mid Time Comparison
#     aggregation.reset_index(inplace=True)
#
#     if any(aggregation.columns == False):
#         aggregation = aggregation.drop(columns = False)
#     if any(aggregation.columns == True):
#         aggregation = aggregation.drop(columns = True)
#
#     columns = [TableColumn(field=col, title=col) for col in aggregation.columns]
#     data_table_labels = DataTable(
#         columns=columns,
#         source=ColumnDataSource(
#             aggregation
#         ),
#         index_position=None,
#         index_header='Label'
#     )
#
#     # Dashboard Matched Event Histogram
#     hist_sources = list(set(combined_data['source']))
#
#     delta_hist_df = [
#         pd.cut(
#             combined_data.loc[combined_data['source'].eq(source), 'delta'],
#             np.append(np.linspace(0, 500, 100), [np.inf])
#         )
#         .value_counts()
#         .sort_index()
#         .to_frame()
#         .reset_index()
#         for source in hist_sources
#     ]
#
#     for n, df in enumerate(delta_hist_df):
#         for idx in df.index:
#             delta_hist_df[n].loc[idx, ['left']] = (
#                 df
#                 .loc[idx, ['index']]
#                 .values[0]
#                 .left)
#             delta_hist_df[n].loc[idx, ['right']] = (
#                 df
#                 .loc[idx, ['index']]
#                 .values[0]
#                 .right)
#             delta_hist_df[n].loc[:,'delta_cum_sum'] = df['delta'].cumsum()
#             delta_hist_df[n].loc[df.index[-1],'right'] = df.loc[df.index[-2],'right'] + df.loc[df.index[1],'right']
#
#     source_matched = [ColumnDataSource(df) for df in delta_hist_df]
#
#     p3 = figure(
#         plot_width=800,
#         plot_height=450,
#         title="Histogram",
#         x_range=(0, source_matched[0].data['right'][-1]),
#         y_range=(0, source_matched[0].data['delta'].max()),
#         tools="",
#         x_axis_label='Time Difference (s)',
#         y_axis_label='Matched Events'
#     )
#
#     p3.quad(
#         top=source_matched[0].data['delta'],
#         bottom=0,
#         left=source_matched[0].data['left'],
#         right=source_matched[0].data['right'],
#         fill_color="blue",
#         line_color="white",
#         fill_alpha=0.4
#     )
#
#     p3.quad(
#         top=source_matched[1].data['delta'],
#         bottom=0,
#         left=source_matched[1].data['left'],
#         right=source_matched[1].data['right'],
#         fill_color="red",
#         line_color="white",
#         fill_alpha=0.4
#     )
#
#     p3.extra_y_ranges = {"cum_sum": Range1d(start=0, end=source_matched[0].data['delta_cum_sum'][-1])}
#     p3.add_layout(
#         LinearAxis(
#             y_range_name="cum_sum",
#             axis_label="Cumulative Sum of Matched Events"
#         ),
#         'right'
#     )
#
#     p3.line(
#         x=source_matched[0].data['right'],
#         y=source_matched[0].data['delta_cum_sum'],
#         y_range_name='cum_sum'
#     )
#
#     p3.line(
#         x=source_matched[1].data['right'],
#         y=source_matched[1].data['delta_cum_sum'],
#         y_range_name='cum_sum'
#     )
#
#     # Dashboard Precision and Recall Plot
#     p4 = figure(
#         plot_width=800,
#         plot_height=450,
#         title="Precision and Recall",
#         x_range=(0, result.index[-1]),
#         y_range=(0, 1),
#         tools="",
#         x_axis_label='Tolerance (s)',
#         y_axis_label='Percentage (%)'
#     )
#
#     p4.multi_line(
#         xs=[list(result.index), list(result.index)],
#         ys=[list(result.iloc[:,0]), list(result.iloc[:,1])],
#         line_color=['red', 'blue']
#     )
#
#     # Dashboard
#     output_file(filename="dashboard.html")
#     file_path = save(layout([
#         [div],
#         [rs1],
#         [p1],
#         [equipment_select, rs2],
#         [p2],
#         [data_table_labels],
#         [p3, p4]
#     ]))
#     print(file_path)
#
#     return None
