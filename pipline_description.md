Data Quality Analysis Pipeline

Run / Collector
- Collect data from sources using adapters
  - **Input - Adapters**
    - Spec: minesite, shift, source (min partition specification) (what we ask adapter for)
    - Split by: equipment_id, label
  - **Output - dict data**
      - Key - equipment_id, label
      - Value - DF with start_time, time, mid_time, stop_time

- Combine data from different sources into single df
  - **Input - dict data**
  - **Output - df combined_data**
    - Data - source, start_time, time, mid_time, stop_time, eid, label
- Partition combined_data into 2 level dict sorted by eid and label
  - **Input - df combined_data**
  - **Output - dict dict_data**
    - Data - source, start_time, time, mid_time, stop_time
    - Partition - eid -> label
- Merge similar events based on broad tolerance (~300s)
  - **Input - df dict_data[equipment_id][label]**
  - Operation done using 'merge_multiple_data_attributes_to_similar_event_index' function within nested dict
  - **Output - dict merged_data**
    - Data - proprietary, modular, label - start_time, mid_time, stop_time
    - Partition - eid -> label

Put two sources on common index (we say they reference the same real-world event):
    input: dict of DFS to index
    output: DF:
        - event_id - identifies the real world event
        - dict keys: start time, mid time, end time

- Calculate comparison matrix - add to merged_data
  - **Input - df merged_data[equipment_id][label]**
  - Operation done using 'calculate_abs_delta', 'calculate_durations_and_match_percentage' functions within nested dict
  - **Output - dict merged_data**
    - Data - proprietary, modular, label - start_time, mid_time, stop_time, duration, delta, match percentages
    - Partition - eid -> label
- Above main() - whole collector process repeats for all shifts
  - **Output - dict merged_data**
  - Data - proprietary, modular, label - start_time, mid_time, stop_time, duration, delta, match percentages
  - Partition - shift -> eid -> label

Show / Comparison
- Calculate match bool df
  - **Input - df merged_data[shift_id][equipment_id][label]**
  - Operation done using 'calculate_match' function within nested dict
  - **Output - dict match_data**
    - Data - event id vs tolerance
    - Partition - shift -> comparison_matrix -> eid -> label -> source
- Calculate precision recall data
  - **Input - df merged_data[shift_id][equipment_id][label]**
  - **Input - df match_data[shift_id][comparison_matrix][equipment_id][label][analysed_source]**
  - Operation done using 'calculate_precision_recall' function within nested dict
  - **Output - dict precision_recall_data**
    - Data - precision, recall, TP, FP, FN, total positive truth, total positive analysed
    - Partition - shift -> comparison_matrix -> eid -> label -> source
- Calculate total instances of both analysed data being FN or FP
  - **Input - precision_recall_data[shift_id][comparison_matrix][equipment_id][label][analysed_source]**
  - **Input - merged_data[shift_id][equipment_id][label]**
  - **Input - match_data[shift_id][comparison_matrix][equipment_id][label]**
  - Operation done using 'calculate_both_false_label' function within nested dict
  - **Output - dict precision_recall_data**
    - Data - both_false_negative, both_false_positive
    - Partition - shift -> comparison_matrix -> eid -> label -> source
- Calculate precision recall aggregate
  - **Input - dict precision_recall_data \*parsed as nested dict_**
  - **Output - df aggregate \*_output just as reference - not used in bokeh_**
    - Data - comparison_matrix, eid, label, source, tolerance, TP, FP, ...
  - **Output - df merged_aggregate \*_merged diff source to same index_**
    - Data - comparison_matrix, eid, label, tolerance, proprietary TP, proprietary FP, mod TP, mod FP, ...

Show / Presentation
- Visualise time series
  - **Input - df combined_data[shift_id] \*_will switch to merged_data_**
  - Repeated for each shift_id could add 'select' widget for shift view
- Plot heatmap
  - **Input - precision_recall_data[shift_id][comparison_matrix]**
  - Currently limited to one label per plot
  - Repeated for each shift_id and comparison_matrix
- Plot all aggregate
  - **Input - df merged_aggregate**
- Plot aggregate filtred by selectable shift, eid, label, source, tolerance
  - **Input - df merged_aggregate**
