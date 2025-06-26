import pickle
from collections import defaultdict
import numpy as np
import pandas as pd

from bokeh.io import output_file, save, show, curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Legend, HoverTool, Select, SingleIntervalTicker, RangeSlider, TableColumn, \
    DataTable, HTMLTemplateFormatter
from bokeh.layouts import layout
from bokeh.palettes import Bokeh5, Spectral6
from bokeh.transform import factor_cmap

with open("match_dict.pkl", "rb") as f:
    match_dict = pickle.load(f) # {(mid, sid, eid, label): DF with (sources)_ + start_time, mid_time, stop_time}
with open("precision_recall_dict.pkl", "rb") as f:
    precision_recall_dict = pickle.load(f) #   {(mid, sid, eid, label, comparison_matrix):
    # {analysed_source: DF with TP, FP, FN, precision, recall vs threshold}}
with open("merged_precision_recall_dict.pkl", "rb") as f:
    merged_precision_recall_dict = pickle.load(f) #   {(mid, sid, eid, label, comparison_matrix):
    # DF with (sources)_ + TP, FP, FN, eventcount, total_pos, total_neg, precision, recall vs threshold}

precision_recall_df = pd.concat(merged_precision_recall_dict, axis=1)

# # Normalise time attritubes and setup data structure for time series visualisatioon
# # Output: bokeh.layout bokeh_time_series
# #     [mid select widget, sid select widget, eid select widget, label select widget]
# #     [timeseries range slider]
# #     [timeseries plot]
# sources = list(list(precision_recall_dict.values())[0].keys()) + ['label']
# params = ['start_time', 'mid_time', 'stop_time', 'width']
# spec_defs = ('mid', 'sid', 'eid', 'label')
# activity_list = ['Out of Bounds', 'Load', 'Dump']
#
# def convert_match_df_unix_epoch_to_minutes(match_df):
#     offset = 0
#     for source in sources:
#         if offset > match_df.loc[0, source + '_start_time'] or source == sources[0]:
#             offset = match_df.loc[0, source + '_start_time']
#     for source in sources:
#         match_df[source + '_start_time'] = (match_df.loc[:, source + '_start_time'] - offset) / 60
#         match_df[source + '_mid_time'] = (match_df.loc[:, source + '_mid_time'] - offset) / 60
#         match_df[source + '_stop_time'] = (match_df.loc[:, source + '_stop_time'] - offset) / 60
#         match_df[source + '_width'] = (
#             match_df.loc[:, [source + '_start_time', source + '_stop_time']].diff(axis=1).abs().iloc[:, 1]
#         )
#
#     return match_df
#
# def convert_match_dict_to_ts_dict(match_dict):
#     ts_dict = defaultdict(pd.DataFrame)  # {(mid, sid, eid, label): DF with sources, start_time, mid_time, stop_time, width}
#     for key, match_df in match_dict.items():
#         match_df = convert_match_df_unix_epoch_to_minutes(match_df)
#         for source in sources:
#             temp_df = (
#                 match_df
#                 .loc[:,[source+'_'+param for param in params]]
#                 .rename(columns={source+'_'+param: param for param in params})
#                 .fillna(0)
#             )
#             temp_df.insert(0, 'source', source)
#             ts_dict[key] = pd.concat([ts_dict[key],temp_df],axis=0)
#
#     return ts_dict
#
# def create_bokeh_time_series(ts_dict):
#     ts_dict_keys = list(set(ts_dict.keys()))
#     ts_dict_values = ts_dict.values()
#
#     def update_sid_option(mid, old, new):
#         sid_options_mask = [ts_dict_keys[n][0]==mid_select.value for n in range(len(ts_dict_keys))]
#         sid_options = sorted(set([b for a, b in zip(sid_options_mask, sid_list) if a]))
#         sid_select.options = sid_options
#         sid_select.value = sid_options[0]
#         update_eid_option([],[],[])
#
#     def update_eid_option(sid, old, new):
#         eid_options_mask = sid_options_mask and [ts_dict_keys[n][1] == sid_select.value for n in range(len(ts_dict_keys))]
#         eid_options = sorted(set([b for a, b in zip(eid_options_mask, eid_list) if a]))
#         eid_select.options = eid_options
#         eid_select.value = eid_options[0]
#         update_label_option([], [], [])
#
#     def update_label_option(eid, old, new):
#         label_options_mask = eid_options_mask and [ts_dict_keys[n][2] == eid_select.value for n in range(len(ts_dict_keys))]
#         label_options = sorted(set([b for a, b in zip(label_options_mask, label_list) if a]))
#         label_select.options = label_options
#         label_select.value = label_options[0]
#         update_ts([], [], [])
#
#     def update_ts(label, old, new):
#         print("Updated timeseries")
#         ts_source.data = ColumnDataSource.from_df(ts_dict[(mid_select.value, sid_select.value, eid_select.value, label_select.value)])
#
#         p_timeseries.x_range.end = int(max(ts_source.data['stop_time']))
#         rs_timeseries.end = int(max(ts_source.data['stop_time']))
#         rs_timeseries.sizing_mode='stretch_width'
#         rs_timeseries.js_link("value", p_timeseries.x_range, "start", attr_selector=0)
#         rs_timeseries.js_link("value", p_timeseries.x_range, "end", attr_selector=1)
#
#     mid_options = sorted(set([ts_dict_keys[n][0] for n in range(len(ts_dict_keys))]))
#     mid_select = Select(title="Select Minesite ID:", options=mid_options, value=mid_options[0])
#
#     sid_options_mask = [ts_dict_keys[n][0]==mid_options[0] for n in range(len(ts_dict_keys))]
#     sid_list = [ts_dict_keys[n][1] for n in range(len(ts_dict_keys))]
#     sid_options = sorted(set([b for a, b in zip(sid_options_mask, sid_list) if a]))
#     sid_select = Select(title="Select Shift ID:", options=sid_options, value=sid_options[0])
#
#     eid_options_mask = sid_options_mask and [ts_dict_keys[n][1]==sid_options[0] for n in range(len(ts_dict_keys))]
#     eid_list = [ts_dict_keys[n][2] for n in range(len(ts_dict_keys))]
#     eid_options = sorted(set([b for a, b in zip(eid_options_mask, eid_list) if a]))
#     eid_select = Select(title="Select Equipment ID:", options=eid_options, value=eid_options[0])
#
#     label_options_mask = eid_options_mask and [ts_dict_keys[n][2]==eid_options[0] for n in range(len(ts_dict_keys))]
#     label_list = [ts_dict_keys[n][3] for n in range(len(ts_dict_keys))]
#     label_options = sorted(set([b for a, b in zip(label_options_mask, label_list) if a]))
#     label_select = Select(title="Select Label:", options=label_options, value=label_options[0])
#
#     ts_source = ColumnDataSource(ts_dict[(mid_select.value, sid_select.value, eid_select.value, label_select.value)])
#
#     mid_select.on_change('value', update_sid_option)
#     sid_select.on_change('value', update_eid_option)
#     eid_select.on_change('value', update_label_option)
#     label_select.on_change('value', update_ts)
#
#     ts_selects = [mid_select, sid_select, eid_select, label_select]
#
#     p_timeseries = figure(
#             plot_width=1800,
#             plot_height=300,
#             title='Shift event time series',
#             x_range=(0, int(max(ts_source.data['stop_time']))),
#             y_range=sources,
#             tools=['xwheel_pan', 'hover'],
#             x_axis_label='Time (min)'
#         )
#     p_timeseries.xaxis.ticker = SingleIntervalTicker(interval=15)
#
#     r_timeseries = p_timeseries.rect(
#         x='mid_time',
#         y='source',
#         width='width',
#         height=1,
#         line_color='black',
#         line_width=2,
#         fill_color='cyan',
#         # factor_cmap(
#         #     'label',
#         #     palette=Spectral6,
#         #     factors=activity_list
#         # ),
#         fill_alpha=0.8,
#         source=ts_source
#     )
#     hover = p_timeseries.select(dict(type=HoverTool))
#     hover.tooltips = [
#         ("Mid Time", "@mid_time"),
#         ("Count", "@index")
#     ]
#     rs_timeseries = RangeSlider(
#         title='Adjust x-axis range',  # a title to display above the slider
#         start=0,  # set the minimum value for the slider
#         end=int(max(ts_source.data['stop_time'])),  # set the maximum value for the slider
#         step=1,  # increments for the slider
#         value=(p_timeseries.x_range.start, p_timeseries.x_range.end),  # initial values for slider
#         sizing_mode='stretch_width'
#     )
#     rs_timeseries.js_link("value", p_timeseries.x_range, "start", attr_selector=0)
#     rs_timeseries.js_link("value", p_timeseries.x_range, "end", attr_selector=1)
#
#     return [ts_selects,[rs_timeseries],[p_timeseries]]
#
# ts_dict = convert_match_dict_to_ts_dict(match_dict)
# bokeh_time_series = create_bokeh_time_series(ts_dict)
#
# # Plot precision recall heatmap
# # Output: bokeh.layout bokeh_pr_heatmap
# #     [mid select widget, sid select widget, mtrx select widget, label select widget]
# #     [source1 precision heatmap, source1 recall heatmap]
# #     [source2 precision heatmap, soruce2 recall heatmap]
# #     ...
#
# def convert_pr_dict_to_hm_dict(pr_dict):
#     equip_vs_tol_dict = defaultdict(dict) # {(mid, sid, mtrx, label): {(source, param): df}}
#     for key, pr_dict in pr_dict.items(): # {(mid, sid, mtrx, label): {source: df}}
#         mid, sid, eid, label, mtrx = key
#         for source, pr_df in pr_dict.items(): # {source: df}
#             for param in ['precision','recall']:
#                 if not isinstance(equip_vs_tol_dict[mid, sid, mtrx, label], defaultdict):
#                     equip_vs_tol_dict[mid, sid, mtrx, label] = defaultdict(defaultdict)
#                 if not isinstance(equip_vs_tol_dict[mid, sid, mtrx, label][(source, param)], pd.DataFrame):
#                     equip_vs_tol_dict[mid, sid, mtrx, label][(source, param)] = pd.DataFrame(index=pr_df.index)
#                 equip_vs_tol_dict[mid, sid, mtrx, label][(source, param)][eid] = list(pr_df.loc[:,param])
#
#     i = pd.IntervalIndex.from_tuples([(-1, 0), (0, 50), (50, 75), (75, 90), (90, 99), (99, 100), (100, 200)])
#     mapper = pd.Series(['#DCDCDC'] + list(Bokeh5) + ['black'], index=i)
#     # i = pd.IntervalIndex.from_tuples([(-1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50)])
#     # mapper = pd.Series(RdYlGn10[::-1], index=i)
#
#     hm_dict = defaultdict(dict)
#     for key, equip_vs_tol_dfs in equip_vs_tol_dict.items():  # {(mid, sid, mtrx, label): {(source, param): df}}
#         for sub_key, equip_vs_tol_df in equip_vs_tol_dfs.items():
#             source, param = sub_key
#             equip_vs_tol_df = equip_vs_tol_df.transpose().stack().fillna(0)
#             if not isinstance(hm_dict[key], defaultdict):
#                 hm_dict[key] = defaultdict(dict)
#             if not isinstance(hm_dict[key][source], pd.DataFrame):
#                 hm_dict[key][source] = pd.DataFrame()
#             hm_dict[key][source][param] = equip_vs_tol_df
#             hm_dict[key][source][param+'_colour'] = [mapper[n * 100] for n in equip_vs_tol_df]
#
#     return hm_dict
#
# def create_bokeh_pr_heatmap(hm_dict):
#     hm_dict_keys = list(set(hm_dict.keys()))
#
#     def update_sid_option(mid, old, new):
#         sid_options_mask = [hm_dict_keys[n][0] == mid_select.value for n in range(len(hm_dict_keys))]
#         sid_options = sorted(set([b for a, b in zip(sid_options_mask, sid_list) if a]))
#         sid_select.options = sid_options
#         sid_select.value = sid_options[0]
#         update_eid_option([], [], [])
#
#     def update_mtrx_option(sid, old, new):
#         mtrx_options_mask = sid_options_mask and [hm_dict_keys[n][1] == sid_select.value for n in range(len(hm_dict_keys))]
#         mtrx_options = sorted(set([b for a, b in zip(mtrx_options_mask, mtrx_list) if a]))
#         mtrx_select.options = mtrx_options
#         mtrx_select.value = mtrx_options[0]
#         update_label_option([], [], [])
#
#     def update_label_option(eid, old, new):
#         label_options_mask = mtrx_options_mask and [hm_dict_keys[n][2] == mtrx_select.value for n in range(len(hm_dict_keys))]
#         label_options = sorted(set([b for a, b in zip(label_options_mask, label_list) if a]))
#         label_select.options = label_options
#         label_select.value = label_options[0]
#         update_pr([], [], [])
#
#     def update_pr(label, old, new):
#         print("Updated heatmap")
#         hm_dfs = hm_dict[(mid_select.value, sid_select.value, mtrx_select.value, label_select.value)]
#         for source, hm_df in hm_dfs.items():
#             hm_df = hm_df.reset_index().rename(columns={'level_0': 'eid', 'level_1': 'threshold'})
#             hm_sources[source].data = ColumnDataSource.from_df(hm_df)
#             y_range = sorted(list(set(hm_sources[source].data['eid'])), reverse=True)
#             for param in ['precision','recall']:
#                 p_pr_heatmap[(source, param)].plot_height = len(y_range)*25+70
#                 p_pr_heatmap[(source, param)].x_range.start = (
#                     hm_sources[source].data['threshold'].min() -
#                     (hm_sources[source].data['threshold'][1] - hm_sources[source].data['threshold'][0])
#                 )
#                 p_pr_heatmap[(source, param)].x_range.end = (
#                     hm_sources[source].data['threshold'].max() +
#                     (hm_sources[source].data['threshold'][1] - hm_sources[source].data['threshold'][0])
#                 )
#                 p_pr_heatmap[(source, param)].y_range.factors = y_range
#                 r_pr_heatmap[(source,param)].glyph.width = (
#                     hm_sources[source].data['threshold'][1] - hm_sources[source].data['threshold'][0]
#                 )
#     mid_options = sorted(set([hm_dict_keys[n][0] for n in range(len(hm_dict_keys))]))
#     mid_select = Select(title="Select Minesite ID:", options=mid_options, value=mid_options[0])
#
#     sid_options_mask = [hm_dict_keys[n][0] == mid_options[0] for n in range(len(hm_dict_keys))]
#     sid_list = [hm_dict_keys[n][1] for n in range(len(hm_dict_keys))]
#     sid_options = sorted(set([b for a, b in zip(sid_options_mask, sid_list) if a]))
#     sid_select = Select(title="Select Shift ID:", options=sid_options, value=sid_options[0])
#
#     mtrx_options_mask = sid_options_mask and [hm_dict_keys[n][1] == sid_options[0] for n in range(len(hm_dict_keys))]
#     mtrx_list = [hm_dict_keys[n][2] for n in range(len(hm_dict_keys))]
#     mtrx_options = sorted(set([b for a, b in zip(mtrx_options_mask, mtrx_list) if a]))
#     mtrx_select = Select(title="Select Comparison Matrix:", options=mtrx_options, value=mtrx_options[0])
#
#     label_options_mask = mtrx_options_mask and [hm_dict_keys[n][2] == mtrx_options[0] for n in range(len(hm_dict_keys))]
#     label_list = [hm_dict_keys[n][3] for n in range(len(hm_dict_keys))]
#     label_options = sorted(set([b for a, b in zip(label_options_mask, label_list) if a]))
#     label_select = Select(title="Select Label:", options=label_options, value=label_options[0])
#
#     mid_select.on_change('value', update_sid_option)
#     sid_select.on_change('value', update_mtrx_option)
#     mtrx_select.on_change('value', update_label_option)
#     label_select.on_change('value', update_pr)
#
#     hm_selects = [mid_select, sid_select, mtrx_select, label_select]
#     bokeh_pr_heatmap = [hm_selects]
#
#     hm_sources = {}
#     p_pr_heatmap = {}
#     r_pr_heatmap = {}
#
#     hm_dfs = hm_dict[(mid_select.value, sid_select.value, mtrx_select.value, label_select.value)]
#     for source, hm_df in hm_dfs.items():
#         hm_df = hm_df.reset_index().rename(columns={'level_0': 'eid', 'level_1': 'threshold'})
#         hm_sources[source] = ColumnDataSource(hm_df)
#         y_range = sorted(list(set(hm_sources[source].data['eid'])), reverse=True)
#         for param in ['precision','recall']:
#             p_pr_heatmap[(source,param)] = figure(
#                 plot_width=900,
#                 plot_height=len(y_range)*25+70,
#                 title=f"{source} {param} heat map",
#                 x_range=(hm_sources[source].data['threshold'].min() - (
#                                 hm_sources[source].data['threshold'][1]-hm_sources[source].data['threshold'][0]),
#                          hm_sources[source].data['threshold'].max() + (
#                                 hm_sources[source].data['threshold'][1] - hm_sources[source].data['threshold'][0])),
#                 y_range=y_range,
#                 tools="hover",
#                 x_axis_label='Threshold',
#                 y_axis_label='Equipment ID',
#             )
#             r_pr_heatmap[(source,param)] = p_pr_heatmap[(source,param)].rect(
#                 x='threshold',
#                 y='eid',
#                 width=hm_sources[source].data['threshold'][1] - hm_sources[source].data['threshold'][0],
#                 height=1,
#                 line_color='white',
#                 line_width=2,
#                 fill_color=param+'_colour',
#                 source=hm_sources[source],
#             )
#
#             hover = p_pr_heatmap[(source,param)].select(dict(type=HoverTool))
#             hover.tooltips = [
#                 ("Equipment ID", "@equipment_id"),
#                 ("Threshold", "@threshold"),
#                 ("Precision", "@precision"),
#                 ("Recall", "@recall"),
#                 ("Total True Positive", "@true_positive"),
#                 ("Total Positive in Truth Source", "@total_positives_truth")
#             ]
#
#         bokeh_pr_heatmap.append([p_pr_heatmap[(source,'precision')],p_pr_heatmap[(source,'recall')]])
#
#     return bokeh_pr_heatmap
#
# hm_dict = convert_pr_dict_to_hm_dict(precision_recall_dict)
# bokeh_pr_heatmap = create_bokeh_pr_heatmap(hm_dict)
#
# # Calculate and tabulate result aggregate
# # Output: bokeh.layout bokeh_agg_table
# #     [mid select widget, sid select widget, mtrx select widget, th select widget, label select widget]
# #     [result table of all eid in a shift]
#
# sources = list(list(precision_recall_dict.values())[0].keys())
#
# def convert_merged_pr_dict_to_merged_agg_dict(merged_precision_recall_dict):
#     merged_agg_dict = defaultdict(pd.DataFrame) # {(mid, sid, mtrx, th, label):
#     # DF with (source_) + TP, FP, FN, eventcount, precision, recall, bothFN, both FP
#
#     for key, merged_precision_recall_df in merged_precision_recall_dict.items():
#         for th in merged_precision_recall_df.index:
#             mid, sid, eid, label, mtrx = key
#             merged_agg_dict[(mid,sid,mtrx,th,label)] = (
#                 pd.concat(
#                     [merged_agg_dict[(mid,sid,mtrx,th,label)],
#                      pd.DataFrame(merged_precision_recall_df.loc[th,:]).transpose().rename(index={th: eid})],
#                     axis=0)
#             )
#
#     for key, merged_agg_df in merged_agg_dict.items():
#         merged_agg_dict[key] = merged_agg_df.loc[:,[source+attr for source in sources for attr in ['_true_positive','_false_positive','_false_negative','_event_count','_precision','_recall']]+['both_false_negative','both_false_positive']]
#
#     return merged_agg_dict
#
# def calculate_precision_recall_aggregate_average(merged_aggregate):
#
#     column_list = set(merged_aggregate.columns)
#     unwanted_columns = ['shift_id','equipment_id','label','comparison_matrix','tolerance']
#
#     for col in unwanted_columns:
#         if col in column_list:
#             merged_aggregate.drop(columns=col,inplace=True)
#
#     average_aggregate = merged_aggregate.sum(axis=0).to_frame().T
#
#     precision_recall_column_dict = {}
#     precision_recall_column_list = [b for a, b in zip([n.endswith(('recall', 'precision'))
#                                                        for n in average_aggregate.columns], average_aggregate.columns) if a]
#
#     for col in precision_recall_column_list:
#         source, data = col.split('_')
#         if data == 'precision':
#             average_aggregate[col] = (
#                 average_aggregate[source+'_true_positive'] /
#                 (average_aggregate[source+'_true_positive'] + average_aggregate[source+'_false_positive'])
#             )
#         if data == 'recall':
#             average_aggregate[col] = (
#                 average_aggregate[source+'_true_positive'] /
#                 (average_aggregate[source+'_true_positive'] + average_aggregate[source+'_false_negative'])
#             )
#
#     return average_aggregate
#
#
# def convert_merged_agg_dict_to_avg_merged_agg_dict(merged_agg_dict):
#     avg_merged_agg_dict = defaultdict(dict)
#     for key, merged_agg_df in merged_agg_dict.items():
#         avg_merged_agg_dict[key] = calculate_precision_recall_aggregate_average(merged_agg_df)
#
#     return avg_merged_agg_dict
#
# def create_agg_table(merged_agg_dict, avg_merged_agg_dict):
#     merged_agg_dict_keys = list(set(merged_agg_dict.keys()))
#
#     def update_sid_option(mid, old, new):
#         sid_options_mask = [merged_agg_dict_keys[n][0] == mid_select.value for n in range(len(merged_agg_dict_keys))]
#         sid_options = sorted(set([b for a, b in zip(sid_options_mask, sid_list) if a]))
#         sid_select.options = sid_options
#         sid_select.value = sid_options[0]
#         update_mtrx_option([], [], [])
#
#     def update_mtrx_option(sid, old, new):
#         mtrx_options_mask = sid_options_mask and [merged_agg_dict_keys[n][1] == sid_select.value for n in range(len(merged_agg_dict_keys))]
#         mtrx_options = sorted(set([b for a, b in zip(mtrx_options_mask, mtrx_list) if a]))
#         mtrx_select.options = mtrx_options
#         mtrx_select.value = mtrx_options[0]
#         update_th_option([], [], [])
#
#     def update_th_option(sid, old, new):
#         th_options_mask = mtrx_options_mask and [merged_agg_dict_keys[n][2] == mtrx_select.value for n in range(len(merged_agg_dict_keys))]
#         th_options = sorted(set([b for a, b in zip(th_options_mask, th_list) if a]))
#         th_select.options = th_options
#         th_select.value = th_options[0]
#         update_label_option([], [], [])
#
#     def update_label_option(eid, old, new):
#         label_options_mask = mtrx_options_mask and [merged_agg_dict_keys[n][3] == float(th_select.value) for n in range(len(merged_agg_dict_keys))]
#         label_options = sorted(set([b for a, b in zip(label_options_mask, label_list) if a]))
#         label_select.options = label_options
#         label_select.value = label_options[0]
#         update_pr([], [], [])
#
#     def update_pr(label, old, new):
#         print("Updated aggregate table")
#         table_source.data = ColumnDataSource.from_df(merged_agg_dict[(mid_select.value, sid_select.value, mtrx_select.value,  float(th_select.value), label_select.value)])
#         avg_table_source.data = ColumnDataSource.from_df(avg_merged_agg_dict[(mid_select.value, sid_select.value, mtrx_select.value,  float(th_select.value), label_select.value)])
#
#
#     mid_options = sorted(set([merged_agg_dict_keys[n][0] for n in range(len(merged_agg_dict_keys))]))
#     mid_select = Select(title="Select Minesite ID:", options=mid_options, value=mid_options[0])
#
#     sid_options_mask = [merged_agg_dict_keys[n][0] == mid_options[0] for n in range(len(merged_agg_dict_keys))]
#     sid_list = [merged_agg_dict_keys[n][1] for n in range(len(merged_agg_dict_keys))]
#     sid_options = sorted(set([b for a, b in zip(sid_options_mask, sid_list) if a]))
#     sid_select = Select(title="Select Shift ID:", options=sid_options, value=sid_options[0])
#
#     mtrx_options_mask = sid_options_mask and [merged_agg_dict_keys[n][1] == sid_options[0] for n in range(len(merged_agg_dict_keys))]
#     mtrx_list = [merged_agg_dict_keys[n][2] for n in range(len(merged_agg_dict_keys))]
#     mtrx_options = sorted(set([b for a, b in zip(mtrx_options_mask, mtrx_list) if a]))
#     mtrx_select = Select(title="Select Comparison Matrix:", options=mtrx_options, value=mtrx_options[1])
#
#     th_options_mask = mtrx_options_mask and [merged_agg_dict_keys[n][2] == mtrx_options[1] for n in range(len(merged_agg_dict_keys))]
#     th_list = [str(merged_agg_dict_keys[n][3]) for n in range(len(merged_agg_dict_keys))]
#     th_options = list(map(str,sorted(list(map(float,set([b for a, b in zip(th_options_mask, th_list) if a]))))))
#     th_select = Select(title="Select Threshold:", options=th_options, value=th_options[0])
#
#     label_options_mask = th_options_mask and [merged_agg_dict_keys[n][3] == float(th_options[0]) for n in range(len(merged_agg_dict_keys))]
#     label_list = [merged_agg_dict_keys[n][4] for n in range(len(merged_agg_dict_keys))]
#     label_options = sorted(set([b for a, b in zip(label_options_mask, label_list) if a]))
#     label_select = Select(title="Select Label:", options=label_options, value=label_options[0])
#
#     mid_select.on_change('value', update_sid_option)
#     sid_select.on_change('value', update_mtrx_option)
#     mtrx_select.on_change('value', update_th_option)
#     th_select.on_change('value', update_label_option)
#     label_select.on_change('value', update_pr)
#
#     agg_selects = [[mid_select, sid_select, mtrx_select, th_select, label_select]]
#     table_source = ColumnDataSource(merged_agg_dict[(mid_select.value, sid_select.value, mtrx_select.value, float(th_select.value), label_select.value)])
#     source_cols = list(table_source.data.keys())
#     percentage_cols = [b for a, b in zip([n.endswith(('recall','precision'))
#                                                  for n in source_cols], source_cols) if a]
#     not_percentage_cols = [b for a, b in zip([not n.endswith(('recall','precision'))
#                                                  for n in source_cols], source_cols) if a]
#
#     def get_html_formatter(my_col):
#         template = """
#             <div style="background:<%=
#                 (function colorfromint(){
#                     if(result_col < 0.7){
#                         return('#ff5c5c')}
#                     else if (result_col < 0.8)
#                         {return('#ffb85c')}
#                     else if (result_col < 0.9)
#                         {return('#ffde5c')}
#                     else if (result_col < 0.95)
#                         {return('#c1ff5c')}
#                     else if (result_col <= 1)
#                         {return('#5cff6c')}
#                     }()) %>;
#                 color: black">
#             <%= value %>
#             </div>
#         """.replace('result_col', my_col)
#
#         return HTMLTemplateFormatter(template=template)
#
#     table_columns = [
#                         TableColumn(field=col, title=col.replace('_', ' ').replace('index', 'Equipment ID').title())
#                         for col in not_percentage_cols
#                     ] + [
#                         TableColumn(field=col, title=col.replace('_', ' ').title(), formatter=get_html_formatter(col))
#                         for col in percentage_cols
#                     ]
#
#     p_table = DataTable(
#         columns=table_columns,
#         source=table_source,
#         index_position=None,
#         fit_columns=True,
#         height=600,
#         width=1800
#     )
#
#     avg_table_source = ColumnDataSource(avg_merged_agg_dict[(mid_select.value, sid_select.value, mtrx_select.value, float(th_select.value), label_select.value)])
#     p_avg_table = DataTable(
#         columns=table_columns,
#         source=avg_table_source,
#         index_position=None,
#         fit_columns=True,
#         height=50,
#         width=1800
#     )
#
#     return [[agg_selects],[p_avg_table],[p_table]]
#
# merged_agg_dict = convert_merged_pr_dict_to_merged_agg_dict(merged_precision_recall_dict)
# avg_merged_agg_dict = convert_merged_agg_dict_to_avg_merged_agg_dict(merged_agg_dict)
# bokeh_agg_table = create_agg_table(merged_agg_dict, avg_merged_agg_dict)
#
# curdoc().add_root(layout(bokeh_time_series + bokeh_pr_heatmap + bokeh_agg_table))
# curdoc().title = "Time Series and Precision Recall Heat Map"
#
# # Calculate and tabulate result aggregate
# # Output: bokeh.layout bokeh_avg_agg_table
# #     [mid select widget, sid select widget, mtrx select widget, label select widget]
# #     [averaged shift result table - across all eid and th in a shift]
#
# # avg_merged_agg_dict = convert_merged_agg_dict_to_avg_merged_agg_dict(merged_agg_dict)
# # bokeh_avg_agg_table = create_avg_agg_table(avg_merged_agg_dict)
#
# # Run 'bokeh serve --show present.py' in terminal to run
#

