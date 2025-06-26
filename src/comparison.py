import pandas as pd
import numpy as np


def merge_data_attributes_to_similar_event_index(data, source_list, merging_column):
    """ Merge similar events from multiple sources to same df index

    Input pd.DataFrame data: must have columns:
        [str source: data source, float merging_column: numeric data column to be merged, int event_id: event id]

    Output pd.DataFrame merged_data: will have columns:
        [float source_1: source_1 merged data, float source_n: source_n merged data, ...]

    :param pd.DataFrame data: data of event labels with similar equipment id and label from combined sources
    :param list(str) source_list: list of data source names
    :param str merging_column: name of column in data to be merged
    :return: pd.DataFrame merged_data: data with label times and sources of similar events combined into the same index
    """

    required_columns = {'source', merging_column, 'event_id'}
    missing = required_columns.difference(set(data))
    assert not missing, f'DF is missing columns: {missing}'

    if not isinstance(data,pd.DataFrame):
        data = pd.DataFrame(data)

    # TODO: Implement better way to deal with repeating label - Maybe add to list to be classify as false positive
    # Remove repeating labels within tolerance by merging label instance and averaging their time
    data = data.groupby(['event_id','source']).mean().reset_index()

    merged_data = (
        pd.DataFrame(columns=source_list)
        .append(
            data
            .pivot(
                index=['event_id'],
                columns=['source'],
                values=merging_column
            )
        )
    )

    return merged_data

def merge_multiple_data_attributes_to_similar_event_index(data, source_list, merging_column_list):
    """ Merge multiple numeric data columns from multiply sources to same df event index

    Input pd.DataFrame data: must have columns:
        [str source: data source, float merging_column1: numeric data column to be merged,
            float merging_column2: numeric data column to be merged, ..., ..., int event_id: event id]

    Output pd.DataFrame merged_data: will have columns:
        [float source1+'_'+merging_column1: source1 merged data1,
            float source2+'_'+merging_column1: source2 merged data1,
            float source1+'_'+merging_column2: source1 merged data1, ..., ..., ...]

    :param pd.DataFrame data: data of event labels with similar equipment id and label from combined sources
    :param list(str) source_list: list of data source names
    :param str merging_column_list: list of names of column in data to be merged
    :return: pd.DataFrame merged_data: data with label times and sources of similar events combined into the same index
    """

    merging_column_list = [merging_column_list] if isinstance(merging_column_list, str) else merging_column_list
    required_columns = set().union(*[['source', 'event_id'],merging_column_list])
    missing = required_columns.difference(set(data))
    assert not missing, f'DF is missing columns: {missing}'

    output_columns = [x+'_'+y for x in source_list for y in merging_column_list]
    merged_data = pd.DataFrame(columns=output_columns)

    for merging_column in merging_column_list:
        merged_data = merged_data.append(
            merge_data_attributes_to_similar_event_index(data, source_list, merging_column)
            .rename(columns={
                source:
                    source+'_'+merging_column
                for source in source_list
            })
        )

    return merged_data.reset_index().groupby('index').max()

def calculate_abs_delta(merged_data, data_attribute_name, truth_source, analysed_source):
    """ Calculate delta from 2 columns of similar data attributes and append to merged_data with column name
        (delta_+'data_attribute')

    Input pd.DataFrame data: must have columns:
        [float/int column_list[0]: first set of data attribute, float/int column_list[1]: second set of data attribute]

    :param pd.DataFrame merged_data: data with label times and sources of similar events combined into the same index
    :param str data_attribute: name of data attribute
    :param list(str) column_list: list of 2 columns of similar data attributes
    :return pd.DataFrame: merged_data with added column, with name (delta_+'data_attribute') containing delta of data attribute
    """

    required_columns = {truth_source+'_'+data_attribute_name, analysed_source+'_'+data_attribute_name}
    missing = required_columns.difference(set(merged_data))
    assert not missing, f'DF is missing columns: {missing}'

    merged_data[analysed_source+'_delta_'+data_attribute_name] = (
        merged_data.loc[:,required_columns]
        .diff(axis=1)
        .abs()
        .iloc[:,1]
    )

    return merged_data

# def calculate_abs_delta(data1, data2):
#     """ Calculate delta between 2 sets of data, with type list or pd.Series
#
#     :param pd.Series/list data1: first set of data
#     :param pd.Series/list data2: second set of data
#     :return pd.Series: absolute delta between two sets of data
#     """
#
#     df = pd.DataFrame(list(zip(data1, data2)))
#
#     delta = df()
#
#     return merged_data
