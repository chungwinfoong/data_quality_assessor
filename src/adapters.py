import os.path
from functools import lru_cache
from re import search
import pandas as pd
import numpy as np

# Snowflake query
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL

# MM library
from ressys_dp.tools import Minesite
from ressys_dp.utils.time import to_epoch

from ressys_dp.chunks import unwrap_chunks
from ressys_dp.tools.minesite import Minesite
from ressys_dp.tools.ldr import LDRHelper
from ressys_dp.chunk_downloader import ChunkDownloader, ChunkConfiguration


class BaseAdapter:

    _OUTPUT_COLUMNS_POINT_DATA = ["source", "time", "equipment_id", "label"]
    _OUTPUT_COLUMNS_INTERVAL_DATA = [
        "source",
        "start_time",
        "time",
        "mid_time",
        "stop_time",
        "equipment_id",
        "label",
    ]

    def get_data(self, minesite_id, shift_id):
        """Parent function of get_data defining constant output columns

        @param str minesite_id: Minesite ID as per database, e.g. 'macmahon-telfer'
        @param int shift_id: Shift ID as per database, e.g. '202110021'
        @return None
        """
        raise NotImplementedError("Child class must implement this method")

    def _remove_out_of_shift_bounds(
        self, df, minesite, shift_id, shift_interval_tolerance=60
    ):

        shift_item = minesite.get_shift_item_for_shift_id(shift_id)
        interval = [
            shift_item.get_start_time().timestamp() - shift_interval_tolerance,
            shift_item.get_end_time().timestamp() + shift_interval_tolerance,
        ]

        out_of_shift_bound = (df["stop_time"] <= interval[0]) | (
            df["start_time"] >= interval[1]
        )

        df.loc[out_of_shift_bound, "label"] = "Out of Bounds"

        return df

    def _convert_equipment_site_name_id_to_eid(self, df, minesite_id):

        ldr = LDRHelper(minesite_id)

        equipment_site_name = list(set(df.loc[:, "equipment_id"]))

        mapper = {
            name: str(
                int(
                    ldr.lookup_property(
                        "EquipmentId",
                        [df.loc[:, "start_time"].median()],
                        EquipmentSiteName=name,
                    )
                    .fillna(0)
                    .iloc[0]
                )
            )
            for name in equipment_site_name
        }

        df.loc[:, "equipment_id"] = df.loc[:, "equipment_id"].map(mapper)

        return df


# Adapter for proprietary data on snowflake
class ADSAdapter(BaseAdapter):

    def __init__(self, credentials=None):
        self._engine = self._create_engine()

    def get_data(self, minesite_id, shift_id):
        """Get Proprietary shift data from Snowflake ADS

        Output pd.DataFrame: Proprietary shift data
        | source           | time                | equipment_id      | label            |
        |------------------|---------------------|-------------------|------------------|
        | str: Data source | int: Epoch time (s) | str: Equipment ID | str: Event label |

        @param str minesite_id: Minesite ID as per database, e.g. 'macmahon-telfer'
        @param int shift_id: Shift ID as per database, e.g. '202110021'
        @return pd.DataFrame: Proprietary shift data
        """

        df = self._query_engine(minesite_id, shift_id)
        df = self._format_data(df, minesite_id, shift_id)
        df = self._remove_out_of_shift_bounds(
            df, Minesite.from_central_resources(minesite_id), shift_id
        )
        df = self._convert_equipment_site_name_id_to_eid(df, minesite_id)

        return df.reset_index(drop=True)

    def _create_engine(self):
        """Create SQL query engine

        @return sqlalchemy.Engine: SQL query engine
        """
        engine = create_engine(
            URL(
                account="GM04445.ap-southeast-2",
                user="CHUNG_FOONG",
                password="cV95x54UKEB4",
                database="proprietary_data",
                schema="bi_reporting",
                warehouse="analytics_wh",
                role="analyst_dev",
            )
        )
        return engine

    def _query_engine(self, minesite_id, shift_id):
        """Query data from SQL server

        Output pd.DataFrame: Unformated Proprietary shift data
        | equipmentsitename        | starttime                     | stoptime                      | proprietaryactivity  |
        |--------------------------|-------------------------------|-------------------------------|------------------|
        | str: Equipment site name | str: Start time (Timestamp()) | str: Stop time (Timestamp())  | str: Event label |

        @param str minesite_id: Minesite ID as per database, e.g. 'macmahon-telfer'
        @param int shift_id: Shift ID as per database, e.g. '202110021'
        @return pd.DataFrame: Unformated Proprietary shift data
        """
        minesite = Minesite.from_central_resources(minesite_id)
        shift_item = minesite.get_shift_item_for_shift_id(shift_id)
        shift_date = shift_item.day_date.strftime("%Y-%m-%d")
        shift_type = shift_item.get_shift_in_day_details()["name"]
        query = f"""
            select EQUIPMENTSITENAME, STARTTIME, STOPTIME, PROPRIETARYACTIVITY
            from "PROPRIETARY_DATA"."BI_REPORTING"."ACTIVITY_DETAIL"
            where SHIFTDATE = '{shift_date}'
                and SHIFTTYPE = '{shift_type}'
                and MINESITE_ID = '{minesite}'
                and (PROPRIETARYACTIVITY = 'Load' or PROPRIETARYACTIVITY = 'Dump')
            order by EQUIPMENTSITENAME, STARTTIME
        """  # TODO: Remove 'Load' label filtre                and PROPRIETARYACTIVITY = 'Load'
        return pd.read_sql(query, con=self._engine)

    def _format_data(self, df, minesite_id, shift_id):
        """Format queried Proprietary data

        Input pd.DataFrame df: Unformated Proprietary shift data
        | equipmentsitename        | starttime                     | stoptime                      | proprietaryactivity  |
        |--------------------------|-------------------------------|-------------------------------|------------------|
        | str: Equipment site name | str: Start time (Timestamp()) | str: Stop time (Timestamp())  | str: Event label |

        Output pd.DataFrame: Formatted Proprietary shift data
        | source           | time                | equipment_id      | label            |
        |------------------|----------------- ---|-------------------|------------------|
        | str: Data source | int: Epoch time (s) | str: Equipment ID | str: Event label |

        @param pd.DataFrame df: Unformatted Proprietary data
        @param str minesite_id: Minesite ID as per database, e.g. 'macmahon-telfer'
        @return pd.DataFrame: Formatted Proprietary shift data
        """
        # assert any(df.columns == 'equipmentsitename')
        # assert any(df.columns == 'starttime')
        # assert any(df.columns == 'stoptime')
        # assert any(df.columns == 'proprietaryactivity')
        required_columns = {"equipmentsitename", "starttime"}
        missing = required_columns.difference(set(df))
        assert not missing, f"DF is missing columns {missing}"

        minesite = Minesite.from_central_resources(minesite_id)

        df["start_time"] = to_epoch(df["starttime"], minesite.timezone_str)
        df["stop_time"] = to_epoch(df["stoptime"], minesite.timezone_str)
        df["time"] = df[["start_time", "stop_time"]].mean(axis=1)
        df["mid_time"] = df["time"]

        df["source"] = "proprietary"

        df["equipment_id"] = df["equipmentsitename"]

        df.rename(columns={"proprietaryactivity": "label"}, inplace=True)

        return df[self._OUTPUT_COLUMNS_INTERVAL_DATA]


# # Adapter for processed modular data on AWS (point label)
# class ModularLoadingEventsAdapter(BaseAdapter):
#
#     def __init__(self):
#         self.active_api = dict()
#
#     def get_data(self, minesite_id, shift_id):
#         """ Get shift data from AWS Cloud
#
#         Output pd.DataFrame: Shift data
#         | source           | time                | equipment_id      | label            |
#         |------------------|---------------------|-------------------|------------------|
#         | str: Data source | int: Epoch time (s) | str: Equipment ID | str: Event label |
#
#         @param str minesite_id: Minesite ID as per database, e.g. 'macmahon-telfer'
#         @param int shift_id: Shift ID as per database, e.g. '202110021'
#         @return pd.DataFrame: Shift data
#         """
#         if minesite_id not in self.active_api.keys():
#             self.active_api[minesite_id] = ModularLoadingEventsAPI(minesite_id)
#
#         df = self._format_data(self.active_api[minesite_id].get_data(shift_id))
#
#         return df
#
#
#     def _format_data(self, df):
#         """ Format local CSV data
#
#         Input pd.DataFrame df: Unformated Proprietary shift data
#         | timestamp            | haul_truck_site_name      | ... |
#         |----------------------|---------------------------|-----|
#         | int: Epock time (ms) | str: Haul truck site name | ... |
#
#         Output pd.DataFrame: Formatted shift data
#         | source           | time                | equipment_id      | label            |
#         |------------------|---------------------|-------------------|------------------|
#         | str: Data source | int: Epoch time (s) | str: Equipment ID | str: Event label |
#
#         @param pd.DataFrame df: Unformatted CSV data
#         @return pd.DataFrame: Formatted shift data
#         """
#         df['source'] = 'modular'
#         df['time'] = df['timestamp'].div(1000)
#         df.rename(columns={"haul_truck_site_name": "equipment_id"},inplace=True)
#         df['label'] = "Load"  # TODO: Softcode data labels
#
#         df.drop_duplicates(subset=['time', 'equipment_id'], inplace=True)
#         return df[self._OUTPUT_COLUMNS_POINT_DATA]
#
# # API for adapter for processed modular data on AWS (point label)
# class ModularLoadingEventsAPI:
#     _CHUNK_CONFIGURATION = ChunkConfiguration('loading_events', 'fleet', None)
#
#     def __init__(self, minesite_id):
#         self._minesite = Minesite.from_central_resources(minesite_id)
#         self._chunk_downloader = self._get_chunk_downloader(self._minesite)
#
#     def get_data(self, shift_id, clip=True):
#         """ Returns modular data for shift
#
#         @param int shift_id: Shift ID
#         @param bool clip: Whether to clip data to shift interval. If False, returns all data retrieved from upload chunks
#         @return pd.DataFrame: Loading events data
#         """
#         interval = self._get_shift_interval(shift_id)
#         dfs = self._decode_tables_for_interval(interval)
#         df = pd.concat(dfs, ignore_index=True)
#         if clip:
#             mask = df['timestamp'].div(1000).between(*interval)  # note: timestamps are ms
#             df = df.loc[mask, :].reset_index(drop=True)
#
#         return df
#
#     def _decode_tables_for_interval(self, interval):
#         chunks = self._chunk_downloader.get_chunks_for_interval(self._CHUNK_CONFIGURATION, *interval)
#         chunk_details, chunk_data = zip(*chunks)
#         chunk_versions = (
#             chunk_details[0].wrapper_version,
#             chunk_details[0].meta_version,
#             chunk_details[0].payload_version
#         )
#         decoded = unwrap_chunks('loading_events', chunk_versions, chunk_data, 'no_payload')
#         return [x.payload for x in decoded]
#
#     @staticmethod
#     def _get_chunk_downloader(minesite):
#         bucket = minesite.get_s3_bucket('upload_chunks')
#         return ChunkDownloader(bucket, minesite.aws_region)
#
#     def _get_shift_interval(self, shift_id):
#         shift_item = self._minesite.get_shift_item_for_shift_id(shift_id)
#         return shift_item.get_start_time().timestamp(), shift_item.get_end_time().timestamp()


# Temporary adapter to import modular data from manual downloaded tsv file
class ModularTSVAdapter(BaseAdapter):

    def __init__(self, root_dir):
        self._root_dir = root_dir

    def get_data(self, minesite_id, shift_id):
        """Get shift data from local CSV file

        Output pd.DataFrame: Shift data
        | source           | time                | equipment_id      | label            |
        |------------------|---------------------|-------------------|------------------|
        | str: Data source | int: Epoch time (s) | str: Equipment ID | str: Event label |

        @param str minesite_id: Minesite ID as per database, e.g. 'macmahon-telfer'
        @param int shift_id: Shift ID as per database, e.g. '202110021'
        @return pd.DataFrame: Shift data
        """
        path = self._make_path(minesite_id, shift_id)
        df = pd.read_csv(path, sep="\t")
        minesite = self._get_minesite(minesite_id)
        df = self._format_data(df, minesite, shift_id)
        df = self._remove_out_of_shift_bounds(df, minesite, shift_id)
        df = self._convert_equipment_site_name_id_to_eid(df, minesite_id)

        return df.reset_index(drop=True)

    @lru_cache(maxsize=16)
    def _get_minesite(self, minesite_id):
        return Minesite.from_central_resources(minesite_id)

    def _make_path(self, minesite_id, shift_id):
        """Create path to local CSV file

        @param str minesite_id: Minesite ID as per database, e.g. 'macmahon-telfer'
        @param int shift_id: Shift ID as per database, e.g. '202110021'
        @return str: Local CSV file path
        """
        return os.path.join(self._root_dir, minesite_id, str(shift_id), "data.tsv")


class ModularTSVAdapterMMTelfer(ModularTSVAdapter):

    def _format_data(self, df, minesite, shift_id):
        """Format local CSV data

        Input pd.DataFrame df: Unformated Proprietary shift data
        | timestamp            | haul_truck_site_name      | ... |
        |----------------------|---------------------------|-----|
        | int: Epock time (ms) | str: Haul truck site name | ... |

        Output pd.DataFrame: Formatted shift data
        | source           | time                | equipment_id      | label            |
        |------------------|---------------------|-------------------|------------------|
        | str: Data source | int: Epoch time (s) | str: Equipment ID | str: Event label |

        @param pd.DataFrame df: Unformatted CSV data
        @return pd.DataFrame: Formatted shift data
        """

        df = pd.concat(
            [
                df.assign(label="Load").rename(
                    columns={
                        "LoadTimeArrive": "LocalStartTime",
                        "LoadTimeFull": "LocalEndTime",
                    }
                ),
                df.assign(label="Dump").rename(
                    columns={
                        "DumpTimeArrive": "LocalStartTime",
                        "DumpTimeEmpty": "LocalEndTime",
                    }
                ),
            ]
        )
        df.drop(index=df.index[df.loc[:, "LocalStartTime"].isna()], inplace=True)
        df["source"] = "modular"
        df["start_time"] = to_epoch(df["LocalStartTime"], minesite.timezone_str)
        df["stop_time"] = to_epoch(df["LocalEndTime"], minesite.timezone_str)
        df.drop(index=df.index[df.loc[:, "start_time"] < 0], inplace=True)
        df["time"] = df[["start_time", "stop_time"]].mean(axis=1)
        df["mid_time"] = df["time"]
        df.rename(columns={"Truck": "equipment_id"}, inplace=True)

        # shift_item = minesite.get_shift_item_for_shift_id(shift_id)
        # interval = [shift_item.get_start_time().timestamp(), shift_item.get_end_time().timestamp()]
        # out_of_shift_bound = (df['start_time'] <= interval[0]-10) | (df['stop_time'] >= interval[1]+10)
        # df.loc[out_of_shift_bound, 'label'] = 'Haul'
        # df.drop(df.index[out_of_shift_bound], inplace=True)

        df.drop_duplicates(
            subset=["start_time", "stop_time", "equipment_id"], inplace=True
        )
        return df[self._OUTPUT_COLUMNS_INTERVAL_DATA].sort_values(
            by=["equipment_id", "start_time"]
        )


class ModularTSVAdapterMMTropicana(ModularTSVAdapter):

    def _format_data(self, df, minesite, shift_id):
        """Format local CSV data

        Input pd.DataFrame df: Unformated Proprietary shift data
        | timestamp            | haul_truck_site_name      | ... |
        |----------------------|---------------------------|-----|
        | int: Epock time (ms) | str: Haul truck site name | ... |

        Output pd.DataFrame: Formatted shift data
        | source           | time                | equipment_id      | label            |
        |------------------|---------------------|-------------------|------------------|
        | str: Data source | int: Epoch time (s) | str: Equipment ID | str: Event label |

        @param pd.DataFrame df: Unformatted CSV data
        @return pd.DataFrame: Formatted shift data
        """

        df.drop(index=df.index[df.loc[:, "LoadingStartTime"].isna()], inplace=True)
        df["source"] = "modular"
        df["label"] = "Load"
        df["start_time"] = to_epoch(df["LoadingStartTime"], minesite.timezone_str)
        df["stop_time"] = to_epoch(df["LoadingEndTime"], minesite.timezone_str)
        df["time"] = df[["start_time", "stop_time"]].mean(axis=1)
        df["mid_time"] = df["time"]
        df.rename(columns={"Equipment": "equipment_id"}, inplace=True)

        df.drop_duplicates(
            subset=["start_time", "stop_time", "equipment_id"], inplace=True
        )
        return df[self._OUTPUT_COLUMNS_INTERVAL_DATA].sort_values(
            by=["equipment_id", "start_time"]
        )


class LabelCSVAdapter(BaseAdapter):

    def __init__(self, root_dir):
        self._root_dir = root_dir

    @lru_cache(maxsize=16)
    def _get_minesite(self, minesite_id):
        return Minesite.from_central_resources(minesite_id)

    def get_data(self, minesite_id, shift_id):
        """Get shift data from local CSV file

        Output pd.DataFrame: Shift data
        | source           | time                | equipment_id      | label            |
        |------------------|---------------------|-------------------|------------------|
        | str: Data source | int: Epoch time (s) | str: Equipment ID | str: Event label |

        @param str minesite_id: Minesite ID as per database, e.g. 'macmahon-telfer'
        @param int shift_id: Shift ID as per database, e.g. '202110021'
        @return pd.DataFrame: Shift data
        """
        equipment_id_list = self._get_available_equipment_id_list(minesite_id, shift_id)

        df = pd.DataFrame()
        for n in range(len(equipment_id_list)):
            path = self._make_path(minesite_id, shift_id, equipment_id_list[n])
            df = pd.concat([pd.read_csv(path), df])

        df = self._map_simplified_label(df)
        df = self._format_data(df)
        df = self._remove_out_of_shift_bounds(
            df, Minesite.from_central_resources(minesite_id), shift_id
        )
        df = self._remove_nan_label(df)
        df = self._merge_intercepting_labels(df)

        return df.reset_index(drop=True)

    def _get_available_equipment_id_list(self, minesite_id, shift_id):

        file_name_list = os.listdir(
            os.path.join(self._root_dir, minesite_id, str(shift_id))
        )

        return [
            search("-eq(.*?).csv", file_name_list[n]).group(1)
            for n in range(len(file_name_list))
        ]

    def _make_path(self, minesite_id, shift_id, equipment_id):
        """Create path to local CSV file

        @param str minesite_id: Minesite ID as per database, e.g. 'macmahon-telfer'
        @param int shift_id: Shift ID as per database, e.g. '202110021'
        @return str: Local CSV file path
        """
        return os.path.join(
            self._root_dir,
            minesite_id,
            str(shift_id),
            str(shift_id) + "-eq" + str(equipment_id) + ".csv",
        )

    def _map_simplified_label(self, df):
        label_map = {
            "Dump": [
                "BoxScrape",
                "Dump",
                "DumpPrep",
                "MidDumpWait",
                "MidDumpWaitOrIdle",
                "PostDumpWait",
                "PostDumpWaitOrIdle",
                "PreDumpWait",
                "PreDumpWaitOrIdle",
                "RunOutTip",
                "RunOutTipAndLower",
                "ShakeOut",
                "StaticTip",
                "StaticTipAndLower",
                "TrayLower",
                "TrayLowerAndIdle",
            ],
            "Idle": [
                "DumpInWaitOrIdle",
                "DumpOutWaitOrIdle",
                "LoadInWaitOrIdle",
                "LoadOutWaitOrIdle",
                "LocationWait",
                "LocationWaitOrIdle",
                "PreStartWaitOrIdle",
                "PullOverWait",
                "PullOverWaitOrIdle",
                "WaitIntersection",
                "WaitOrIdleIntersection",
                "SpottingDumpIdle",
                "SpottingLoadIdle",
            ],
            "Load": [
                "Load",
                "LoadChute",
                "LoadPass",
                "LoadShuffle",
                "LoadUnknown",
                "MidLoadIdle",
                "MidLoadWait",
                "MidLoadWaitOrIdle",
                "PostLoadWait",
                "PostLoadWaitOrIdle",
                "PreLoadWait",
                "PreLoadWaitOrIdle",
            ],
        }

        mapper = {}
        for label in label_map.keys():
            for key in label_map[label]:
                mapper[key] = label

        df.loc[:, "label"] = df.loc[:, "label"].map(mapper)

        return df

    def _format_data(self, df):
        """Format local CSV data

        Input pd.DataFrame df: Unformated Proprietary shift data
        | timestamp            | haul_truck_site_name      | ... |
        |----------------------|---------------------------|-----|
        | int: Epock time (ms) | str: Haul truck site name | ... |

        Output pd.DataFrame: Formatted shift data
        | source           | time                | equipment_id      | label            |
        |------------------|---------------------|-------------------|------------------|
        | str: Data source | int: Epoch time (s) | str: Equipment ID | str: Event label |

        @param pd.DataFrame df: Unformatted CSV data
        @return pd.DataFrame: Formatted shift data
        """
        df["source"] = "label"
        df.rename(columns={"end_time": "stop_time"}, inplace=True)
        df["time"] = df[["start_time", "stop_time"]].mean(axis=1)
        df["mid_time"] = df["time"]
        df["equipment_id"] = df["equipment_id"].astype(str)

        df.drop_duplicates(
            subset=["start_time", "stop_time", "equipment_id"], inplace=True
        )

        return df[self._OUTPUT_COLUMNS_INTERVAL_DATA].sort_values(
            by=["equipment_id", "start_time"]
        )

    def _remove_nan_label(self, df):

        return df.loc[df.loc[:, "label"].notna(), :]

    def _merge_intercepting_labels(self, df):

        df.reset_index(drop=True, inplace=True)
        df["start_stop_diff"] = (
            df.loc[:, "start_time"].shift(-1) - df.loc[:, "stop_time"]
        )
        df["start_start_diff"] = (
            df.loc[:, "start_time"].shift(-1) - df.loc[:, "start_time"]
        )
        df["stop_stop_diff"] = df.loc[:, "stop_time"].shift(-1) - df.loc[:, "stop_time"]
        for idx in df.index[:-2]:
            if (
                (
                    (df.loc[idx, "start_stop_diff"] <= 0)
                    | (df.loc[idx, "start_start_diff"] <= 0)
                    & (df.loc[idx, "stop_stop_diff"] <= 0)
                )
                & (df.loc[idx, "label"] == df.loc[idx + 1, "label"])
                & (df.loc[idx, "equipment_id"] == df.loc[idx + 1, "equipment_id"])
            ):
                df.loc[idx + 1, "start_time"] = df.loc[
                    idx : idx + 1, "start_time"
                ].min()
                df.loc[idx + 1, "stop_time"] = df.loc[idx : idx + 1, "stop_time"].max()
                df.loc[idx + 1, "time"] = df.loc[
                    idx + 1, ["start_time", "stop_time"]
                ].mean()
                df.loc[idx + 1, "mid_time"] = df.loc[idx + 1, "time"]
                df = df.drop(index=idx)
                df.loc[idx + 1, "start_stop_diff"] = (
                    df.loc[idx + 2, "start_time"] - df.loc[idx + 1, "stop_time"]
                )
                df.loc[idx + 1, "start_start_diff"] = (
                    df.loc[idx + 2, "start_time"] - df.loc[idx + 1, "start_time"]
                )
                df.loc[idx + 1, "stop_stop_diff"] = (
                    df.loc[idx + 2, "stop_time"] - df.loc[idx + 1, "stop_time"]
                )

        return df[self._OUTPUT_COLUMNS_INTERVAL_DATA].sort_values(
            by=["equipment_id", "start_time"]
        )
