import random
import pickle
from glob import glob

import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader

from .utils import clamp_and_slice

from ...utils import Problem

from .era_dataset import EraDataset


class EraController:
    def __init__(
        self,
        start_date,
        end_metric_start_date,
        queue_length,
        stock_df,
        fundamental_df,
        meta_df,
        macro_df,
        conf,
    ):
        self.conf = conf
        self.stock_df, self.fundamental_df, self.meta_df, self.macro_df = (
            stock_df,
            fundamental_df,
            meta_df,
            macro_df,
        )

        self.path_dict = None
        if "pre_proc_data_dir" in self.conf and self.conf.pre_proc_data_dir is not None:
            self.path_dict = {
                path.split("/")[-1]: path
                for path in sorted(glob(self.conf.pre_proc_data_dir + "/*"))
            }

        self.dates_have_overlapped = False
        self.loader_to_na_dict = {}

        self.infront_dates = pd.date_range(
            start=start_date, periods=queue_length, freq="M"
        )
        self.end_dates = pd.date_range(
            start=end_metric_start_date, periods=queue_length, freq="M"
        )

        self.infront_loaders = self.dates_to_loader(self.infront_dates)
        self.end_loaders = self.dates_to_loader(self.end_dates)

        self.total = len(
            pd.date_range(start=self.infront_dates[0], end=self.end_dates[0], freq="M")
        )

        self.date = self.infront_dates[0]
        self.past_dates = []

    def get_dataset(self, date):
        if self.path_dict is not None:
            with open(self.path_dict[str(date)], "rb") as f:
                dataset: EraDataset = pickle.load(f)
        else:
            dataset = EraDataset(
                date,
                self.conf.training_w,
                self.conf.forecast_w,
                self.stock_df,
                self.fundamental_df,
                self.meta_df,
                self.macro_df,
                self.conf.forecast_problem,
            )

        dataset = clamp_and_slice(dataset, conf=self.conf)

        return dataset

    def date_to_loader(self, date) -> DataLoader:
        dataset_infront = self.get_dataset(date)

        loader = DataLoader(
            dataset_infront,
            batch_size=self.conf.batch_size,  # len(dataset_infront),
            shuffle=False,
            num_workers=self.conf.cpus,
        )

        self.loader_to_na_dict[date] = dataset_infront.na_percentage
        return loader

    def combine_data(self, loader, date):
        past_dataset = self.get_dataset(date)
        dataset = ConcatDataset([loader.dataset, past_dataset])
        dataloader = DataLoader(dataset, batch_size=self.conf.batch_size, shuffle=True)
        return dataloader

    def dates_to_loader(self, dates):
        dataloaders = []
        for date in dates:
            loader_infront = self.date_to_loader(date)
            dataloaders.append(loader_infront)
        return dataloaders

    def get_next_month(self, date):
        month_factor = 1 if date.day == date.days_in_month else 2
        next_month_end = date + pd.tseries.offsets.MonthEnd() * month_factor
        return next_month_end

    def validation_loaders(self):
        return self.infront_loaders, self.end_loaders

    def __iter__(self):
        return self

    def __next__(self):
        if self.infront_dates[-1] == self.end_dates[-1]:
            raise StopIteration

        self.date = self.infront_dates[0]
        dataloader_train, dataloader_val, self.date = (
            self.infront_loaders.pop(0),
            self.infront_loaders[0],
            self.infront_dates[0],
        )

        if (
            "include_past" in self.conf
            and self.conf.include_past
            and len(self.past_dates) > 0
        ):
            # Randomly sample a past date
            date = random.choice(self.past_dates)
            dataloader_train = self.combine_data(dataloader_train, date)

        self.past_dates.append(self.date)

        self.infront_dates = self.infront_dates[1:]
        next_date = self.get_next_month(self.infront_dates[-1])

        if (
            (self.infront_dates[0] <= self.end_dates[-1])
            and (next_date >= self.end_dates[0])
            and not self.dates_have_overlapped
        ):  # If overlap
            self.dates_have_overlapped = True

            self.infront_dates = self.infront_dates.append(
                pd.DatetimeIndex([self.end_dates[0]])
            )  # At first overlap, fix infront_dates to end_date start
            self.infront_loaders.append(self.end_loaders[0])

        else:
            self.infront_dates = self.infront_dates.append(
                pd.DatetimeIndex([next_date])
            )
            self.infront_loaders.append(self.date_to_loader(next_date))

        return dataloader_train, dataloader_val
