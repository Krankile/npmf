import pickle
import random
from glob import glob

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader

from .era_dataset import EraDataset
from .utils import clamp_and_slice


class EraController:
    class Mode:
        sequential = "sequential"
        random = "random"

    def __init__(
        self,
        era_controller,
        start_date,
        end_date,
        stock_df,
        fundamental_df,
        meta_df,
        macro_df,
        conf,
        **_,
    ):

        self.mode = era_controller["mode"]
        self.params = era_controller
        self.conf = conf

        self.path_dict = None
        if "pre_proc_data_dir" in self.conf and self.conf.pre_proc_data_dir is not None:
            self.path_dict = {
                path.split("/")[-1].split(" ")[0]: path
                for path in sorted(glob(self.conf.pre_proc_data_dir + "/*"))
            }
        else:
            self.stock_df, self.fundamental_df, self.meta_df, self.macro_df = (
                stock_df,
                fundamental_df,
                meta_df,
                macro_df,
            )

        self.loader_to_na_dict = {}
        self.n_iters = 0

        if self.mode == self.Mode.sequential:
            queue_length = era_controller["queue_length"]
            self.infront_dates = pd.date_range(
                start=start_date, periods=queue_length, freq="M"
            )
            self.date = self.infront_dates[0]
            self.end_dates = pd.date_range(
                start=end_date, periods=queue_length, freq="M"
            )

            self.infront_loaders = self.dates_to_loader(self.infront_dates)
            self.end_loaders = self.dates_to_loader(self.end_dates)

            self.dates_have_overlapped = False
            self.total = len(
                pd.date_range(
                    start=self.infront_dates[0], end=self.end_dates[0], freq="M"
                )
            )
            self.past_dates = []
            self._next = self.sequential_next

        elif self.mode == self.Mode.random:
            self.max_samplings = era_controller["max_samplings"]
            self.sample_size = era_controller["sample_size"]

            metric_dates = pd.date_range(start=end_date, end="2018-12-31", freq="M")

            self.infront_dates = metric_dates[: len(metric_dates) // 2]
            self.end_dates = metric_dates[len(metric_dates) // 2 :]

            self.past_dates = list(
                pd.date_range(start=start_date, end=end_date, freq="M")
            )
            self.date = self.past_dates[-1]

            self.infront_loaders = self.dates_to_loader(self.infront_dates)
            self.end_loaders = self.dates_to_loader(self.end_dates)

            self.total = era_controller["max_samplings"]
            self._next = self.random_next

        else:
            raise ValueError("Invalid EraController mode specified")

    def register_pbar(self, pbar):
        self.pbar = pbar
        self.update_pbar_desc()

    def update_pbar_desc(self):
        if self.mode == self.Mode.sequential:
            self.pbar.set_description(
                f"Era {self.date} [{self.n_iters+1}/{self.total}]"
            )

        elif self.mode == self.Mode.random:
            self.pbar.set_description(
                f"Sampling before {self.date} [{self.n_iters+1}/{self.total}]"
            )

    def get_dataset(self, date):

        if self.path_dict is not None:
            with open(self.path_dict[str(date.date())], "rb") as f:
                dataset: EraDataset = pickle.load(f)
        else:
            dataset = EraDataset(
                current_time=date,
                stock_df=self.stock_df,
                fundamental_df=self.fundamental_df,
                meta_df=self.meta_df,
                macro_df=self.macro_df,
                **self.conf,
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

        # self.loader_to_na_dict[date] = dataset_infront.na_percentage
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

    def get_random_loader(self):
        random_dates = np.random.choice(
            self.past_dates, size=self.sample_size, replace=True
        )
        return DataLoader(
            ConcatDataset([self.get_dataset(date) for date in random_dates]),
            batch_size=self.conf.batch_size,
            shuffle=True,
        )

    def __iter__(self):
        return self

    def sequential_next(self):
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

        self.n_iters += 1
        self.update_pbar_desc()

        return dataloader_train, dataloader_val

    def random_next(self):

        if self.n_iters == self.max_samplings:
            raise StopIteration

        dataloader_train = self.get_random_loader()
        self.n_iters += 1

        self.update_pbar_desc()
        return dataloader_train, self.infront_loaders[0]

    def __next__(self):
        return self._next()
