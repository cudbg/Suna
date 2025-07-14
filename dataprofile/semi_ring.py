import torch
import pandas as pd
from dataclasses import dataclass
from typing import Dict
import time


@dataclass
class BaseSemiRing:
    device: str

    def concat_semi_ring(self, other: 'BaseSemiRing'):
        raise NotImplementedError("Should be implemented by subclass")

    def slice_semi_ring(self, ind: int) -> 'BaseSemiRing':
        raise NotImplementedError("Should be implemented by subclass")

    def ind_semi_ring(self, inds) -> 'BaseSemiRing':
        raise NotImplementedError("Should be implemented by subclass")

    def set_ind_zero(self, ind: int):
        raise NotImplementedError("Should be implemented by subclass")


@dataclass
class MomentSemiRing(BaseSemiRing):
    moments: Dict[int, torch.Tensor]

    def __init__(self, moments, device='cpu'):
        super().__init__(device)
        self.moments = {}
        for k, v in moments.items():
            if v.device != device:
                self.moments[k] = v.to(self.device)
            else:
                self.moments[k] = v

    def concat_semi_ring(self, msr):
        if msr.device == self.device:
            for key, val in msr.moments.items():
                self.moments[key] = torch.cat([self.moments[key], val], dim=1)
        else:
            for key, val in msr.moments.items():
                self.moments[key] = torch.cat(
                    [self.moments[key], val.to(self.device)], dim=1)

    def slice_semi_ring(self, ind):
        moments = {}
        for key, val in self.moments.items():
            moments[key] = val[:, :ind]
            self.moments[key] = val[:, ind:]
        return MomentSemiRing(moments, self.device)

    def ind_semi_ring(self, inds):
        moments = {}
        for key, val in self.moments.items():
            moments[key] = val[:, inds]
        return MomentSemiRing(moments, self.device)

    def set_ind_zero(self, ind):
        for key, val in self.moments.items():
            self.moments[key][:, ind:ind+1] = 0


def moment_semi_ring_test(df, join_key_domain, join_keys):
    index_ranges = [join_key_domain[col] for col in join_keys]
    join_key_index = pd.MultiIndex.from_product(index_ranges, names=join_keys)

    data_df = df.drop(columns=join_keys)
    ordered_columns = list(data_df.columns)

    seller_count = df.groupby(join_keys).size().to_frame('count')
    seller_moments = {0: seller_count}

    # compute the variance semi-ring
    for i in range(1, 5):
        df_m = data_df[ordered_columns]**i
        df_m[join_keys] = df[join_keys]

        seller_moments[i] = df_m.groupby(join_keys).sum()[ordered_columns]

    # indexing for join key spanning more than a single column
    for i in range(5):
        seller_moments[i].index = pd.MultiIndex.from_arrays(
            [seller_moments[i].index], names=join_keys)

    # Temporary DataFrame to facilitate 'inner' join
    temp_df = pd.DataFrame(index=join_key_index)

    for i in range(5):
        seller_moments[i] = seller_moments[i].reindex(
            join_key_index, fill_value=0)
        if i == 0:
            seller_moments[i] = torch.tensor(
                seller_moments[i][seller_moments[i].index.isin(
                    temp_df.index)].values,
                dtype=torch.int).view(-1, 1)
            seller_moments[i] = seller_moments[i].expand(
                -1, len(ordered_columns))
        else:
            seller_moments[i] = torch.tensor(
                seller_moments[i][seller_moments[i].index.isin(
                    temp_df.index)].values,
                dtype=torch.float32)
    msr = MomentSemiRing(seller_moments)
    return msr


