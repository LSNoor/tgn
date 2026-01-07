import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.contrib import tenumerate

def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:

        s = next(f)
        for idx, line in tenumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = float(e[3])

            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
        return pd.DataFrame({'u': u_list,
                             'i': i_list,
                             'ts': ts_list,
                             'label': label_list,
                             'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=True):

    new_df = df.copy()

    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique())) # No overlap in user_id
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique())) # No overlap in items_id


        # Shift all items_id to have no overlap with user_id
        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i

    # Transform to 1-indexation
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    return new_df



def run(data_name, bipartite=True):
  Path("./data/").mkdir(parents=True, exist_ok=True)
  PATH = f'./data/{data_name}.csv'
  OUT_DF = f'./data/ml_{data_name}.csv'
  OUT_FEAT = f'./data/ml_{data_name}.npy'
  OUT_NODE_FEAT = f'./data/ml_{data_name}_node.npy'

  df, feat = preprocess(PATH)
  new_df = reindex(df, bipartite)

  empty = np.zeros(feat.shape[1]).reshape(1, -1)
  feat = np.vstack([empty, feat])

  max_idx = max(new_df.u.max(), new_df.i.max())
  rand_feat = np.zeros((max_idx + 1, 172))

  new_df.to_csv(OUT_DF, index=False)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

run(args.data, bipartite=args.bipartite)