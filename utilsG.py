import pandas as pd
import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def preprocess_ml1m(path, min_interactions=5):
    print("Loading data...")
    
    df = pd.read_csv(path, delimiter='::', engine='python', header=None,
                     names=['userId', 'movieId', 'rating', 'timestamp'])

    # Keep only positive interactions
    df = df[df['rating'] >= 4]

    # Sort each user's interactions by time
    df = df.sort_values(by=['userId', 'timestamp'])

    # Keep users with at least 5 interactions
    user_counts = df.groupby('userId').size()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df['userId'].isin(valid_users)]
    
    # Reindex users/items so ids start from 1
    usermap = {uid: i + 1 for i, uid in enumerate(df['userId'].unique())}
    itemmap = {iid: i + 1 for i, iid in enumerate(df['movieId'].unique())}
    
    df['userId'] = df['userId'].map(usermap)
    df['movieId'] = df['movieId'].map(itemmap)

    # Build one interaction sequence per user
    user_sequences = df.groupby('userId')['movieId'].apply(list).to_dict()

    return user_sequences


def split_data(user_sequences):
    user_train, user_valid, user_test = {}, {}, {}
    usernum, itemnum = 0, 0

    for user, seq in user_sequences.items():
        usernum = max(usernum, user)
        itemnum = max(itemnum, max(seq))

        if len(seq) < 3:
            continue

        # Leave-one-out split
        user_train[user] = seq[:-2]
        user_valid[user] = [seq[-2]]
        user_test[user]  = [seq[-1]]

    return user_train, user_valid, user_test, usernum, itemnum


def random_neq(l, r, s):
    # Sample a random item not in set s
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):
        # Make sure the user has enough history
        while len(user_train[uid]) <= 1:
            uid = np.random.randint(1, usernum + 1)

        # seq = input sequence
        # pos = positive next item
        # neg = sampled negative item
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        nxt = user_train[uid][-1]
        idx = maxlen - 1

        # Items already interacted by the user
        ts = set(user_train[uid])

        # Build training pairs from right to left
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum + 1, dtype=np.int32)
    counter = 0

    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)

        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1

        result_queue.put(tuple(zip(*one_batch)))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []

        # Start parallel workers for faster batch sampling
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def _sample_eval_items(target_item, rated, itemnum, num_neg=100):
    # Create candidate set: 1 positive + 100 negatives
    item_idx = [target_item]
    for _ in range(num_neg):
        t = np.random.randint(1, itemnum + 1)
        while t in rated:
            t = np.random.randint(1, itemnum + 1)
        item_idx.append(t)
    return item_idx


def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG_10 = 0.0
    RECALL_10 = 0.0
    NDCG_20 = 0.0
    RECALL_20 = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if u not in train or u not in valid or u not in test:
            continue
        if len(train[u]) < 1 or len(valid[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        # For test, use train + valid prefix to predict the test item
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        # Exclude all seen items when sampling negatives
        rated = set(train[u])
        rated.update(valid[u])
        rated.update(test[u])
        rated.add(0)

        item_idx = _sample_eval_items(test[u][0], rated, itemnum)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        # Rank of the true item among candidates
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            RECALL_10 += 1
        
        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            RECALL_20 += 1
            
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG_10 / valid_user, RECALL_10 / valid_user, \
           NDCG_20 / valid_user, RECALL_20 / valid_user


# Evaluate on validation set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG_10 = 0.0
    RECALL_10 = 0.0
    NDCG_20 = 0.0
    RECALL_20 = 0.0
    valid_user = 0.0
  
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if u not in train or u not in valid or u not in test:
            continue
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        # For validation, use only the training prefix
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        # Exclude seen items when sampling negatives
        rated = set(train[u])
        rated.update(valid[u])
        rated.update(test[u])
        rated.add(0)

        item_idx = _sample_eval_items(valid[u][0], rated, itemnum)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            RECALL_10 += 1
        
        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            RECALL_20 += 1
            
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG_10 / valid_user, RECALL_10 / valid_user, \
           NDCG_20 / valid_user, RECALL_20 / valid_user