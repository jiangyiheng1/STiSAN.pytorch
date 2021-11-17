import time as Time
import joblib
import numpy as np
from collections import Counter
import torch.optim
from torch.utils.data import DataLoader
from collate_fn import *
from utils import *
from neg_sampler import *
from tqdm import tqdm
from knn_poi_query import *
from utils import *
from model import *
from loss_fn import *


def evaluate(model, eval_dataset, eval_neg_sampler, eval_num_neg, eval_batch_size, max_len, region_processer, poi2quadkey, device):
    model.eval()
    reset_random_seed(42)
    cnt = Counter()
    array = np.zeros(eval_num_neg + 1)
    data_loader = DataLoader(eval_dataset,
                             sampler=LadderSampler(eval_dataset, eval_batch_size),
                             num_workers=24,
                             batch_size=eval_batch_size,
                             collate_fn=lambda e: generate_eval_batch(e, eval_dataset, eval_neg_sampler, eval_num_neg, max_len, region_processer, poi2quadkey))
    with torch.no_grad():
        for _, (poi, timestamp, quadkey, tm, gm, tgt, tgt_quadkeys, data_size) in enumerate(data_loader):
            poi = poi.to(device)
            timestamp = timestamp
            quadkey = quadkey.to(device)
            tm = torch.stack(tm)
            tm = tm.to(device=device)
            gm = torch.stack(gm).to(device)
            gm = gm.to(device=device)
            tgt_poi = tgt.to(device)
            tgt_quadkey = tgt_quadkeys.to(device)
            padding_mask = gengerate_padding_mask(max_len, data_size, device)
            attn_mask = generate_attn_mask(max_len, device)
            key_padding_mask = None
            mem_mask = None
            output = model(poi, quadkey, timestamp, tm, gm, padding_mask, attn_mask, tgt_poi, tgt_quadkey, key_padding_mask, mem_mask, data_size)
            idx = output.sort(descending=True, dim=-1)[1]
            order = idx.topk(1, dim=-1, largest=False)[1]
            cnt.update(order.squeeze().tolist())
    for k, v in cnt.items():
        array[k] = v
    # hit rate and NDCG
    hr = array.cumsum()
    ndcg = 1 / np.log2(np.arange(0, eval_num_neg + 1) + 2)
    ndcg = ndcg * array
    ndcg = ndcg.cumsum() / hr.max()
    hr = hr / hr.max()
    return hr[:10], ndcg[:10]


def train(model, train_dataset, eval_dataset, train_neg_sampler, eval_neg_sampler, train_num_neg, eval_num_neg, train_batch_size,
          eval_batch_size, max_len, num_epochs, optimizer, loss_fn, region_processer, poi2quadkey, device, result_path, model_path):
    for epoch_idx in range(num_epochs):
        start_time = Time.time()
        running_loss = 0.
        processed_batch = 0.
        data_loader = DataLoader(train_dataset,
                                 sampler=LadderSampler(train_dataset, train_batch_size),
                                 num_workers=24,
                                 batch_size=train_batch_size,
                                 collate_fn=lambda e: generate_train_batch(e, train_dataset, train_neg_sampler, train_num_neg, max_len, region_processer, poi2quadkey))
        print("=====epoch {:>2d}=====".format(epoch_idx + 1))
        batch_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)

        model.train()
        for batch_idx, (poi, timestamp, quadkey, tm, gm, tgt, tgt_quadkeys, tgt_probs, tgt_nov, data_size) in batch_iterator:
            poi = poi.to(device)
            timestamp = timestamp
            quadkey = quadkey.to(device)
            tm = torch.stack(tm)
            tm = tm.to(device=device)
            gm = torch.stack(gm).to(device)
            gm = gm.to(device=device)
            tgt_poi = tgt.to(device)
            tgt_quadkey = tgt_quadkeys.to(device)
            padding_mask = gengerate_padding_mask(max_len, data_size, device)
            attn_mask = generate_attn_mask(max_len, device)
            key_padding_mask = generate_key_padding_mask(max_len, train_num_neg, data_size, device)
            mem_mask = generate_mem_mask(max_len, train_num_neg, device)
            optimizer.zero_grad()
            output = model(poi, quadkey, timestamp, tm, gm, padding_mask, attn_mask, tgt_poi, tgt_quadkey, key_padding_mask, mem_mask, data_size)
            output = output.view(poi.size(0), -1, poi.size(1)).permute(0, 2, 1)
            pos_scores, neg_scores = output.split([1, train_num_neg], -1)
            loss = loss_fn(pos_scores, neg_scores, tgt_probs)
            keep = fix_length([torch.ones(e, dtype=torch.float32).to(device) for e in data_size], max_len=100)
            loss = torch.sum(loss * keep) / torch.sum(torch.tensor(data_size).to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            processed_batch += 1
            batch_iterator.set_postfix_str(f"loss={loss.item():.4f}")

        epoch_time = Time.time() - start_time
        print("epoch {:>2d} completed.".format(epoch_idx + 1))
        print("time taken: {:.2f} sec".format(epoch_time))
        print("avg. loss: {:.4f}".format(running_loss / processed_batch))

    print("training completed!")
    model.save(model_path)
    print("=====evaluation=====")
    hr, ndcg = evaluate(model, eval_dataset, eval_neg_sampler, eval_num_neg, eval_batch_size, max_len, region_processer, poi2quadkey, device)
    print("Hit@5: {:.4f}, NDCG@5: {:.4f}".format(hr[4], ndcg[4]))
    print("Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[9], ndcg[9]))
    f = open(result_path, 'a')
    print("Hit@5: {:.4f}, NDCG@5: {:.4f}, Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[4], ndcg[4], hr[9], ndcg[9]), file=f)
    f.close()


if __name__ == "__main__":
    data = ' '
    train_data = ' '
    eval_data = ' '

    print("Unserializing dataset...")
    dataset = unserialize(data)
    print("Finished!")
    print("Loading train dataset...")
    train_dataset = joblib.load(train_data)
    print("Loaded!")
    print("Loading eval dataset...")
    eval_dataset = joblib.load(eval_data)
    print("Loaded!")

    region_processer = dataset.QUADKEY
    n_poi = train_dataset.n_poi
    n_quadkey = len(region_processer.vocab.itos)
    geo_dim = 128
    n_geo_layer = 2
    emb_dim = 128
    n_IAAB = 4
    n_TAAD = 1
    kt = 864000.
    kg = 10000.
    max_len = 100
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dropout = 0.5

    print("Constructing Model...")
    reset_random_seed(42)
    model = STiSAN(n_poi,
                   n_quadkey,
                   geo_dim,
                   n_geo_layer,
                   emb_dim,
                   n_IAAB,
                   n_TAAD,
                   kt,
                   kg,
                   device,
                   dropout)
    model.to(device)

    loss_fn = WeightedBinaryCELoss(temperature=100.0)
    poi_query_sys = QuerySystem()
    query_tree = ' '
    poi_query_sys.load(query_tree)
    user_visited_pois = get_visited_locs(dataset)
    train_neg_sampler = KNNSampler(poi_query_sys, user_visited_pois, num_nearest=2000, exclude_visited=True)
    eval_neg_sampler = KNNSampler(poi_query_sys, user_visited_pois, num_nearest=100, exclude_visited=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))
    poi2quadkey = dataset.poi2quadkey
    train_batch_size = 32
    train_num_neg = 15
    num_epochs = 20
    eval_batch_size = 128
    eval_num_neg = 100
    model_path = ' '
    result_path = ' '

    if os.path.exists(model_path):
        model = model.load_state_dict(torch.load(model_path))
        print("=====evaluation=====")
        hr, ndcg = evaluate(model, eval_dataset, eval_neg_sampler, eval_num_neg, eval_batch_size, max_len, region_processer, poi2quadkey, device)
        print("Hit@5: {:.4f}, NDCG@5: {:.4f}".format(hr[4], ndcg[4]))
        print("Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[9], ndcg[9]))
        f = open(result_path, 'a')
        print("Hit@5: {:.4f}, NDCG@5: {:.4f}, Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[4], ndcg[4], hr[9], ndcg[9]), file=f)
        f.close()
    else:
        train(model,
              train_dataset,
              eval_dataset,
              train_neg_sampler,
              eval_neg_sampler,
              train_num_neg,
              eval_num_neg,
              train_batch_size,
              eval_batch_size,
              max_len,
              num_epochs,
              optimizer,
              loss_fn,
              region_processer,
              poi2quadkey,
              device,
              result_path,
              model_path)