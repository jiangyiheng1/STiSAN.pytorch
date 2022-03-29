from batch_generater import cf_eval_quadkey
from utils import *
from torch.utils.data import DataLoader
from collections import Counter


def evaluate(model, max_len, eval_data, eval_sampler, eval_batch_size, eval_num_neg, quadkey_processor, loc2quadkey, device, num_workers):
    model.eval()
    loader = DataLoader(eval_data, batch_size=eval_batch_size, num_workers=num_workers,
                        collate_fn=lambda e: cf_eval_quadkey(e, eval_data, max_len, eval_sampler, quadkey_processor, loc2quadkey, eval_num_neg))
    cnt = Counter()
    array = np.zeros(1 + eval_num_neg)
    with torch.no_grad():
        for _, (src_locs_, src_quadkeys_, src_times_, t_mat_, g_mat_, trg_locs_, trg_quadkeys_, data_size) in enumerate(loader):
            src_loc = src_locs_.to(device)
            src_quadkey = src_quadkeys_.to(device)
            src_time = src_times_.to(device)
            t_mat = t_mat_.to(device)
            g_mat = g_mat_.to(device)
            trg_loc = trg_locs_.to(device)
            trg_quadkey = trg_quadkeys_.to(device)
            pad_mask = get_pad_mask(data_size, max_len, device)
            attn_mask = get_attn_mask(max_len, device)
            mem_mask = None
            key_pad_mask = None
            output = model(src_loc, src_quadkey, src_time, t_mat, g_mat, pad_mask, attn_mask,
                           trg_loc, trg_quadkey, key_pad_mask, mem_mask, data_size)
            idx = output.sort(descending=True, dim=1)[1]
            order = idx.topk(k=1, dim=1, largest=False)[1]
            cnt.update(order.squeeze().tolist())
    for k, v in cnt.items():
        array[k] = v
    Hit_Rate = array.cumsum()
    NDCG = 1 / np.log2(np.arange(0, eval_num_neg + 1) + 2)
    NDCG = NDCG * array
    NDCG = NDCG.cumsum() / Hit_Rate.max()
    Hit_Rate = Hit_Rate / Hit_Rate.max()

    return Hit_Rate, NDCG