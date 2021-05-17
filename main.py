import time as Time
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader

import Collect_Function
from Utils import *
from Collect_Function import *
from Loss_Function import *
from Data_Prerocess import *
from Model import *
from Neg_Sampler import *


def Evaluate(model, test_dataset, test_neg_sampler, region_processer, poi2quadkey, device, test_batch_size, test_num_neg):
    model.eval()
    loader = DataLoader(test_dataset,
                        sampler=LadderSampler(test_dataset, test_batch_size),
                        batch_size=test_batch_size,
                        collate_fn=lambda e: collect_func_pe_test(e, test_dataset, test_neg_sampler, region_processer, poi2quadkey,
                                                                  k=test_num_neg))
    cnt = Counter()
    array = np.zeros(test_num_neg + 1)
    with torch.no_grad():
        for _, (poi, quadkey, time_interval_matrix, geo_interval_matrix, tgt_poi, tgt_quadkey, data_size) in enumerate(loader):
            poi = poi.to(device)
            quadkey = quadkey.to(device)
            time_interval_matrix = torch.stack(time_interval_matrix)
            time_interval_matrix = time_interval_matrix.to(device=device, dtype=torch.long)
            geo_interval_matrix = torch.stack(geo_interval_matrix).to(device)
            geo_interval_matrix = geo_interval_matrix.to(device=device, dtype=torch.long)
            tgt_poi = tgt_poi.to(device)
            tgt_quadkey = tgt_quadkey.to(device)
            src_mask = generate_square_mask(100, device)
            tgt_mask = generate_tgt_mask(1, test_num_neg, device)
            output = model(poi, quadkey, time_interval_matrix, geo_interval_matrix, tgt_poi, tgt_quadkey, src_mask, tgt_mask,
                           None, data_size)
            idx = output.sort(descending=True, dim=0)[1]
            order = idx.topk(1, dim=0, largest=False)[1]
            cnt.update(order.squeeze().tolist())
        for k, v in cnt.items():
            array[k] = v
            # hit rate and NDCG
        hr = array.cumsum()
        ndcg = 1 / np.log2(np.arange(0, test_num_neg + 1) + 2)
        ndcg = ndcg * array
        ndcg = ndcg.cumsum() / hr.max()
        hr = hr / hr.max()
        return hr[:10], ndcg[:10]


def Train(model, train_dataset, test_dataset, optimizer, loss_fn, train_neg_sampler, test_neg_sampler, region_processer,
          poi2quadkey, device, train_num_neg, train_batch_size, num_epochs, test_batch_size, test_num_neg, num_workers):
    f = open("results/loss_demo_BinaryCE.txt", 'wt')
    print("Training...")
    for epoch_idx in range(num_epochs):
        start_time = Time.time()
        running_loss = 0.
        processed_batch = 0.
        dataloader = DataLoader(train_dataset,
                                sampler=LadderSampler(train_dataset, train_batch_size),
                                num_workers=num_workers,
                                batch_size=train_batch_size,
                                collate_fn=lambda e: collect_func_pe_train(e, train_dataset, train_neg_sampler, region_processer,
                                                                           poi2quadkey, k=train_num_neg))
        batch_iterator = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
        model.train()
        print("epoch {:>2d}.".format(epoch_idx + 1))
        for batch_idx, (poi, quadkey, time_interval_matrix, geo_interval_matrix, tgt_poi, tgt_quadkey,
                        tgt_nov_, tgt_probs, data_size) in batch_iterator:
            optimizer.zero_grad()
            poi = poi.to(device)
            quadkey = quadkey.to(device)
            time_interval_matrix = torch.stack(time_interval_matrix)
            time_interval_matrix = time_interval_matrix.to(device=device, dtype=torch.long)
            geo_interval_matrix = torch.stack(geo_interval_matrix).to(device)
            geo_interval_matrix = geo_interval_matrix.to(device=device, dtype=torch.long)
            tgt_poi = tgt_poi.to(device)
            tgt_quadkey = tgt_quadkey.to(device)
            tgt_probs = tgt_probs.to(device)
            src_mask = generate_square_mask(100, device)
            tgt_mask = generate_tgt_mask(100, train_num_neg, device)
            mem_mask = generate_tgt_mask(100, train_num_neg, device)
            output = model(poi, quadkey, time_interval_matrix, geo_interval_matrix, tgt_poi, tgt_quadkey, src_mask, tgt_mask,
                           mem_mask, data_size)
            output = output.view(-1, poi.size(0), poi.size(1)).permute(2, 1, 0)
            pos_score, neg_score = output.split([1, train_num_neg], -1)
            loss = loss_fn(pos_score, neg_score, tgt_probs).to(device)
            loss = torch.sum(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            processed_batch += 1
            batch_iterator.set_postfix_str(f"loss={loss.item():.4f}")
        epoch_time = Time.time() - start_time
        print("epoch {:>2d} completed.".format(epoch_idx + 1))
        print("time taken: {:.2f} sec".format(epoch_time))
        print("avg. loss: {:.4f}".format(running_loss / processed_batch))
        print("epoch={:d}, loss={:.4f}".format(epoch_idx + 1, running_loss / processed_batch), file=f)
        print("Evaluating epoch {:>2d}...".format(epoch_idx + 1))
        hr, ndcg = Evaluate(model, test_dataset, test_neg_sampler, region_processer, poi2quadkey, device, test_batch_size,
                            test_num_neg)
        print("Hit@5: {:.4f}, NDCG@5: {:.4f}".format(hr[4], ndcg[4]))
        print("Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[9], ndcg[9]))
        
    print("Training completed!")
    f.close()
    model.save('model/model_for_demo.pt')
    print("")
    print("Evaluating...")
    hr, ndcg = Evaluate(model, test_dataset, test_neg_sampler, region_processer, poi2quadkey, device, test_batch_size,
                        test_num_neg)
    print("Hit@5: {:.4f}, NDCG@5: {:.4f}".format(hr[4], ndcg[4]))
    print("Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[9], ndcg[9]))
    with open('results/demo_results.txt', 'a') as f:
        print("Hit@5: {:.4f}, NDCG@5: {:.4f}".format(hr[4], ndcg[4]), file=f)
        print("Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[9], ndcg[9]), file=f)
        print("\n", file=f)


if __name__ == "__main__":
    data_name = 'demo'
    data_raw = 'data/demo.txt'
    data_clean = 'data/demo.data'

    if os.path.exists(data_clean):
        dataset = unserialize(data_clean)
    else:
        dataset = LBSNDataset(data_raw)
        serialize(dataset)

    # print("Loading Data...")
    # train_dataset = joblib.load('data/demo_train.data')
    # print("Train data Loaded!")
    # test_dataset = joblib.load('data/demo_test.data')
    # print("Train data Loaded!")
    train_dataset = torch.load('data/demo_train.pt')
    print("Train data Loaded!")
    test_dataset = torch.load('data/demo_test.pt')
    print("Train data Loaded!")

    emb_dim = 30
    n_geo_enc_layer = 1
    n_multi_enc_layer = 1
    n_dec_layer = 1
    len_quadkey = 12
    len_seq = 1100
    dropout = 0.5
    optimizer_name = 'Adam'
    loss_fn = BinaryCELoss()
    loss_fn_name = 'Binary Cross Entropy'
    train_neg_sampler = UniformNegativeSampler(train_dataset.n_poi)
    train_neg_sampler_name = 'UniformNegativeSampler'
    test_neg_sampler = UniformNegativeSampler(train_dataset.n_poi)
    test_neg_sampler_name = 'UniformNegativeSampler'
    poi2quadkey = dataset.poi2quadkey
    num_epoch = 100
    train_batch_size = 8
    train_num_neg = 10
    test_batch_size = 16
    test_num_neg = 100
    num_workers = 0
    region_processer = dataset.QUADKEY
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Building Model...")
    model = STAPT_PE(n_poi=train_dataset.n_poi,
                     n_quadkey=len(region_processer.vocab.itos),
                     emb_dim=emb_dim,
                     n_geo_enc_layer=n_geo_enc_layer,
                     n_multi_enc_layer=n_multi_enc_layer,
                     n_dec_layer=n_dec_layer,
                     len_quadkey=len_quadkey,
                     len_seq=len_seq,
                     dropout=dropout)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98))

    print(T.strftime('%Y-%m-%d %H:%M:%S', T.localtime(T.time())))
    print("Dataset:", data_name)
    print("Embedding dimension:", emb_dim)
    print("Layers of GeoEncoder:", n_geo_enc_layer)
    print("Layers of Multi-Encoder:", n_multi_enc_layer)
    print("Layers of Decoder:", n_dec_layer)
    print("Num epoch:", num_epoch)
    print("Train batch size:", train_batch_size, "Num neg samples:", train_num_neg)
    print("Test batch size:", test_batch_size, "Num neg samples:", test_num_neg)
    print("Train Sampler:", train_neg_sampler_name)
    print("Test Sampler:", test_neg_sampler_name)
    print("Loss function:", loss_fn_name)
    print("Opimitizer:", optimizer_name, "lr:", 0.01)
    print("\n")

    with open('results/demo_results.txt', 'a') as f:
        print(T.strftime('%Y-%m-%d %H:%M:%S', T.localtime(T.time())), file=f)
        print("Dataset:", data_name, file=f)
        print("Embedding dimension:", emb_dim, file=f)
        print("Layers of GeoEncoder:", n_geo_enc_layer, file=f)
        print("Layers of Multi-Encoder:", n_multi_enc_layer, file=f)
        print("Layers of Decoder:", n_dec_layer, file=f)
        print("Num epoch:", num_epoch, file=f)
        print("Train batch size:", train_batch_size, "Num neg samples:", train_num_neg, file=f)
        print("Test batch size:", test_batch_size, "Num neg samples:", test_num_neg, file=f)
        print("Train Sampler:", train_neg_sampler_name, file=f)
        print("Test Sampler:", test_neg_sampler_name, file=f)
        print("Loss function:", loss_fn_name, file=f)
        print("Opimitizer:", optimizer_name, "lr:", 0.01, file=f)
        print("\n", file=f)
    f.close()

    if os.path.exists('model/model_for_demo.pt'):
        model.load('model/model_for_demo.pt')
        print("Train for demo data is Completed and Evaluating...")
        hr, ndcg = Evaluate(model,
                            test_dataset=test_dataset,
                            test_neg_sampler=test_neg_sampler,
                            region_processer=region_processer,
                            poi2quadkey=poi2quadkey,
                            device=device,
                            test_batch_size=test_batch_size,
                            test_num_neg=test_num_neg)
        print("Hit@5: {:.4f}, NDCG@5: {:.4f}".format(hr[4], ndcg[4]))
        print("Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[9], ndcg[9]))

        with open('results/demo_results.txt', 'a') as f:
            print('\n', file=f)
            print(T.strftime('%Y-%m-%d %H:%M:%S',T.localtime(T.time())), file=f)
            print('Dataset: demo_data', file=f)
            print('Embedding dim:', 16, 'Attention dim:', 32, 'FeedForward dim:', 64, file=f)
            print('Layers of GeoEncoder: ', 2, 'Layers of MultiEncoder: ', 1, 'MultiDecoder: ', True, file=f)
            print('Train Batch Size:', 2, 'Neg Samples:', 5, 'Epoch:', 10 , file=f)
            print('Test Batch Size:', 1, 'Neg Samples:', 100, file=f)
            print("Hit@5: {:.4f}, NDCG@5: {:.4f}".format(hr[4], ndcg[4]), file=f)
            print("Hit@10: {:.4f}, NDCG@10: {:.4f}".format(hr[9], ndcg[9]), file=f)
    else:
        Train(model,
              train_dataset=train_dataset,
              test_dataset=test_dataset,
              optimizer=optimizer,
              loss_fn=loss_fn,
              train_neg_sampler=train_neg_sampler,
              test_neg_sampler=test_neg_sampler,
              region_processer=region_processer,
              poi2quadkey=poi2quadkey,
              device=device,
              train_num_neg=train_num_neg,
              train_batch_size=train_batch_size,
              num_epochs=num_epoch,
              test_batch_size=test_batch_size,
              test_num_neg=test_num_neg,
              num_workers=num_workers
              )


