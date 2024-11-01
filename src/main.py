import argparse
import time
import random
import json
import pickle as pkl
import torch
import os
import numpy as np
from tqdm import tqdm
from dataloader import KGDataset
from torch.cuda.amp import autocast, GradScaler
from utils import scatter_sum

class Option(object):
    def __init__(self, d):
        self.__dict__ = d
    def save(self):
        if not os.path.exists(self.exps_dir):
            os.mkdir(self.exps_dir)
        self.exp_dir = os.path.join(self.exps_dir, self.exp_name)
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        with open(os.path.join(self.exp_dir, "option.txt"), "w") as f:
            json.dump(self.__dict__, f, indent=1)
        return True

class Option_test(object):
    def __init__(self, path):
        with open(os.path.join(path, 'option.txt'), mode='r') as f:
            self.__dict__ = json.load(f)

def set_seed(option):
    random.seed(option.seed)
    np.random.seed(option.seed)
    torch.manual_seed(option.seed)
    os.environ['PYTHONHASHSEED'] = str(option.seed)
    if option.use_gpu: torch.cuda.manual_seed_all(option.seed)

def save_data(target_relation, kg, entity2id, relation2id, triple2id, file_path=None):
    print(len(kg))
    with open(os.path.join(file_path, 'kg_{}.pkl'.format(target_relation.replace('/', '|'))), mode='wb') as fw:
        pkl.dump(kg, fw)
    with open(os.path.join(file_path, 'entity2id_{}.pkl'.format(target_relation.replace('/', '|'))), mode='wb') as fw:
        pkl.dump(entity2id, fw)
    with open(os.path.join(file_path, 'relation2id_{}.pkl'.format(target_relation.replace('/', '|'))), mode='wb') as fw:
        pkl.dump(relation2id, fw)
    with open(os.path.join(file_path, 'triple2id_{}.pkl'.format(target_relation.replace('/', '|'))), mode='wb') as fw:
        pkl.dump(triple2id, fw)

def create_graph(kg, device):
    i_h = torch.from_numpy(kg[0].values).int().to(device)
    i_t = torch.from_numpy(kg[2].values).int().to(device)
    i_r = torch.from_numpy(kg[1].values).int().to(device)
    i_v = torch.ones_like(i_h).bool()

    e2triple = (i_h, None, i_v)

    triple2e = (None, i_t, i_v)

    r2triple = (i_r, None, i_v)

    return e2triple, triple2e, r2triple


def mask_data(values, indices, score):
    for index in indices:
        values[index] = score

def get_planning(min_time, max_time, relation_rates, test_data, thr=0.8):
    total = 0
    relation_time = []
    total_time = 0
    relation_set = set()
    for index, data in test_data:
        target = index
        relation_set.add(target)
    if max_time != -1:
        mean_rate = 0
        for r, rate in relation_rates:
            if total < thr:
                total += rate
                continue
            mean_rate = rate
            break
        for r, rate in relation_rates:
            if r not in relation_set: continue
            if rate > mean_rate:
                running_time = min(max_time, rate / mean_rate * min_time)
            else:
                running_time = min_time
            total_time += running_time
            relation_time.append((r, running_time))
    else:
        for r, rate in relation_rates:
            if r not in relation_set: continue
            relation_time.append((r, None))
    return relation_time, total_time

def calculate_hits(state, top_k, batches, graph, graph_inv, dataset, hits):
    scores, indices = torch.topk(state, k=top_k)
    scores = scores.detach().cpu()
    indices = indices.detach().cpu()
    for j, items in enumerate(batches):
        h, r, t = items['h'], items['r'], items['t']
        indices_list = list(indices[j])
        if t in indices_list:
            truth_index = indices_list.index(t)
            if r < dataset.relation_size:
                scores_tail = scores[j, :].clone()
                if (h, r) in graph.indices.keys():
                    group = graph.get_group((h, r))
                    truths = set(group[2].values.tolist())
                    for tail in indices_list:
                        if tail.item() not in truths: continue
                        if tail.item() == t: continue
                        scores_tail[indices_list.index(tail)] = -1e20
                    sorted = torch.topk(scores_tail, k=10)
                    indices_ = list(sorted[1])
                    if truth_index in indices_:
                        rank = indices_.index(truth_index)
                        hits[rank:] += 1
            else:
                scores_head = scores[j, :].clone()
                if (h, r - dataset.relation_size) in graph_inv.indices.keys():
                    group = graph_inv.get_group((h, r - dataset.relation_size))
                    truths = set(group[0].values.tolist())
                    for head in indices_list:
                        if head.item() not in truths: continue
                        if head.item() == t: continue
                        scores_head[indices_list.index(head)] = -1e20
                sorted = torch.topk(scores_head, k=10)
                indices_ = list(sorted[1])
                if truth_index in indices_:
                    rank = indices_.index(truth_index)
                    hits[rank:] += 1

def calculate_hits_full(state, batches, e2triple, triple2e, r2triple, graph, graph_inv, dataset, hits, use_eql=False):
    scores_all = state
    mrr = 0
    input_x = []
    input_y = []
    r_mask_x = []
    r_mask_y = []
    truths = []
    truths_eval = []
    for j, items in enumerate(batches):
        h, r, t = items['h'], items['r'], items['t']
        scores = scores_all[j]
        truth_score = scores[t].clone()
        truths.append(truth_score)
        if r < dataset.relation_size:
            input_x.append([j, h])
            mask = (r2triple[0] == r).float()
            r_mask_x.append(mask)
            if (h, r) in graph.indices.keys():
                group = graph.get_group((h, r))
                eval = torch.from_numpy(group[2].values).to(scores_all.device).long()
            else:
                eval = torch.LongTensor([]).to(scores_all.device)
        else:
            input_y.append([j, h])
            r_ = r - dataset.relation_size
            mask = (r2triple[0] == r_).float()
            r_mask_y.append(mask)
            if (h, r - dataset.relation_size) in graph_inv.indices.keys():
                group = graph_inv.get_group((h, r - dataset.relation_size))
                eval = torch.from_numpy(group[0].values).to(scores_all.device).long()
            else:
                eval = torch.LongTensor([]).to(scores_all.device)

        eval = torch.sparse.FloatTensor(torch.stack([eval, torch.zeros_like(eval)], dim=0),
                                          torch.ones_like(eval), torch.Size([scores_all.shape[1], 1]))
        truths_eval.append(eval)

    truths_eval = torch.stack(truths_eval, dim=0).to_dense().squeeze(dim=-1)
    if len(r_mask_x) != 0:
        r_mask_x = torch.stack(r_mask_x, dim=0)
        input_x = torch.LongTensor(input_x).to(scores_all.device)
        input_x_oh = torch.nn.functional.one_hot(input_x[:, 1], e2triple[0].shape[0])
        x_facts = torch.index_select(input_x_oh, index=e2triple[0], dim=1) * r_mask_x
        x_mask = scatter_sum(x_facts, triple2e[1].long(), dim=1, dim_size=scores_all.shape[-1])
        scores_all[input_x[:, 0]] = scores_all[input_x[:, 0]] - x_mask * 1e20 - truths_eval[input_x[:, 0]] * 1e20
    if len(r_mask_y) != 0:
        r_mask_y = torch.stack(r_mask_y, dim=0)
        input_y = torch.LongTensor(input_y).to(scores_all.device)
        input_y_oh = torch.nn.functional.one_hot(input_y[:, 1], e2triple[0].shape[0])
        y_facts = torch.index_select(input_y_oh, index=triple2e[1], dim=1) * r_mask_y
        y_mask = scatter_sum(y_facts, e2triple[0].long(), dim=1, dim_size=scores_all.shape[-1])
        scores_all[input_y[:, 0]] = scores_all[input_y[:, 0]] - y_mask * 1e20 - truths_eval[input_y[:, 0]] * 1e20

    for j, items in enumerate(batches):
        truth_score = truths[j]
        scores = scores_all[j]
        m = (scores > truth_score).int().sum()
        if use_eql:
            n = (scores == truth_score).int().sum() + 1
            rank = m + (n + 1) / 2
            # print(m, n, rank, truth_score)
        else:
            rank = m + 1
        hits[round(rank.item()) - 1:] += 1
        mrr += 1 / rank.item()
    return mrr

def topk(matrix, k):
    index_part = np.argpartition(matrix, -k)[-k:]
    part_sort_k = np.argsort(matrix[index_part], axis=-1)
    top_k_index = np.flip(index_part[part_sort_k], axis=-1)
    top_k_scores = matrix[top_k_index]
    return top_k_scores, top_k_index

def valid_process(valid_data, dataset, model, e2triple, triple2e, r2triple, graph_entity,
                  option, raw=False, name='Valid', top_k=100, is_sampled=False):
    model.eval()
    hits = np.zeros(10)
    count = 0
    mrr = 0
    batch_size = option.batch_size
    # if not option.wot_inv: batch_size *= 2
    with torch.inference_mode():
        graph = dataset.graph_entity[0]
        graph_inv = dataset.graph_entity[1]
        data = []
        for index in valid_data.index:
            h, r, t = valid_data[0][index], valid_data[1][index], valid_data[2][index]
            data.append({'h': h, 'r': r, 't': t})
            data.append({'h': t, 'r': r + dataset.relation_size, 't': h})
        # if is_sampled: data = np.random.choice(data, 50)
        if len(data) == 0: return
        if len(data) % batch_size == 0:
            batch_num = int(len(data) / batch_size)
        else:
            batch_num = int(len(data) / batch_size) + 1
        for i in tqdm(range(batch_num)):
            input_x = []
            input_r = []
            batches = data[i * batch_size: (i + 1) * batch_size]
            for items in batches:
                h = items['h']
                r = items['r']
                input_x.append(h)
                input_r.append(r)
            input_x = torch.LongTensor(input_x)
            input_r = torch.LongTensor(input_r)
            if option.use_gpu:
                input_x = input_x.cuda()
                input_r = input_r.cuda()
            # input_x = torch.nn.functional.one_hot(input_x, dataset.entity_dict.shape[0]).bool()
            try:
                state = model(input_x, input_r, e2triple, triple2e, r2triple, is_training=False)
            except Exception as e:
                with autocast(dtype=torch.bfloat16):
                    state = model(input_x, input_r, e2triple, triple2e, r2triple, is_training=False)
            if option.sparse: state = state.to_dense()
            if name == 'Valid' and option.raw:
                scores = state.detach().cpu()
                for j, items in enumerate(batches):
                    h, r, t = items['h'], items['r'], items['t']
                    truth_score_ori = scores[j][t]
                    rank = torch.sum((scores[j, :] >= truth_score_ori).int())
                    hits[rank:] += 1
                    count += 1
            else:
                if option.eval_top:
                    calculate_hits(state, top_k, batches, graph, graph_inv, dataset, hits)
                else:
                    mrr_batch = calculate_hits_full(state, batches, e2triple, triple2e, r2triple,
                                                    graph, graph_inv, dataset, hits, use_eql=True)
                    mrr += mrr_batch
                count += len(batches)

    if count > 0:
        hits /= count
        mrr /= count
    print(hits)
    if option.eval_top:
        hk_prev = 0
        for i in range(10):
            hk = hits[i]
            hk_diff = hk - hk_prev
            mrr += hk_diff * (1.0 / (i + 1))
            hk_prev = hk

    print('{} Count:{}\tMrr:{}\tHit@1:{}\tHit@3:{}\tHit@10:{}'.format(name, count, mrr, hits[0], hits[2], hits[9]))
    return mrr, hits[0], hits[2], hits[9], count

def train(dataset, valid_data, graph_entity, e2triple, triple2e, r2triple, option):
    entity_size = dataset.entity_dict.shape[0]
    kg_size = dataset.kg.shape[0]
    print('Current Time: {}, Max running time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), option.max_time))
    print('Entity Num:', entity_size)
    print('Relation Num:', len(dataset.relation2id))
    print('Train KG Size:', kg_size)
    model = Model(len(dataset.relation2id), option.step, option.length, entity_size, e2triple[-1].shape[0],
                    option.emb_size, option.tau_1, option.tau_2, option.mask_w, option.use_gpu, option.dropout,
                    option.top_k, option.top_k_mask, option.use_soft, option.use_topk)

    if option.use_gpu:
        model = model.cuda()
        torch.cuda.empty_cache()
    if option.use_soft: model.weight.cpu()
    # model = torch.compile(model)
    for parameter in model.parameters():
        print(parameter)

    optimizer = torch.optim.Adam(model.parameters(), lr=option.learning_rate)

    end_flag = False
    max_score = -1
    max_record = {'mrr': 0, 'hit_1': 0, 'hit_3': 0, 'hit_10': 0, 'epoch': 0}
    saved_flag = False
    scaler = GradScaler()
    ori_time = time.time()
    min_time = option.min_time
    for e in range(option.max_epoch):
        model.train()
        total_loss = 0
        if end_flag: break
        for k, batch in enumerate(dataset.batch_iter()):
            start_time = time.time()
            if min_time != -1 and start_time - ori_time > min_time:
                torch.save(model.state_dict(), os.path.join(option.exp_dir, 'model-{}.pt'.format(min_time)))
                min_time += option.min_time

            if option.max_time != -1 and start_time - ori_time > option.max_time:
                end_flag = True
                break
            indices, triples, _ = batch
            mask_data(e2triple[-1], indices, False)
            loss = 0
            input_x = []
            input_r = []
            input_y = []
            for x, r, y in triples:
                input_x.append(x)
                input_r.append(r)
                input_y.append(y)

            input_x = torch.LongTensor(input_x)
            input_r = torch.LongTensor(input_r)
            input_y = torch.LongTensor(input_y)
            if option.use_gpu:
                input_x = input_x.cuda()
                input_r = input_r.cuda()
                input_y = input_y.cuda()
            with autocast(dtype=torch.float32):
                state = model(input_x, input_r, e2triple, triple2e, r2triple, is_training=True)
            end_time_forward = time.time()
            loss = model.log_loss(state, input_y)

            end_time_loss = time.time()
            total_loss += loss.item()
            loss.backward()
            if (k + 1) % option.accum_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            end_time = time.time()
            if (k + 1) % 10 == 0:
                print('Epoch: {}, Batch: {}, Loss: {}, Time cost: {}\t{}\t{}'.format(e, k, loss.item(),
                                                                                     end_time_forward - start_time,
                                                                                     end_time_loss - start_time,
                                                                                     end_time - start_time))
            mask_data(e2triple[-1], indices, True)
        print('Epoch: {}, Total Loss: {}'.format(e, total_loss))
        if option.early_stop and e > -1:
            mrr, hit_1, hit_3, hit_10, _ = valid_process(valid_data, dataset, model, e2triple, triple2e,
                                                 r2triple, graph_entity, option, is_sampled=False, name='Valid')
            if hit_10 > max_score:
                max_score = hit_10
                max_record['mrr'] = {'valid': mrr}
                max_record['hit_1'] = {'valid': hit_1}
                max_record['hit_3'] = {'valid': hit_3}
                max_record['hit_10'] = {'valid': hit_10}
                max_record['epoch'] = e

                torch.save(model.state_dict(), os.path.join(option.exp_dir, 'model.pt'))
                saved_flag = True
            print('=' * 100)
            print('Valid Max Score : Epoch:{}\tMrr:{}\tHit@1:{}\tHit@3:{}\tHit@10:{}'.format(max_record['epoch'],
                                                                            max_record['mrr']['valid'], max_record['hit_1']['valid'],
                                                                            max_record['hit_3']['valid'], max_record['hit_10']['valid']))
            if max_score == 1: end_flag = True
            if e - max_record['epoch'] >= option.max_epoch // 2: end_flag = True
    if not saved_flag: torch.save(model.state_dict(), os.path.join(option.exp_dir, 'model.pt'))

def test(dataset, graph_entity, e2triple, triple2e, r2triple, option):
    model = Model(len(dataset.relation2id), option.step, option.length, dataset.entity_dict.shape[0],
                    e2triple[-1].shape[0], option.emb_size, option.tau_1, option.tau_2, option.mask_w,
                    option.use_gpu, option.dropout, option.top_k, option.top_k_mask, option.use_soft, option.use_topk)
    if option.ckpt_step == -1:
        model_save_path = os.path.join(option.exp_dir, 'model.pt')
    else:
        model_save_path = os.path.join(option.exp_dir, 'model-{}.0.pt'.format(option.ckpt_step))
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    if option.use_gpu:
        model = model.cuda()
        torch.cuda.empty_cache()
    if option.use_soft: model.weight.cpu()
    test_data = dataset.test_data
    if option.eval_valid: test_data = dataset.valid_data
    mrr, hit_1, hit_3, hit_10, _ = valid_process(test_data, dataset, model, e2triple, triple2e,
                                                     r2triple, graph_entity, option, name='Test')
    return mrr, hit_1, hit_3, hit_10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment setup")
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--exps_dir', default=None, type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--use_gpu',  default=False, action="store_true")
    parser.add_argument('--gpu_id', default=4, type=int)
    # model architecture
    parser.add_argument('--length', default=50, type=int)
    parser.add_argument('--step', default=3, type=int)
    parser.add_argument('--tau_1', default=1, type=float)
    parser.add_argument('--tau_2', default=1, type=float)
    parser.add_argument('--mask_w', default=False, action="store_true")
    parser.add_argument('--use_topk', default=False, action="store_true")
    parser.add_argument('--top_k', default=100000, type=int)
    parser.add_argument('--top_k_mask', default=100000, type=int)
    parser.add_argument('--emb_size', default=128, type=int)
    parser.add_argument('--lammda', default=0.0, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--eval_top', default=False, action="store_true")
    parser.add_argument('--sparse', default=False, action="store_true")
    parser.add_argument('--use_soft', default=False, action="store_true")
    # optimization
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--accum_step', default=1, type=int)
    parser.add_argument('--iteration_per_batch', default=300000, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--early_stop', default=False, action="store_true")
    parser.add_argument('--do_train', default=False, action="store_true")
    parser.add_argument('--do_test', default=False, action="store_true")
    parser.add_argument('--wot_inv', default=False, action="store_true")
    parser.add_argument('--eval_valid', default=False, action="store_true")
    parser.add_argument('--raw', default=False, action="store_true")
    parser.add_argument('--max_time', default=-1, type=float)
    parser.add_argument('--min_time', default=-1, type=float)
    parser.add_argument('--threshold', default=1e-6, type=float)
    parser.add_argument('--negative_sampling_size', default=100, type=int)
    parser.add_argument('--prob', default=0.9, type=float)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--ckpt_step', default=-1, type=int)

    d = vars(parser.parse_args())

    option = Option(d)
    if option.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(option.gpu_id)
    option.tag = time.strftime("%y-%m-%d %H:%M:%S")
    set_seed(option)
    dataset = KGDataset(option)

    device = 'cpu'
    if option.use_gpu: device = 'cuda'
    e2triple, triple2e, r2triple = create_graph(dataset.kg, device)
    graph_entity = None
    if option.early_stop or option.do_test: graph_entity = dataset.graph_entity

    if option.model_name == 'NeuralLP':
        if option.sparse:
            from model_NeuralLP_sp import Model
        else:
            from model_NeuralLP import Model
    elif option.model_name == 'DRUM':
        if option.sparse:
            from model_DRUM_sp import Model
        else:
            from model_DRUM import Model
    elif option.model_name == 'mmDRUM':
        if option.sparse:
            from model_mmDRUM_sp import Model
        else:
            from model_mmDRUM import Model
    elif option.model_name == 'smDRUM':
        if option.sparse:
            from model_smDRUM_sp import Model
        else:
            from model_smDRUM import Model
    else:
        print('This model has not been implemented.')
        exit(1)
    if option.do_train:
        bl = option.save()
        print("Option saved.")
        ori_time = time.time()
        train(dataset, dataset.valid_data, graph_entity, e2triple, triple2e, r2triple, option)
        print('Current Time: {}, Total cost: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), time.time() - ori_time))
    if option.do_test:
        option.exp_dir = os.path.join(option.exps_dir, option.exp_name)
        ori_time = time.time()
        mrr_all, hit_1_all, hit_3_all, hit_10_all = test(dataset, graph_entity, e2triple, triple2e, r2triple,
                                                         option)
        print('Test Score: Mrr:{}\tHit@1:{}\tHit@3:{}\tHit@10:{}'.format(mrr_all, hit_1_all, hit_3_all, hit_10_all))
        print('Current Time: {}, Total cost: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), time.time() - ori_time))



