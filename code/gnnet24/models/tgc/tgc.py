import math
import random
import sys
from os import makedirs

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import softmax
from torch.optim import SGD
from torch.utils.data import DataLoader
# from torch.nn import Linear
# from torch.optim import Adam
# from sklearn.cluster import KMeans

from .dataset import TGCDataSet
from ..kmeans import KMeans
from ..utils.evaluate import evaluate

FType = torch.FloatTensor
LType = torch.LongTensor

# DID = 0

DATA = "/home/nelsonpassos/data/tgc"


class TGC:
    def __init__(self, args, split="train"):
        self.args = args
        self.seed = args.seed
        self.setting = args.setting
        self.static = args.static
        self.directed = args.directed
        self.the_data = args.dataset
        self.file_path = f"{self.data}/edges.txt"
        self.feature_path = f"{self.data}/x-node2vec.emb"
        self.label_path = f"{self.data}/y.txt"
        self.weights = f"{args.weights}_{args.setting}_seed={args.seed}.emb"
        self.labels = self.read_label()
        self.emb_size = args.emb_size
        self.neg_size = args.neg_size
        self.hist_len = args.hist_len
        self.batch = args.batch_size
        self.clusters = args.clusters
        self.epochs = args.epoch
        self.best = {}

        self.data = TGCDataSet(self.file_path, self.neg_size, self.hist_len, self.feature_path, args.directed, static=args.static)

        self.node_dim = self.data.get_node_dim()
        self.edge_num = self.data.get_edge_num()
        self.feature = self.data.get_feature()

        self.node_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(), requires_grad=True)
        self.pre_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(), requires_grad=False)
        self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)

        self.cluster_layer = Variable((torch.zeros(self.clusters, self.emb_size) + 1.).type(FType).cuda(), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        kmeans = KMeans(n_clusters=int(self.clusters), n_init=20, seed=args.seed, device=args.kmeans_device)
        _ = kmeans.fit(self.feature).predict(self.feature)
        self.cluster_layer.data = kmeans.cluster_centers_.clone().detach() if kmeans.device == "gpu" else torch.tensor(kmeans.cluster_centers_).cuda()

        self.v = 1.0
        self.batch_weight = math.ceil(self.batch / self.edge_num)

        self.opt = SGD(lr=args.learning_rate, params=[self.node_emb, self.delta, self.cluster_layer])
        self.loss = torch.FloatTensor()

        makedirs('../emb/%s/' % self.the_data, exist_ok=True)

        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            random.seed(args.seed)

    def read_label(self, path=None):
        n2l = dict()
        labels = []
        with open(path or self.label_path, 'r') as reader:
            for line in reader:
                parts = line.strip().split()
                n_id, l_id = int(parts[0]), int(parts[1])
                n2l[n_id] = l_id
        reader.close()
        for i in range(len(n2l)):
            labels.append(int(n2l[i]))
        return labels

    def kl_loss(self, z, p):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        the_kl_loss = F.kl_div((q.log()), p, reduction='batchmean')  # l_clu
        return the_kl_loss

    def target_dis(self, emb):
        q = 1.0 / (1.0 + torch.sum(torch.pow(emb.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        tmp_q = q.data
        weight = tmp_q ** 2 / tmp_q.sum(0)
        p = (weight.t() / weight.sum(1)).t()

        return p

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)
        s_pre_emb = self.pre_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)

        s_p = self.target_dis(s_pre_emb)
        s_kl_loss = self.kl_loss(s_node_emb, s_p)
        l_node = s_kl_loss

        new_st_adj = torch.cosine_similarity(s_node_emb, t_node_emb)  # [b]
        res_st_loss = torch.norm(1 - new_st_adj, p=2, dim=0)
        new_sh_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), h_node_emb, dim=2)  # [b,h]
        new_sh_adj = new_sh_adj * h_time_mask
        new_sn_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), n_node_emb, dim=2)  # [b,n]
        res_sh_loss = torch.norm(1 - new_sh_adj, p=2, dim=0).sum(dim=0, keepdims=False)
        res_sn_loss = torch.norm(0 - new_sn_adj, p=2, dim=0).sum(dim=0, keepdims=False)
        l_batch = res_st_loss + res_sh_loss + res_sn_loss

        l_framework = l_node + l_batch

        att = softmax(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2).neg(), dim=1)

        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()

        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)
        p_lambda = p_mu + (att * p_alpha * torch.exp(delta * Variable(d_time)) * Variable(h_time_mask)).sum(
            dim=1)  # [b]

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()

        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(delta * Variable(d_time)).unsqueeze(2)) * (
            Variable(h_time_mask).unsqueeze(2))).sum(dim=1)

        loss = -torch.log(p_lambda.sigmoid() + 1e-6) - torch.log(n_lambda.neg().sigmoid() + 1e-6).sum(dim=1)

        total_loss = loss.sum() + l_framework

        return total_loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        self.opt.zero_grad()
        loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
        self.loss += loss.data
        loss.backward()
        self.opt.step()

    def train(self):
        if self.setting:
            val = TGC(self.args, "val")
            test = TGC(self.args, "test")

        for epoch in range(self.epochs):
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=1)

            for i_batch, sample_batched in enumerate(loader):
                if i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)))
                    sys.stdout.flush()

                self.update(sample_batched['source_node'].type(LType).cuda(),
                            sample_batched['target_node'].type(LType).cuda(),
                            sample_batched['target_time'].type(FType).cuda(),
                            sample_batched['neg_nodes'].type(LType).cuda(),
                            sample_batched['history_nodes'].type(LType).cuda(),
                            sample_batched['history_times'].type(FType).cuda(),
                            sample_batched['history_masks'].type(FType).cuda())

            eva = self.evaluation(self.clusters, self.labels, self.node_emb, self.node_emb)
            sys.stdout.write('\repoch %d: loss=%.4f  ' % (epoch, (self.loss.cpu().numpy() / len(self.data))))
            sys.stdout.write('%s%s\n' % ('Train - ' if self.setting else '', eva))

            if self.setting:
                for batched in DataLoader(val.data, batch_size=val.batch, shuffle=True, num_workers=1):
                    val.forward(batched['source_node'].type(LType).cuda(),
                                batched['target_node'].type(LType).cuda(),
                                batched['target_time'].type(FType).cuda(),
                                batched['neg_nodes'].type(LType).cuda(),
                                batched['history_nodes'].type(LType).cuda(),
                                batched['history_times'].type(FType).cuda(),
                                batched['history_masks'].type(FType).cuda())

                eva = self.evaluation(self.clusters, val.labels, self.node_emb, val.node_emb)
                sys.stdout.write('\repoch %d: loss=%.4f  ' % (epoch, (self.loss.cpu().numpy() / len(self.data))))
                sys.stdout.write('%s%s\n' % ('Val - ' if self.setting else '', eva))

            sys.stdout.flush()

            if epoch == 0 or eva["acc"] > self.best.get("acc"):
                self.best = eva
                self.save_node_embeddings(self.weights)

        print(f'Best - {self.best}')

        # Test
        if self.setting:
            for batched in DataLoader(test.data, batch_size=test.batch, shuffle=True, num_workers=1):
                test.forward(batched['source_node'].type(LType).cuda(),
                             batched['target_node'].type(LType).cuda(),
                             batched['target_time'].type(FType).cuda(),
                             batched['neg_nodes'].type(LType).cuda(),
                             batched['history_nodes'].type(LType).cuda(),
                             batched['history_times'].type(FType).cuda(),
                             batched['history_masks'].type(FType).cuda())

            node_emb = self.load_node_embeddings(self.weights)
            eva = self.evaluation(self.clusters, test.labels, node_emb, test.node_emb)
            print('Test - %s' % eva)

    def save_node_embeddings(self, path):
        embeddings = self.node_emb.cpu().data.numpy() if torch.cuda.is_available() else self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')

        writer.close()

    def load_node_embeddings(self, path):
        node_emb = dict()
        with open(path, 'r') as reader:
            reader.readline()
            for line in reader.readlines():
                node_id = int(line.strip().split()[0])
                embeds = np.fromstring(line.split(" ",1)[-1].strip(), dtype=float, sep=' ')
                node_emb[node_id] = embeds
            reader.close()
        feature = []
        for i in range(len(node_emb)):
            feature.append(node_emb[i])
        return Variable(torch.from_numpy(np.array(feature)).type(FType).cuda(), requires_grad=False)

    @staticmethod
    def evaluation(k, labels, emb, pred):
        embeddings = emb.cpu().data.numpy()
        predicts = pred.cpu().data.numpy()
        model = KMeans(n_clusters=k, n_init=20)
        cluster_id = model.fit(embeddings).predict(predicts)
        return evaluate(labels, cluster_id)
