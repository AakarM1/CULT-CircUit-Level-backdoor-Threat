from __future__ import print_function

from copy import deepcopy
import os

import torch
import torch.nn.functional as F
import logging
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from collections import defaultdict, Counter

from utils import utils
from utils.backdoor_semantic_utils import SemanticBackdoor_Utils
from utils.backdoor_utils import Backdoor_Utils
from utils.flguardian_defense import FLGuardianDefense
import time
import json
import csv
try:
    from scipy.stats import wasserstein_distance as _wasserstein_distance
    from scipy.spatial.distance import jensenshannon as _jensenshannon
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def find_separate_point(d):
    # d should be flatten and np or list
    d = sorted(d)
    sep_point = 0
    max_gap = 0
    for i in range(len(d)-1):
        if d[i+1] - d[i] > max_gap:
            max_gap = d[i+1] - d[i]
            sep_point = d[i] + max_gap/2
    return sep_point

def DBSCAN_cluster_minority(dict_data):
    ids = np.array(list(dict_data.keys()))
    values = np.array(list(dict_data.values()))
    if len(values.shape) == 1:
        values = values.reshape(-1,1)
    cluster_ = DBSCAN(n_jobs=-1).fit(values)
    offset_ids = find_minority_id(cluster_)
    minor_id = ids[list(offset_ids)]
    return minor_id

def Kmean_cluster_minority(dict_data):
    ids = np.array(list(dict_data.keys()))
    values = np.array(list(dict_data.values()))
    if len(values.shape) == 1:
        values = values.reshape(-1,1)
    cluster_ = KMeans(n_clusters=2, random_state=0).fit(values)
    offset_ids = find_minority_id(cluster_)
    minor_id = ids[list(offset_ids)]
    return minor_id

def find_minority_id(clf):
    count_1 = sum(clf.labels_ == 1)
    count_0 = sum(clf.labels_ == 0)
    mal_label = 0 if count_1 > count_0 else 1
    atk_id = np.where(clf.labels_ == mal_label)[0]
    atk_id = set(atk_id.reshape((-1)))
    return atk_id

def find_majority_id(clf):
    counts = Counter(clf.labels_)
    major_label = max(counts, key=counts.get)
    major_id = np.where(clf.labels_ == major_label)[0]
    #major_id = set(major_id.reshape(-1))
    return major_id

def find_targeted_attack_complex(dict_lHoGs, cosine_dist=False):
    """Construct a set of suspecious of targeted and unreliable clients
    by using [normalized] long HoGs (dict_lHoGs dictionary).
    We use two ways of clustering to find all possible suspicious clients:
      - 1st cluster: Using KMeans (K=2) based on Euclidean distance of
      long_HoGs==> find minority ids.
      - 2nd cluster: Using KMeans (K=2) based on angles between
      long_HoGs to median (that is calculated based on only
      normal clients output from the 1st cluster KMeans).
    """
    id_lHoGs = np.array(list(dict_lHoGs.keys()))
    value_lHoGs = np.array(list(dict_lHoGs.values()))
    cluster_lh1 = KMeans(n_clusters=2, random_state=0).fit(value_lHoGs)
    offset_tAtk_id1 = find_minority_id(cluster_lh1)
    sus_tAtk_id1 = id_lHoGs[list(offset_tAtk_id1)]
    logging.info(f"sus_tAtk_id1: {sus_tAtk_id1}")

    offset_normal_ids = find_majority_id(cluster_lh1)
    normal_ids = id_lHoGs[list(offset_normal_ids)]
    normal_lHoGs = value_lHoGs[list(offset_normal_ids)]
    median_normal_lHoGs = np.median(normal_lHoGs, axis=0)
    d_med_lHoGs = {}
    for idx in id_lHoGs:
        if cosine_dist:
            # cosine similarity between median and all long HoGs points.
            d_med_lHoGs[idx] = np.dot(dict_lHoGs[idx], median_normal_lHoGs)
        else:
            # Euclidean distance
            d_med_lHoGs[idx] = np.linalg.norm(dict_lHoGs[idx]- median_normal_lHoGs)

    cluster_lh2 = KMeans(n_clusters=2, random_state=0).fit(np.array(list(d_med_lHoGs.values())).reshape(-1,1))
    offset_tAtk_id2 = find_minority_id(cluster_lh2)
    sus_tAtk_id2 = id_lHoGs[list(offset_tAtk_id2)]
    logging.debug(f"d_med_lHoGs={d_med_lHoGs}")
    logging.info(f"sus_tAtk_id2: {sus_tAtk_id2}")
    sus_tAtk_uRel_id = set(list(sus_tAtk_id1)).union(set(list(sus_tAtk_id2)))
    logging.info(f"sus_tAtk_uRel_id: {sus_tAtk_uRel_id}")
    return sus_tAtk_uRel_id


def find_targeted_attack(dict_lHoGs):
    """Construct a set of suspecious of targeted and unreliable clients
    by using long HoGs (dict_lHoGs dictionary).
      - cluster: Using KMeans (K=2) based on Euclidean distance of
      long_HoGs==> find minority ids.
    """
    id_lHoGs = np.array(list(dict_lHoGs.keys()))
    value_lHoGs = np.array(list(dict_lHoGs.values()))
    cluster_lh1 = KMeans(n_clusters=2, random_state=0).fit(value_lHoGs)
    #cluster_lh = DBSCAN(eps=35, min_samples=7, metric='mahalanobis', n_jobs=-1).fit(value_lHoGs)
    #logging.info(f"DBSCAN labels={cluster_lh.labels_}")
    offset_tAtk_id1 = find_minority_id(cluster_lh1)
    sus_tAtk_id = id_lHoGs[list(offset_tAtk_id1)]
    logging.info(f"This round TARGETED ATTACK: {sus_tAtk_id}")
    return sus_tAtk_id

class Server():
    def __init__(self, model, dataLoader, criterion=F.nll_loss, device='cpu'):
        self.clients = []
        self.model = model
        self.dataLoader = dataLoader
        self.device = device
        self.model_type = 'quantum' #Find a way to get model type from _main.
        self.emptyStates = None
        self.init_stateChange()
        self.Delta = None
        self.iter = 0
        self.AR = self.FedAvg
        self.func = torch.mean
        self.isSaveChanges = False
        self.savePath = './AggData'
        self.criterion = criterion
        self.path_to_aggNet = ""
        self.sims = None
        self.mal_ids = set()
        self.uAtk_ids = set()
        self.tAtk_ids = set()
        self.flip_sign_ids = set()
        self.unreliable_ids = set()
        self.suspicious_id = set()
        self.log_sims = None
        self.log_norms = None
        # At least tao_0 + delay_decision rounds to get first decision.
        self.tao_0 = 3
        self.delay_decision = 2 # 2 consecutive rounds
        self.pre_mal_id = defaultdict(int)
        self.count_unreliable = defaultdict(int)
        # DBSCAN hyper-parameters:
        self.dbscan_eps = 0.5
        self.dbscan_min_samples=5
        # FLGuardian defense (initialized with default parameters)
        self.flguardian = None
        self.use_flguardian = False
        # Extension-experiment knobs
        self.log_norm_cosine = False
        self.csv_log_path = None
        self.exp_tag = None
        self.attacker_ids = set()
        self.disable_s2 = False
        self._csv_initialized = False

    def set_log_path(self, log_path, exp_name, t_run):
        self.log_path = log_path
        self.log_sim_path = '{}/sims_{}_{}.npy'.format(log_path, exp_name, t_run)
        self.log_norm_path = '{}/norms_{}_{}.npy'.format(log_path, exp_name, t_run)
        self.log_results = f'{log_path}/acc_prec_rec_f1_{exp_name}_{t_run}.txt'
        
        # Ensure the directory for the log file exists
        log_file_dir = os.path.dirname(self.log_results)
        if not os.path.isdir(log_file_dir):
            os.makedirs(log_file_dir)
            
        self.output_file = open(self.log_results, 'w', encoding='utf-8')

    def close(self):
        if self.log_sims is None or self.log_norms is None:
            return
        with open(self.log_sim_path, 'wb') as f:
            np.save(f, self.log_sims, allow_pickle=False)
        with open(self.log_norm_path, 'wb') as f:
            np.save(f, self.log_norms, allow_pickle=False)
        self.output_file.close()

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.emptyStates = states

    def attach(self, c):
        self.clients.append(c)
        self.num_clients = len(self.clients)

    def distribute(self):
        for c in self.clients:
            c.setModelParameter(self.model.state_dict())

    def test(self):
        logging.info("[Server] Start testing")
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        count = 0
        try:
            nb_classes = len(self.dataLoader.dataset.classes)
            logging.info(f"Dynamically found {nb_classes} classes.")
        except (AttributeError, TypeError):
            logging.warning("Could not determine number of classes from dataset, defaulting to 10.")
            nb_classes = 10 # Fallback for datasets without a .classes attribute for MNIST, Fashion-MNIST, CIFAR-10
        cf_matrix = torch.zeros(nb_classes, nb_classes)
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                if output.dim() == 1:
                    pred = torch.round(torch.sigmoid(output))
                else:
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += pred.shape[0]
                for t, p in zip(target.view(-1), pred.view(-1)):
                    cf_matrix[t.long(), p.long()] += 1
        test_loss /= count
        accuracy = 100. * correct / count
        self.model.cpu()  ## avoid occupying gpu when idle
        logging.info(
            '[Server] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
                test_loss, correct, count, accuracy))
        logging.info(f"[Sever] Confusion matrix:\n {cf_matrix.detach().cpu()}")
        cf_matrix = cf_matrix.detach().cpu().numpy()
        row_sum = np.sum(cf_matrix, axis=0) # predicted counts
        col_sum = np.sum(cf_matrix, axis=1) # targeted counts
        diag = np.diag(cf_matrix)
        precision = diag / row_sum # tp/(tp+fp), p is predicted positive.
        recall = diag / col_sum # tp/(tp+fn)
        f1 = 2*(precision*recall)/(precision+recall)
        m_acc = np.sum(diag)/np.sum(cf_matrix)
        results = {'accuracy':accuracy,'test_loss':test_loss,
                   'precision':precision.tolist(),'recall':recall.tolist(),
                   'f1':f1.tolist(),'confusion':cf_matrix.tolist(),
                   'epoch':self.iter}
        json.dump(results, self.output_file)
        self.output_file.write("\n")
        self.output_file.flush()
        logging.info(f"[Server] Precision={precision},\n Recall={recall},\n F1-score={f1},\n my_accuracy={m_acc*100.}[%]")

        return test_loss, accuracy

    def test_backdoor_quantum(self):
        """Quantum-trigger backdoor test: returns (loss, attack_accuracy)"""
        # TODO: Doesn't work currently
        logging.info("[Server] Start testing QUANTUM backdoor")
        self.model.to(self.device).eval()

        # Select which trojan to evaluate and its params:
        # You could also pass these in via args or experiment config.
        backdoor_type = "grover"
        params = {"marked_state": [1, 0, 1, 0],
                  "trigger_state": [0.2, 1.3, 2.0, 0.5],
                  "period": 3,
                  "phase": 0.25}

        # 1) Swap in the appropriate trojan QNode
        if backdoor_type == "grover":
            self.model.q_layer = self.model.q_layer_grover(params["marked_state"])
        elif backdoor_type == "noise":
            self.model.q_layer = self.model.q_layer_noise(params["trigger_state"])
        elif backdoor_type == "bitflip":
            self.model.q_layer = self.model.q_layer_bitflip(params["period"])
        elif backdoor_type == "signflip":
            self.model.q_layer = self.model.q_layer_signflip(params["phase"])
        else:
            raise ValueError(f"Unknown quantum backdoor: {backdoor_type}")

        total, correct = 0, 0
        loss_sum = 0.0
        backdoor_label = Backdoor_Utils().backdoor_label

        with torch.no_grad():
            for data, _ in self.dataLoader:
                data = data.to(self.device)
                batch_size = data.size(0)

                # 2) Classical feature extraction up to fc1
                x = F.max_pool2d(F.relu(self.model.conv1(data)), 2)
                x = F.max_pool2d(F.relu(self.model.conv2(x)), 2)
                x = x.view(batch_size, -1)
                features = F.relu(self.model.fc1(x))

                # 3) Inject the trigger into the feature vector
                if backdoor_type == "grover":
                    trigger = torch.tensor(
                        params["marked_state"],
                        dtype=features.dtype,
                        device=features.device
                    ).unsqueeze(0).repeat(batch_size, 1)
                    features = trigger
                # For noise/bitflip/signflip you may leave 'features' as is,
                # since their QNodes internally detect the trigger via phase, QFT, QPE, etc.

                # 4) Forward through the trojaned QNode + remaining head
                q_out = self.model.q_layer(features)
                x2 = F.relu(self.model.fc2(q_out))
                out = self.model.fc3(x2)

                # 5) Compute loss against the backdoor target label
                target_bd = torch.full((batch_size,), backdoor_label,
                                       dtype=torch.long, device=self.device)
                loss = F.cross_entropy(out, target_bd, reduction="sum")
                loss_sum += loss.item()

                # 6) Count how many predictions match the backdoor label
                pred = out.argmax(dim=1)
                correct += (pred == target_bd).sum().item()
                total += batch_size

        avg_loss = loss_sum / total
        accuracy = 100.0 * correct / total
        logging.info(f"[Server] Quantum backdoor ({backdoor_type}) — loss: {avg_loss:.4f}, "
                     f"attack acc: {accuracy:.2f}%")
        return avg_loss, accuracy
    
    def test_backdoor(self):
        # override the old method
        # print(f"[Server] Testing backdoor with model type: {self.model_type}")
        # if self.model_type == "quantum":
        #     return self.test_backdoor_quantum()
        # else:
        #     return self.test_backdoor_classical()
        logging.info("[Server] Start testing backdoor\n")
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        utils = Backdoor_Utils()
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = utils.get_poison_batch(data, target, backdoor_fraction=1,
                                                      backdoor_label=utils.backdoor_label, evaluation=True)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataLoader.dataset)
        accuracy = 100. * correct / len(self.dataLoader.dataset)

        self.model.cpu()  ## avoid occupying gpu when idle
        logging.info(
            '[Server] Test set (Backdoored): Average loss: {:.4f}, Success rate: {}/{} ({:.0f}%)\n'.
                format(test_loss, correct, len(self.dataLoader.dataset), accuracy))
        return test_loss, accuracy
        
    def test_semanticBackdoor(self):
        logging.info("[Server] Start testing semantic backdoor")

        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        utils = SemanticBackdoor_Utils()
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = utils.get_poison_batch(data, target, backdoor_fraction=1,
                                                      backdoor_label=utils.backdoor_label, evaluation=True)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataLoader.dataset)
        accuracy = 100. * correct / len(self.dataLoader.dataset)

        self.model.cpu()  ## avoid occupying gpu when idle
        logging.info(
            '[Server] Test set (Semantic Backdoored): Average loss: {:.4f}, Success rate: {}/{} ({:.0f}%)\n'.
                format(test_loss, correct, len(self.dataLoader.dataset), accuracy))
        return test_loss, accuracy, data, pred

    def train(self, group):
        # 1) Run all selected clients
        selectedClients = [self.clients[i] for i in group]
        client_deltas = {}
        
        # TODO: Add this to parser
        normCosineCheck = bool(getattr(self, 'log_norm_cosine', False)) or bool(getattr(self, 'csv_log_path', None))
        # Capture per-client metrics for optional CSV export
        _round_l2 = {}
        _round_cos = {}
        _round_zdev = {}
        if normCosineCheck:
            logging.info("[Stealth] Norm-Cosine check is enabled.")
            l2_dict       = {}
            cos_dict      = {}
            
            for c in selectedClients:
                if hasattr(c, 'set_server_iter'):
                    c.set_server_iter(self.iter)
                    if hasattr(c, 'is_benign_phase') and hasattr(c, 'honest_stats'):
                        phase = "BENIGN" if c.is_benign_phase() else "STEALTH"
                        if c.is_benign_phase():
                            logging.info(f"[Stealth] Client {c.cid} in {phase} phase, collecting stats")
                        else:
                            stats = c.honest_stats
                            logging.info(f"[Stealth] Client {c.cid} in {phase} phase, using stats: mean_l2={stats.get('mean_l2', 0):.4f}, std_l2={stats.get('std_l2', 0):.4f}")
                c.train()
                c.update()
                
                delta_vec = utils.net2vec(c.getDelta()).detach()
                client_deltas[c.cid] = delta_vec
                
                l2 = torch.norm(delta_vec, p=2).item()
                l2_dict[c.cid] = l2
                
                logging.info(f"[Stealth] Client {c.cid} L2_norm = {l2:.4f}")

            # Honest‐mean Δ
            mean_delta = torch.stack(list(client_deltas.values()), dim=1).mean(dim=1)
            
            # Cosine similarity
            for cid, delta_vec in client_deltas.items():
                cos = F.cosine_similarity(
                    delta_vec.unsqueeze(0),
                    mean_delta.unsqueeze(0),
                    dim=1
                ).item()
                cos_dict[cid] = cos
                logging.info(f"[Stealth] Client {cid} cos_sim = {cos:.4f}")
                
            # ——— NEW OUTLIER DETECTION —————————————————————
            # A) L₂-norm outliers: |l2 − μ| > 2σ
            vals      = np.array(list(l2_dict.values()), dtype=float)
            mu, sigma = vals.mean(), vals.std()
            out_l2    = [cid for cid, v in l2_dict.items() if abs(v - mu) > 2*sigma]
            logging.info(f"[Stealth][Outlier] by L2-norm: {out_l2}")

            # B) Cosine-sim outliers: |cos − μ_c| > 2σ_c
            cvals      = np.array(list(cos_dict.values()), dtype=float)
            mu_c, sigma_c = cvals.mean(), cvals.std()
            out_cos    = [cid for cid, v in cos_dict.items() if abs(v - mu_c) > 2*sigma_c]
            logging.info(f"[Stealth][Outlier] by Cos-sim: {out_cos}")

            # C) K-Means clustering outliers on raw updates
            vecs   = {cid: delta.cpu().numpy() for cid, delta in client_deltas.items()}
            out_km = Kmean_cluster_minority(vecs)
            logging.info(f"[Stealth][Outlier] by KMeans: {list(out_km)}")
            # ————————————————————————————————————————————————

            _round_l2 = dict(l2_dict)
            _round_cos = dict(cos_dict)
            for c in selectedClients:
                _round_zdev[c.cid] = getattr(c, 'last_z_dev', None)
        else:
            for c in selectedClients:
                c.train()
                c.update()
        
        # ========== FLGuardian Defense ==========
        if self.use_flguardian and self.flguardian is not None:
            logging.info("[FLGuardian] Running FLGuardian defense...")
            selectedClients = self.flguardian.defend(selectedClients, return_suspicious_only=False)
            logging.info(f"[FLGuardian] Selected {len(selectedClients)} clients after defense")
        # ========================================
                
        if self.isSaveChanges:
            self.saveChanges(selectedClients)

        tic = time.perf_counter()
        Delta = self.AR(selectedClients)
        toc = time.perf_counter()
        logging.info(f"[Server] The aggregation takes {toc - tic:0.6f} seconds.\n")

        for param in self.model.state_dict():
            self.model.state_dict()[param] += Delta[param]
        self.iter += 1

        # Stash for deferred CSV flush (so test accuracy can be included)
        self._pending_csv = (_round_l2, _round_cos, _round_zdev)

    def flush_round_metrics(self, test_acc=float('nan'), test_loss=float('nan'),
                            backdoor_acc=float('nan')):
        """Call after server.test() each round to append a CSV row that
        includes both divergence metrics and test accuracy."""
        if not getattr(self, 'csv_log_path', None):
            return
        pend = getattr(self, '_pending_csv', None)
        if not pend:
            return
        l2_dict, cos_dict, zdev_dict = pend
        if not l2_dict:
            return
        try:
            self._write_round_csv(l2_dict, cos_dict, zdev_dict,
                                  test_acc=test_acc, test_loss=test_loss,
                                  backdoor_acc=backdoor_acc)
        except Exception as e:
            logging.warning(f"[CSV] divergence write failed: {e}")
        self._pending_csv = None

    def _write_round_csv(self, l2_dict, cos_dict, zdev_dict,
                         test_acc=float('nan'), test_loss=float('nan'),
                         backdoor_acc=float('nan')):
        path = self.csv_log_path
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        atk_ids = set(getattr(self, 'attacker_ids', set()))
        benign_norms = [l2_dict[c] for c in l2_dict if c not in atk_ids]
        mal_norms    = [l2_dict[c] for c in l2_dict if c in atk_ids]
        benign_cos   = [cos_dict[c] for c in cos_dict if c not in atk_ids]
        mal_cos      = [cos_dict[c] for c in cos_dict if c in atk_ids]

        wd = js = float('nan')
        if _HAS_SCIPY and len(benign_norms) > 0 and len(mal_norms) > 0:
            try:
                wd = float(_wasserstein_distance(benign_norms, mal_norms))
            except Exception:
                wd = float('nan')
            try:
                # Bin both into shared histogram for JS
                lo = min(min(benign_norms), min(mal_norms))
                hi = max(max(benign_norms), max(mal_norms))
                if hi <= lo:
                    js = 0.0
                else:
                    bins = np.linspace(lo, hi, 21)
                    p, _ = np.histogram(benign_norms, bins=bins, density=False)
                    q, _ = np.histogram(mal_norms,    bins=bins, density=False)
                    p = p.astype(float) + 1e-12
                    q = q.astype(float) + 1e-12
                    p /= p.sum(); q /= q.sum()
                    js = float(_jensenshannon(p, q))
            except Exception:
                js = float('nan')

        zdev_vals = [v for v in zdev_dict.values() if v is not None]
        z_mean = float(np.mean(zdev_vals)) if zdev_vals else float('nan')

        new_file = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow([
                    'round', 'exp_tag',
                    'test_accuracy', 'test_loss', 'backdoor_acc',
                    'wasserstein_norms', 'jensen_shannon_norms',
                    'mean_z_deviation',
                    'mean_norm_benign', 'mean_norm_mal',
                    'mean_cos_benign', 'mean_cos_mal',
                    'n_benign', 'n_mal',
                    'per_client_norms_json', 'per_client_cos_json',
                ])
            w.writerow([
                self.iter,
                getattr(self, 'exp_tag', '') or '',
                test_acc, test_loss, backdoor_acc,
                wd, js, z_mean,
                float(np.mean(benign_norms)) if benign_norms else float('nan'),
                float(np.mean(mal_norms)) if mal_norms else float('nan'),
                float(np.mean(benign_cos)) if benign_cos else float('nan'),
                float(np.mean(mal_cos)) if mal_cos else float('nan'),
                len(benign_norms), len(mal_norms),
                json.dumps({str(k): float(v) for k, v in l2_dict.items()}),
                json.dumps({str(k): float(v) for k, v in cos_dict.items()}),
            ])

    def saveChanges(self, clients):

        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]

        param_trainable = utils.getTrainableParameters(self.model)

        param_nontrainable = [param for param in Delta.keys() if param not in param_trainable]
        for param in param_nontrainable:
            del Delta[param]
        logging.info(f"[Server] Saving the model weight of the trainable paramters:\n {Delta.keys()}")
        for param in param_trainable:
            ##stacking the weight in the innerest dimension
            param_stack = torch.stack([delta[param] for delta in deltas], -1)
            shaped = param_stack.view(-1, len(clients))
            Delta[param] = shaped

        saveAsPCA = False # True
        saveOriginal = True #False
        if saveAsPCA:
            from utils import convert_pca
            proj_vec = convert_pca._convertWithPCA(Delta)
            savepath = f'{self.savePath}/pca_{self.iter}.pt'
            torch.save(proj_vec, savepath)
            logging.info(f'[Server] The PCA projections of the update vectors have been saved to {savepath} (with shape {proj_vec.shape})')
#             return
        if saveOriginal:
            savepath = f'{self.savePath}/{self.iter}.pt'

            torch.save(Delta, savepath)
            logging.info(f'[Server] Update vectors have been saved to {savepath}')

    def set_AR_param(self, dbscan_eps=0.5, min_samples=5):
        logging.info(f"SET DBSCAN eps={dbscan_eps}, min_samples={min_samples}")
        self.dbscan_eps = dbscan_eps
        self.min_samples=min_samples

    def set_flguardian(self, use_flguardian=False, layer_weighting='quadratic', 
                       trust_threshold=0.5, top_k=None, n_clusters=2, random_state=42):
        """
        Initialize and configure FLGuardian defense mechanism.
        
        Args:
            use_flguardian: Enable/disable FLGuardian
            layer_weighting: 'uniform', 'linear', or 'quadratic'
            trust_threshold: Minimum trust score to keep client (0-1)
            top_k: Keep only top-k clients (overrides threshold if set)
            n_clusters: Number of clusters for K-Means
            random_state: Random seed for reproducibility
        """
        self.use_flguardian = use_flguardian
        if use_flguardian:
            self.flguardian = FLGuardianDefense(
                n_clusters=n_clusters,
                use_cosine_and_euclidean=True,
                layer_weighting=layer_weighting,
                trust_threshold=trust_threshold,
                top_k=top_k,
                random_state=random_state
            )
            logging.info(
                f"[FLGuardian] Initialized with: "
                f"layer_weighting={layer_weighting}, "
                f"trust_threshold={trust_threshold}, "
                f"top_k={top_k}, "
                f"n_clusters={n_clusters}, "
                f"random_state={random_state}"
            )
        else:
            logging.info("[FLGuardian] Defense disabled")

    ## Aggregation functions ##

    def set_AR(self, ar):
        if ar == 'fedavg':
            self.AR = self.FedAvg
        elif ar == 'median':
            self.AR = self.FedMedian
        elif ar == 'gm':
            self.AR = self.geometricMedian
        elif ar == 'krum':
            self.AR = self.krum
        elif ar == 'mkrum':
            self.AR = self.mkrum
        elif ar == 'foolsgold':
            self.AR = self.foolsGold
        elif ar == 'residualbase':
            self.AR = self.residualBase
        elif ar == 'attention':
            self.AR = self.net_attention
        elif ar == 'mlp':
            self.AR = self.net_mlp
        elif ar == 'mudhog':
            self.AR = self.mud_hog
        elif ar == 'fedavg_oracle':
            self.AR = self.fedavg_oracle
        else:
            raise ValueError("Not a valid aggregation rule or aggregation rule not implemented")

    def FedAvg(self, clients):
        out = self.FedFuncWholeNet(clients, lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        return out

    def fedavg_oracle(self, clients):
        normal_clients = []
        for i in range(self.num_clients):
            if i >= 4:
                normal_clients.append(clients[i])
        out = self.FedFuncWholeNet(normal_clients, lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        return out

    def FedMedian(self, clients):
        out = self.FedFuncWholeNet(clients, lambda arr: torch.median(arr, dim=-1, keepdim=True)[0])
        return out

    def krum(self, clients, f=0):
        """
        Krum aggregation rule - selects the gradient closest to its neighbors
        f: number of Byzantine clients to tolerate
        """
        logging.info("[Server] Using Krum aggregation")
        
        # Get all client gradients as vectors
        deltas = [c.getDelta() for c in clients]
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        
        n = len(vecs)
        if f == 0:
            f = min(n // 3, 2)  # Default: tolerate up to n/3 Byzantine clients
        
        logging.info(f"[Krum] n={n} clients, f={f} Byzantine tolerance")
        
        # Compute pairwise distances
        distances = torch.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(vecs[i] - vecs[j], p=2)
                distances[i, j] = dist
                distances[j, i] = dist
        
        # For each client, compute sum of distances to closest n-f-2 neighbors
        scores = torch.zeros(n)
        for i in range(n):
            # Sort distances for client i (excluding distance to self)
            client_distances = distances[i].clone()
            client_distances[i] = float('inf')  # Exclude self
            sorted_distances, _ = torch.sort(client_distances)
            
            # Sum of distances to closest n-f-2 neighbors
            scores[i] = torch.sum(sorted_distances[:n-f-2])
        
        # Select client with minimum score
        selected_idx = torch.argmin(scores).item()
        selected_client = clients[selected_idx]
        
        logging.info(f"[Krum] Selected client {selected_idx} with score {scores[selected_idx]:.6f}")
        
        # Return the gradient of the selected client
        Delta = deepcopy(self.emptyStates)
        selected_delta = selected_client.getDelta()
        
        for param in Delta:
            if param in selected_delta:
                Delta[param] = selected_delta[param]
        
        return Delta

    def mkrum(self, clients, m=None, f=0):
        """
        Multi-Krum aggregation rule - averages the m gradients closest to their neighbors
        m: number of gradients to average (default: n-f-2)
        f: number of Byzantine clients to tolerate
        """
        logging.info("[Server] Using Multi-Krum aggregation")
        
        # Get all client gradients as vectors
        deltas = [c.getDelta() for c in clients]
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        
        n = len(vecs)
        if f == 0:
            f = min(n // 3, 2)  # Default: tolerate up to n/3 Byzantine clients
        if m is None:
            m = n - f - 2  # Default: average all non-Byzantine gradients
        
        m = max(1, min(m, n))  # Ensure m is valid
        
        logging.info(f"[Multi-Krum] n={n} clients, f={f} Byzantine tolerance, m={m} gradients to average")
        
        # Compute pairwise distances
        distances = torch.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(vecs[i] - vecs[j], p=2)
                distances[i, j] = dist
                distances[j, i] = dist
        
        # For each client, compute sum of distances to closest n-f-2 neighbors
        scores = torch.zeros(n)
        for i in range(n):
            # Sort distances for client i (excluding distance to self)
            client_distances = distances[i].clone()
            client_distances[i] = float('inf')  # Exclude self
            sorted_distances, _ = torch.sort(client_distances)
            
            # Sum of distances to closest n-f-2 neighbors
            scores[i] = torch.sum(sorted_distances[:n-f-2])
        
        # Select top m clients with lowest scores
        _, selected_indices = torch.topk(scores, k=m, largest=False)
        selected_clients = [clients[i] for i in selected_indices]
        
        logging.info(f"[Multi-Krum] Selected clients {selected_indices.tolist()} with scores {scores[selected_indices].tolist()}")
        
        # Average the selected gradients
        return self.FedAvg(selected_clients)

    def foolsGold(self, clients):
        """
        FoolsGold aggregation rule - weights clients based on gradient similarity history
        """
        logging.info("[Server] Using FoolsGold aggregation")
        
        # Initialize similarity matrix if not exists
        if not hasattr(self, 'foolsgold_history'):
            self.foolsgold_history = []
            self.foolsgold_weights = torch.ones(len(clients))
        
        # Get current gradients
        deltas = [c.getDelta() for c in clients]
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        
        n = len(vecs)
        
        # Store current gradients in history
        self.foolsgold_history.append(torch.stack(vecs))
        
        # Limit history size to prevent memory issues
        if len(self.foolsgold_history) > 10:
            self.foolsgold_history.pop(0)
        
        # Compute cosine similarities between clients across history
        similarities = torch.zeros((n, n))
        
        for hist_gradients in self.foolsgold_history:
            for i in range(n):
                for j in range(i + 1, n):
                    cos_sim = F.cosine_similarity(
                        hist_gradients[i].unsqueeze(0),
                        hist_gradients[j].unsqueeze(0)
                    ).item()
                    similarities[i, j] += max(0, cos_sim)  # Only positive similarities
                    similarities[j, i] = similarities[i, j]
        
        # Compute FoolsGold weights
        for i in range(n):
            if len(self.foolsgold_history) > 1:
                # Weight inversely proportional to similarity with others
                self.foolsgold_weights[i] = 1.0 / (1.0 + torch.sum(similarities[i]) - similarities[i, i])
            else:
                self.foolsgold_weights[i] = 1.0
        
        # Normalize weights
        self.foolsgold_weights = self.foolsgold_weights / torch.sum(self.foolsgold_weights)
        
        logging.info(f"[FoolsGold] Weights: {self.foolsgold_weights.tolist()}")
        
        # Weighted average of gradients
        Delta = deepcopy(self.emptyStates)
        
        for param in Delta:
            weighted_param = torch.zeros_like(Delta[param])
            for i, client in enumerate(clients):
                client_delta = client.getDelta()
                if param in client_delta:
                    weighted_param += self.foolsgold_weights[i] * client_delta[param]
            Delta[param] = weighted_param
        
        return Delta

    def geometricMedian(self, clients):
        """
        Geometric Median aggregation rule - finds the point that minimizes sum of distances
        """
        logging.info("[Server] Using Geometric Median aggregation")
        
        deltas = [c.getDelta() for c in clients]
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        
        if not vecs:
            logging.warning("[GeometricMedian] No valid gradients found")
            return self.emptyStates
        
        # Weiszfeld's algorithm for geometric median
        X = torch.stack(vecs)  # Shape: (n_clients, n_params)
        n, d = X.shape
        
        # Initialize with coordinate-wise median
        y = torch.median(X, dim=0)[0]
        
        max_iter = 100
        tolerance = 1e-6
        
        for iteration in range(max_iter):
            # Compute distances from current estimate to all points
            distances = torch.norm(X - y.unsqueeze(0), dim=1)
            
            # Avoid division by zero
            distances = torch.clamp(distances, min=1e-8)
            
            # Compute weights (inverse distances)
            weights = 1.0 / distances
            weights = weights / torch.sum(weights)
            
            # Update estimate
            y_new = torch.sum(weights.unsqueeze(1) * X, dim=0)
            
            # Check convergence
            if torch.norm(y_new - y) < tolerance:
                logging.info(f"[GeometricMedian] Converged after {iteration+1} iterations")
                break
            
            y = y_new
        else:
            logging.warning(f"[GeometricMedian] Did not converge after {max_iter} iterations")
        
        # Convert back to parameter format
        Delta = utils.vec2net(y, self.emptyStates)
        return Delta

    def residualBase(self, clients):
        """
        Residual Base aggregation rule - removes outliers based on residual analysis
        """
        logging.info("[Server] Using Residual Base aggregation")
        
        deltas = [c.getDelta() for c in clients]
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        
        if len(vecs) < 3:
            logging.warning("[ResidualBase] Not enough clients for residual analysis, using FedAvg")
            return self.FedAvg(clients)
        
        X = torch.stack(vecs)  # Shape: (n_clients, n_params)
        n = len(X)
        
        # Compute pairwise distances
        distances = torch.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(X[i] - X[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Compute residuals (sum of distances to all other clients)
        residuals = torch.sum(distances, dim=1)
        
        # Remove outliers based on residual threshold
        median_residual = torch.median(residuals)
        mad = torch.median(torch.abs(residuals - median_residual))  # Median Absolute Deviation
        threshold = median_residual + 2.5 * mad  # Conservative threshold
        
        # Select non-outlier clients
        valid_indices = torch.where(residuals <= threshold)[0]
        
        if len(valid_indices) == 0:
            logging.warning("[ResidualBase] All clients marked as outliers, using all clients")
            valid_indices = torch.arange(n)
        
        logging.info(f"[ResidualBase] Selected {len(valid_indices)}/{n} clients (removed {n - len(valid_indices)} outliers)")
        
        # Average valid clients
        Delta = deepcopy(self.emptyStates)
        
        valid_clients = [clients[i] for i in valid_indices]
        
        for param in Delta:
            param_sum = torch.zeros_like(Delta[param])
            for client in valid_clients:
                client_delta = client.getDelta()
                if param in client_delta:
                    param_sum += client_delta[param]
            Delta[param] = param_sum / len(valid_clients)
        
        return Delta

    def net_attention(self, clients):
        """
        Attention-based aggregation - weights clients based on attention mechanism
        """
        logging.info("[Server] Using Attention aggregation")
        
        deltas = [c.getDelta() for c in clients]
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        
        if not vecs:
            logging.warning("[Attention] No valid gradients found")
            return self.emptyStates
        
        X = torch.stack(vecs)  # Shape: (n_clients, n_params)
        n, d = X.shape
        
        # Simple attention mechanism
        # Compute attention weights based on gradient magnitudes and similarities
        grad_norms = torch.norm(X, dim=1)
        
        # Attention based on inverse of gradient norm (smaller updates get more attention)
        attention_weights = 1.0 / (grad_norms + 1e-8)
        attention_weights = torch.softmax(attention_weights, dim=0)
        
        logging.info(f"[Attention] Weights: {attention_weights.tolist()}")
        
        # Weighted average
        weighted_vec = torch.sum(attention_weights.unsqueeze(1) * X, dim=0)
        
        # Convert back to parameter format
        Delta = utils.vec2net(weighted_vec, self.emptyStates)
        return Delta

    def net_mlp(self, clients):
        """
        MLP-based aggregation - uses simple neural network approach for aggregation
        """
        logging.info("[Server] Using MLP aggregation")
        
        deltas = [c.getDelta() for c in clients]
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        
        if not vecs:
            logging.warning("[MLP] No valid gradients found")
            return self.emptyStates
        
        X = torch.stack(vecs)  # Shape: (n_clients, n_params)
        n, d = X.shape
        
        # Simple MLP-like aggregation: compute weighted sum based on gradient statistics
        # Weight based on gradient norm and diversity
        grad_norms = torch.norm(X, dim=1)
        
        # Compute pairwise similarities
        similarities = torch.zeros(n)
        for i in range(n):
            sim_sum = 0
            for j in range(n):
                if i != j:
                    sim = F.cosine_similarity(X[i].unsqueeze(0), X[j].unsqueeze(0)).item()
                    sim_sum += abs(sim)
            similarities[i] = sim_sum / (n - 1) if n > 1 else 0
        
        # Combine norm and diversity for weights
        # Prefer moderate norms and lower similarity (more diverse)
        norm_weights = torch.exp(-torch.abs(grad_norms - torch.median(grad_norms)))
        diversity_weights = torch.exp(-similarities)
        
        mlp_weights = norm_weights * diversity_weights
        mlp_weights = mlp_weights / torch.sum(mlp_weights)
        
        logging.info(f"[MLP] Weights: {mlp_weights.tolist()}")
        
        # Weighted average
        weighted_vec = torch.sum(mlp_weights.unsqueeze(1) * X, dim=0)
        
        # Convert back to parameter format
        Delta = utils.vec2net(weighted_vec, self.emptyStates)
        return Delta

        ## Helper functions, act as adaptor from aggregation function to the federated learning system##

    def add_mal_id(self, sus_flip_sign, sus_uAtk, sus_tAtk):
        all_suspicious = sus_flip_sign.union(sus_uAtk, sus_tAtk)
        for i in range(self.num_clients):
            if i not in all_suspicious:
                if self.pre_mal_id[i] == 0:
                    if i in self.mal_ids:
                        self.mal_ids.remove(i)
                    if i in self.flip_sign_ids:
                        self.flip_sign_ids.remove(i)
                    if i in self.uAtk_ids:
                        self.uAtk_ids.remove(i)
                    if i in self.tAtk_ids:
                        self.tAtk_ids.remove(i)
                else: #> 0
                    self.pre_mal_id[i] = 0
                    # Unreliable clients:
                    if i in self.uAtk_ids:
                        self.count_unreliable[i] += 1
                        if self.count_unreliable[i] >= self.delay_decision:
                            self.uAtk_ids.remove(i)
                            self.mal_ids.remove(i)
                            self.unreliable_ids.add(i)
            else:
                self.pre_mal_id[i] += 1
                if self.pre_mal_id[i] >= self.delay_decision:
                    if i in sus_flip_sign:
                        self.flip_sign_ids.add(i)
                        self.mal_ids.add(i)
                    if i in sus_uAtk:
                        self.uAtk_ids.add(i)
                        self.mal_ids.add(i)
                if self.pre_mal_id[i] >= 2*self.delay_decision and i in sus_tAtk:
                    self.tAtk_ids.add(i)
                    self.mal_ids.add(i)

        logging.debug("mal_ids={}, pre_mal_id={}".format(self.mal_ids, self.pre_mal_id))
        #logging.debug("Count_unreliable={}".format(self.count_unreliable))
        logging.info("FLIP-SIGN ATTACK={}".format(self.flip_sign_ids))
        logging.info("UNTARGETED ATTACK={}".format(self.uAtk_ids))
        logging.info("TARGETED ATTACK={}".format(self.tAtk_ids))

    def mud_hog(self, clients):
        # long_HoGs for clustering targeted and untargeted attackers
        # and for calculating angle > 90 for flip-sign attack
        long_HoGs = {}

        # normalized_sHoGs for calculating angle > 90 for flip-sign attack
        normalized_sHoGs = {}
        full_norm_short_HoGs = [] # for scan flip-sign each round

        # L2 norm short HoGs are for detecting additive noise,
        # or Gaussian/random noise untargeted attack
        short_HoGs = {}

        # STAGE 1: Collect long and short HoGs.
        for i in range(self.num_clients):
            # longHoGs
            sum_hog_i = clients[i].get_sum_hog().detach().cpu().numpy()
            L2_sum_hog_i = clients[i].get_L2_sum_hog().detach().cpu().numpy()
            long_HoGs[i] = sum_hog_i

            # shortHoGs
            sHoG = clients[i].get_avg_grad().detach().cpu().numpy()
            #logging.debug(f"sHoG={sHoG.shape}") # model's total parameters, cifar=sHoG=(11191262,)
            L2_sHoG = np.linalg.norm(sHoG)
            full_norm_short_HoGs.append(sHoG/L2_sHoG)
            short_HoGs[i] = sHoG

            # Exclude the firmed malicious clients
            if i not in self.mal_ids:
                normalized_sHoGs[i] = sHoG/L2_sHoG

        # STAGE 2: Clustering and find malicious clients
        if self.iter >= self.tao_0:
            # STEP 1: Detect FLIP_SIGN gradient attackers
            """By using angle between normalized short HoGs to the median
            of normalized short HoGs among good candidates.
            NOTE: we tested finding flip-sign attack with longHoG, but it failed after long running.
            """
            flip_sign_id = set()
            """
            median_norm_shortHoG = np.median(np.array([v for v in normalized_sHoGs.values()]), axis=0)
            for i, v in enumerate(full_norm_short_HoGs):
                dot_prod = np.dot(median_norm_shortHoG, v)
                if dot_prod < 0: # angle > 90
                    flip_sign_id.add(i)
                    #logging.debug("Detect FLIP_SIGN client={}".format(i))
            logging.info(f"flip_sign_id={flip_sign_id}")
            """
            non_mal_sHoGs = dict(short_HoGs) # deep copy dict
            for i in self.mal_ids:
                non_mal_sHoGs.pop(i)
            median_sHoG = np.median(np.array(list(non_mal_sHoGs.values())), axis=0)
            for i, v in short_HoGs.items():
                #logging.info(f"median_sHoG={median_sHoG}, v={v}")
                v = np.array(list(v))
                d_cos = np.dot(median_sHoG, v)/(np.linalg.norm(median_sHoG)*np.linalg.norm(v))
                if d_cos < 0: # angle > 90
                    flip_sign_id.add(i)
                    #logging.debug("Detect FLIP_SIGN client={}".format(i))
            logging.info(f"flip_sign_id={flip_sign_id}")


            # STEP 2: Detect UNTARGETED ATTACK
            """ Exclude sign-flipping first, the remaining nodes include
            {NORMAL, ADDITIVE-NOISE, TARGETED and UNRELIABLE}
            we use DBSCAN to cluster them on raw gradients (raw short HoGs),
            the largest cluster is normal clients cluster (C_norm). For the remaining raw gradients,
            compute their Euclidean distance to the centroid (mean or median) of C_norm.
            Then find the bi-partition of these distances, the group of smaller distances correspond to
            unreliable, the other group correspond to additive-noise (Assumption: Additive-noise is fairly
            large (since it is attack) while unreliable's noise is fairly small).
            """

            # Step 2.1: excluding sign-flipping nodes from raw short HoGs:
            logging.info("===========using shortHoGs for detecting UNTARGETED ATTACK====")
            for i in range(self.num_clients):
                if i in flip_sign_id or i in self.flip_sign_ids:
                    short_HoGs.pop(i)
            id_sHoGs, value_sHoGs = np.array(list(short_HoGs.keys())), np.array(list(short_HoGs.values()))
            # Find eps for MNIST and CIFAR:
            """
            dist_1 = {}
            for k,v in short_HoGs.items():
                if k != 1:
                    dist_1[k] = np.linalg.norm(v - short_HoGs[1])
                    logging.info(f"Euclidean distance between 1 and {k} is {dist_1[k]}")

            logging.info(f"Average Euclidean distances between 1 and others {np.mean(list(dist_1.values()))}")
            logging.info(f"Median Euclidean distances between 1 and others {np.median(list(dist_1.values()))}")
            """

            # DBSCAN is mandatory success for this step, KMeans failed.
            # MNIST uses default eps=0.5, min_sample=5
            # CIFAR uses eps=50, min_sample=5 (based on heuristic evaluation Euclidean distance of grad of RestNet18.
            start_t = time.time()
            cluster_sh = DBSCAN(eps=self.dbscan_eps, n_jobs=-1,
                min_samples=self.dbscan_min_samples).fit(value_sHoGs)
            t_dbscan = time.time() - start_t
            #logging.info(f"CLUSTER DBSCAN shortHoGs took {t_dbscan}[s]")
            # TODO: comment out this line
            logging.info("labels cluster_sh= {}".format(cluster_sh.labels_))
            offset_normal_ids = find_majority_id(cluster_sh)
            normal_ids = id_sHoGs[list(offset_normal_ids)]
            normal_sHoGs = value_sHoGs[list(offset_normal_ids)]
            normal_cent = np.median(normal_sHoGs, axis=0)
            logging.debug(f"offset_normal_ids={offset_normal_ids}, normal_ids={normal_ids}")

            # suspicious ids of untargeted attacks and unreliable or targeted attacks.
            offset_uAtk_ids = np.where(cluster_sh.labels_ == -1)[0]
            sus_uAtk_ids = id_sHoGs[list(offset_uAtk_ids)]
            logging.info(f"SUSPECTED UNTARGETED {sus_uAtk_ids}")

            # suspicious_ids consists both additive-noise, targeted and unreliable clients:
            suspicious_ids = [i for i in id_sHoGs if i not in normal_ids] # this includes sus_uAtk_ids
            logging.debug(f"suspicious_ids={suspicious_ids}")
            d_normal_sus = {} # distance from centroid of normal to suspicious clients.
            for sid in suspicious_ids:
                d_normal_sus[sid] = np.linalg.norm(short_HoGs[sid]-normal_cent)

            # could not find separate points only based on suspected untargeted attacks.
            #d_sus_uAtk_values = [d_normal_sus[i] for i in sus_uAtk_ids]
            #d_separate = find_separate_point(d_sus_uAtk_values)
            d_separate = find_separate_point(list(d_normal_sus.values()))
            logging.debug(f"d_normal_sus={d_normal_sus}, d_separate={d_separate}")
            sus_tAtk_uRel_id0, uAtk_id = set(), set()
            for k, v in d_normal_sus.items():
                if v > d_separate and k in sus_uAtk_ids:
                    uAtk_id.add(k)
                else:
                    sus_tAtk_uRel_id0.add(k)
            logging.info(f"This round UNTARGETED={uAtk_id}, sus_tAtk_uRel_id0={sus_tAtk_uRel_id0}")


            # STEP 3: Detect TARGETED ATTACK
            """
              - First excluding flip_sign and untargeted attack from.
              - Using KMeans (K=2) based on Euclidean distance of
                long_HoGs==> find minority ids.
            """
            for i in range(self.num_clients):
                if i in self.flip_sign_ids or i in flip_sign_id:
                    if i in long_HoGs:
                        long_HoGs.pop(i)
                if i in uAtk_id or i in self.uAtk_ids:
                    if i in long_HoGs:
                        long_HoGs.pop(i)

            # Using Euclidean distance is as good as cosine distance (which used in MNIST).
            logging.info("=======Using LONG HOGs for detecting TARGETED ATTACK========")
            tAtk_id = find_targeted_attack(long_HoGs)

            # Aggregate, count and record ATTACKERs:
            self.add_mal_id(flip_sign_id, uAtk_id, tAtk_id)
            logging.info("OVERTIME MALICIOUS client ids ={}".format(self.mal_ids))

            # STEP 4: UNRELIABLE CLIENTS
            """using normalized short HoGs (normalized_sHoGs) to detect unreliable clients
            1st: remove all malicious clients (manipulate directly).
            2nd: find angles between normalized_sHoGs to the median point
            which mostly normal point and represent for aggreation (e.g., Median method).
            3rd: find confident mid-point. Unreliable clients have larger angles
            or smaller cosine similarities.
            """
            """
            for i in self.mal_ids:
                if i in normalized_sHoGs:
                    normalized_sHoGs.pop(i)

            angle_normalized_sHoGs = {}
            # update this value again after excluding malicious clients
            median_norm_shortHoG = np.median(np.array(list(normalized_sHoGs.values())), axis=0)
            for i, v in normalized_sHoGs.items():
                angle_normalized_sHoGs[i] = np.dot(median_norm_shortHoG, v)

            angle_sep_nsH = find_separate_point(list(angle_normalized_sHoGs.values()))
            normal_id, uRel_id = set(), set()
            for k, v in angle_normalized_sHoGs.items():
                if v < angle_sep_nsH: # larger angle, smaller cosine similarity
                    uRel_id.add(k)
                else:
                    normal_id.add(k)
            """
            for i in self.mal_ids:
                if i in short_HoGs:
                    short_HoGs.pop(i)

            angle_sHoGs = {}
            # update this value again after excluding malicious clients
            median_sHoG = np.median(np.array(list(short_HoGs.values())), axis=0)
            for i, v in short_HoGs.items():
                angle_sHoGs[i] = np.dot(median_sHoG, v)/(np.linalg.norm(median_sHoG)*np.linalg.norm(v))

            angle_sep_sH = find_separate_point(list(angle_sHoGs.values()))
            normal_id, uRel_id = set(), set()
            for k, v in angle_sHoGs.items():
                if v < angle_sep_sH: # larger angle, smaller cosine similarity
                    uRel_id.add(k)
                else:
                    normal_id.add(k)
            logging.info(f"This round UNRELIABLE={uRel_id}, normal_id={normal_id}")
            #logging.debug(f"anlge_normalized_sHoGs={angle_normalized_sHoGs}, angle_sep_nsH={angle_sep_nsH}")
            logging.debug(f"anlge_sHoGs={angle_sHoGs}, angle_sep_nsH={angle_sep_sH}")

            for k in range(self.num_clients):
                if k in uRel_id:
                    self.count_unreliable[k] += 1
                    if self.count_unreliable[k] > self.delay_decision:
                        self.unreliable_ids.add(k)
                # do this before decreasing count
                if self.count_unreliable[k] == 0 and k in self.unreliable_ids:
                    self.unreliable_ids.remove(k)
                if k not in uRel_id and self.count_unreliable[k] > 0:
                    self.count_unreliable[k] -= 1
            logging.info("UNRELIABLE clients ={}".format(self.unreliable_ids))

            normal_clients = []
            for i, client in enumerate(clients):
                if i not in self.mal_ids and i not in tAtk_id and i not in uAtk_id:
                    normal_clients.append(client)
            self.normal_clients = normal_clients
        else:
            normal_clients = clients
        out = self.FedFuncWholeNet(normal_clients, lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        return out

    def FedFuncWholeNet(self, clients, func):
        '''
        The aggregation rule views the update vectors as stacked vectors (1 by d by n).
        '''
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        # size is relative to number of samples, actually it is number of batches
        sizes = [c.get_data_size() for c in clients]
        total_s = sum(sizes)
        logging.info(f"clients' sizes={sizes}, total={total_s}")
        weights = [s/total_s for s in sizes]
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        weighted_vecs = [w*v for w,v in zip(weights, vecs)]
        result = func(torch.stack(vecs, 1).unsqueeze(0))  # input as 1 by d by n
        result = result.view(-1)
        utils.vec2net(result, Delta)
        return Delta

    def FedFuncWholeStateDict(self, clients, func):
        '''
        The aggregation rule views the update vectors as a set of state dict.
        '''
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        # sanity check, remove update vectors with nan/inf values
        deltas = [delta for delta in deltas if torch.isfinite(utils.net2vec(delta)).all().item()]

        resultDelta = func(deltas)

        Delta.update(resultDelta)
        return Delta


