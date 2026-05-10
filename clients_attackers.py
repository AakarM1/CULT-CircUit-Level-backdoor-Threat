from __future__ import print_function
import math
import random

import torch
import torch.nn.functional as F
import logging
import numpy as np
from sklearn.decomposition import PCA
import statistics

from utils import utils
from utils.backdoor_semantic_utils import SemanticBackdoor_Utils
from utils.backdoor_utils import Backdoor_Utils
from clients import *
from utils.blur_images import GaussianSmoothing
import inspect

class Unreliable_client(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
            device='cpu', mean=0, max_std=0.5, fraction_noise=0.5,
            fraction_train=0.3, blur_method='gaussian_smooth', inner_epochs=1,
            channels=1, kernel_size=5):
        logging.info("init UNRELIABLE Client {}".format(cid))
        super(Unreliable_client, self).__init__(cid, model, dataLoader,
            optimizer, criterion, device, inner_epochs)
        self.max_std = max_std
        self.mean = mean
        self.unreliable_fraction = fraction_noise
        self.fraction_train=fraction_train
        self.seed = 0
        self.channels = channels
        self.kernel_size = kernel_size
        self.blur_method = blur_method

    def data_transform(self, data, target):
        if torch.rand(1) < self.unreliable_fraction:
            if self.blur_method == 'add_noise':
                # APPROACH 1: simple add noise
                torch.manual_seed(self.seed)
                std = torch.rand(data.shape)*self.max_std
                gaussian = torch.normal(mean=self.mean, std=std)
                assert data.shape == gaussian.shape, "Inconsistent Gaussian noise shape"
                data_ = data + gaussian
            else: # gaussian_smooth
                # APPROACH 2: Gaussian smoothing
                smoothing = GaussianSmoothing(self.channels, self.kernel_size, self.max_std)
                data_ = F.pad(data, (2,2,2,2), mode='reflect')
                data_ = smoothing(data_)
        else:
            data_ = data
        self.seed += 1

        return data_, target

    def train(self):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.inner_epochs):
            for batch_idx, (data, target) in enumerate(self.dataLoader):
                if torch.rand(1) > self.fraction_train: #just train 30% of local dataset
                    continue
                data, target = self.data_transform(data, target)
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        self.isTrained = True
        self.model.cpu()  ## avoid occupying gpu when idle


#class Attacker_label_change_all_to_7(Client):
class Attacker_MultiLabelFlipping(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
            device='cpu', inner_epochs=1, source_labels=[1,2,3], target_label=7):
        logging.info(f"init ATTACK MULTI-LABEL-FLIPPING, change label {source_labels} to {target_label} Client {cid}")
        super(Attacker_MultiLabelFlipping, self).__init__(cid, model,
            dataLoader, optimizer, criterion, device, inner_epochs)
        self.source_labels = source_labels
        self.target_label = target_label

    def data_transform(self, data, target):
        #target_ = torch.ones(target.shape, dtype=int)*7 # for all labels -->7
        target_ = torch.tensor(list(map(lambda x: self.target_label if x in self.source_labels else x, target)))
        assert target.shape == target_.shape, "Inconsistent target shape"
        return data, target_

class Attacker_LabelFlipping1to7(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
            device='cpu', inner_epochs=1, source_label=1, target_label=7):
        super(Attacker_LabelFlipping1to7, self).__init__(cid, model, dataLoader,
            optimizer, criterion, device, inner_epochs)
        self.source_label = source_label
        self.target_label = target_label
        logging.info(f"init ATTACK LABEL Change from {source_label} to {target_label} Client {cid}")

    def data_transform(self, data, target):
        target_ = torch.tensor(list(map(lambda x: self.target_label if x == self.source_label else x, target)))
        assert target.shape == target_.shape, "Inconsistent target shape"
        return data, target_


class Attacker_LabelFlipping01swap(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss, device='cpu', inner_epochs=1):
        super(Attacker_LabelFlipping01swap, self).__init__(cid, model, dataLoader, optimizer, criterion, device,
                                                           inner_epochs)

    def data_transform(self, data, target):
        target_ = torch.tensor(list(map(lambda x: 1 - x if x in [0, 1] else x, target)))
        assert target.shape == target_.shape, "Inconsistent target shape"
        return data, target_


class Attacker_Backdoor(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
            device='cpu', inner_epochs=1):
        super(Attacker_Backdoor, self).__init__(cid, model, dataLoader,
            optimizer, criterion, device, inner_epochs)
        self.utils = Backdoor_Utils()
        logging.info("init BACKDOOR ATTACK Client {}".format(cid))

    def data_transform(self, data, target):
        data, target = self.utils.get_poison_batch(data, target,
            backdoor_fraction=0.5, backdoor_label=self.utils.backdoor_label)
        return data, target

    #def testBackdoor(self):
    #    self.model.to(self.device)
    #    self.model.eval()
    #    test_loss = 0
    #    correct = 0
    #    utils = SemanticBackdoor_Utils()
    #    with torch.no_grad():
    #        for data, target in self.dataLoader:
    #            data, target = self.utils.get_poison_batch(data, target,
    #                backdoor_fraction=1.0,
    #                backdoor_label=self.utils.backdoor_label, evaluation=True)
    #            data, target = data.to(self.device), target.to(self.device)
    #            output = self.model(data)
    #            test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
    #            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #            correct += pred.eq(target.view_as(pred)).sum().item()

    #    test_loss /= len(self.dataLoader.dataset)
    #    accuracy = 100. * correct / len(self.dataLoader.dataset)

    #    self.model.cpu()  ## avoid occupying gpu when idle
    #    logging.info('(Testing at the attacker) Test set (Backdoored):'
    #        ' Average loss: {}, Success rate: {}/{} ({}%)'.format(
    #            test_loss, correct, len(self.dataLoader.dataset), accuracy))

    #def update(self):
    #    super().update()
    #    self.testBackdoor()


class Attacker_SemanticBackdoor(Client):
    '''
    suggested by 'How to backdoor Federated Learning'
    https://arxiv.org/pdf/1807.00459.pdf

    For each batch, 20 out of 64 samples (in the original paper) are replaced with semantic backdoor, this implementation replaces on average a 30% of the batch by the semantic backdoor

    '''

    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.cross_entropy, device='cpu', inner_epochs=1):
        super(Attacker_SemanticBackdoor, self).__init__(cid, model, dataLoader, optimizer, criterion, device,
                                                        inner_epochs)
        self.utils = SemanticBackdoor_Utils()

    def data_transform(self, data, target):
        data, target = self.utils.get_poison_batch(data, target, backdoor_fraction=0.3,
                                                   backdoor_label=self.utils.backdoor_label)
        return data, target

    def testBackdoor(self):
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        utils = SemanticBackdoor_Utils()
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = self.utils.get_poison_batch(data, target, backdoor_fraction=1.0,
                                                           backdoor_label=self.utils.backdoor_label, evaluation=True)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataLoader.dataset)
        accuracy = 100. * correct / len(self.dataLoader.dataset)

        self.model.cpu()  ## avoid occupying gpu when idle
        logging.info('(Testing at the attacker) Test set (Semantic Backdoored):'
            ' Average loss: {}, Success rate: {}/{} ({}%)\n'.format(
                test_loss, correct, len(self.dataLoader.dataset), accuracy))

    def update(self):
        super().update()
        self.testBackdoor()


class Attacker_Omniscient(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
            device='cpu', scale=1, inner_epochs=1):
        super(Attacker_Omniscient, self).__init__(cid, model, dataLoader,
            optimizer, criterion, device, inner_epochs)
        logging.info("init ATTACK OMNISCIENT Client {}".format(cid))
        self.scale = scale

    def update(self):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState = self.model.state_dict()
        trainable_parameter = utils.getTrainableParameters(self.model)
        for p in self.originalState:
            self.stateChange[p] = newState[p] - self.originalState[p]
            if p not in trainable_parameter:
                continue
            #             if not "FloatTensor" in self.originalState[p].type():
            #                 continue
            self.stateChange[p] *= (-self.scale)
            self.sum_hog[p] += self.stateChange[p]
            K_ = len(self.hog_avg)
            if K_ == 0:
                self.avg_delta[p] = self.stateChange[p]
            elif K_ < self.K_avg:
                self.avg_delta[p] = (self.avg_delta[p]*K_ + self.stateChange[p])/(K_+1)
            else:
                self.avg_delta[p] += (self.stateChange[p] - self.hog_avg[0][p])/self.K_avg

        self.hog_avg.append(self.stateChange)
        self.isTrained = False

class Attacker_AddNoise_Grad(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
            device='cpu', mean=0, std=0.1 ,inner_epochs=1):
        super(Attacker_AddNoise_Grad, self).__init__(cid, model, dataLoader,
            optimizer, criterion, device, inner_epochs)
        self.mean = mean
        self.std = std
        logging.info("init ATTACK ADD NOISE TO GRAD Client {}".format(cid))

    def update(self):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState = self.model.state_dict()
        trainable_parameter = utils.getTrainableParameters(self.model)
        for p in self.originalState:
            self.stateChange[p] = newState[p] - self.originalState[p]
            if p not in trainable_parameter:
                continue
            std = torch.ones(self.stateChange[p].shape)*self.std
            gaussian = torch.normal(mean=self.mean, std=std)
            self.stateChange[p] += gaussian
            self.sum_hog[p] += self.stateChange[p]
            K_ = len(self.hog_avg)
            if K_ == 0:
                self.avg_delta[p] = self.stateChange[p]
            elif K_ < self.K_avg:
                self.avg_delta[p] = (self.avg_delta[p]*K_ + self.stateChange[p])/(K_+1)
            else:
                self.avg_delta[p] += (self.stateChange[p] - self.hog_avg[0][p])/self.K_avg
        self.hog_avg.append(self.stateChange)
        self.isTrained = False

# Quantum Attacker Classes



class Attacker_QuantumTrojan(Client):
    """
    Advanced stealthy quantum backdoor with Adaptive Intensity control.

    Strategy:
      1. Continuously profile honest updates.
      2. After a poisoning round, calculate the raw malicious update.
      3. Measure the deviation (anomaly score) of this update against the honest profile.
      4. Dynamically set the blending factor 'epsilon' based on this score:
         - High anomaly -> Low epsilon (prioritize stealth).
         - Low anomaly  -> High epsilon (increase attack potency).
      5. Craft the final stealthy delta using the adaptive epsilon.
    """
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
                 device='cpu', inner_epochs=1,
                 attack_type='grover', trigger_params=None,
                 poison_frac=0.90, num_attackers=1):
        super().__init__(cid, model, dataLoader, optimizer, criterion, device, inner_epochs)
        self.client_id       = cid
        self.attack_type     = attack_type
        self.trigger_params  = trigger_params or []
        self.poison_frac     = poison_frac
        self.num_attackers   = num_attackers

        # --- Stealth & Control Parameters ---
        self.round_index     = 0
        self.burn_in_rounds  = 0

        # Adaptive Intensity Parameters
        # Mudhog: 0.01->0.50
        self.max_eps         = 20  # The absolute maximum epsilon allowed
        self.min_eps         = 10  # A floor for epsilon to ensure some learning
        
        # Tempering (still useful)
        # Mudhog: 0.85
        self.loss_scale_factor = 2

        # History & Projection
        # Mudhog: 100, 5
        self.history_size    = 100
        self.k_components    = 1

        # Camouflage
        # Mudhog: 1e-5, 0.05
        self.gauss_scale     = 0
        self.sparsity_frac   = 0.005

        # Internal state
        self.honest_history  = []
        self.original_state  = {}

        # Extension: S2 toggle + per-batch quantum measurement deviation tracking
        self.disable_s2 = False
        self.last_z_dev = None  # mean ||z_attack - z_clean||_2 over last training pass

        logging.info(f"[Attacker {cid}] initialized with Adaptive Intensity strategy.")

    def train(self):
        self.round_index += 1
        
        self.original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        clean_q = self.model.q_layer
        q_layer_factory = getattr(self.model, f"q_layer_{self.attack_type}")
        sig = inspect.signature(q_layer_factory)
        trojan_q = q_layer_factory(self.trigger_params) if len(sig.parameters) == 1 else q_layer_factory(*self.trigger_params)

        self.model.to(self.device).train()
        poisoned_batches = 0
        total_batches = 0
        
        # Decide upfront if this round will involve any poisoning
        is_poisoning_round = (self.round_index > self.burn_in_rounds) and (random.random() <= self.poison_frac)

        # Track quantum-measurement deviation ||z_attack - z_clean||_2 per batch
        z_dev_running = []

        for _ in range(self.inner_epochs):
            for x, y in self.dataLoader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # Poisoning is decided at the round level, but applied per batch for learning
                if is_poisoning_round:
                    self.model.q_layer = trojan_q
                    poisoned_batches += 1
                    try:
                        with torch.no_grad():
                            self.model.q_layer = clean_q
                            z_clean = self.model(x).detach()
                            self.model.q_layer = trojan_q
                            z_atk = self.model(x).detach()
                            z_dev_running.append(torch.norm(z_atk - z_clean, p=2, dim=-1).mean().item())
                    except Exception:
                        pass
                else:
                    self.model.q_layer = clean_q

                out = self.model(x)
                loss = self.criterion(out, y)

                if is_poisoning_round:
                    loss = loss * self.loss_scale_factor

                loss.backward()
                self.optimizer.step()
                total_batches += 1
        
        self.model.q_layer = clean_q

        self.last_z_dev = float(sum(z_dev_running) / len(z_dev_running)) if z_dev_running else 0.0

        phase = "POISONING" if is_poisoning_round else "HONEST"
        logging.info(f"[Attacker {self.client_id}] round {self.round_index} ({phase}): "
                     f"poisoned {poisoned_batches}/{total_batches} batches.")
        self.isTrained = True
        self.model.cpu()

    def getDelta(self):
        new_s = self.model.state_dict()
        raw = {k: new_s[k] - self.original_state[k] for k in self.original_state}
        flat_raw = self._flatten(raw).detach()

        # S1-only mode: skip update crafting entirely
        if getattr(self, 'disable_s2', False):
            return raw

        # In burn-in or if history is too small, act honestly and build history
        if self.round_index <= self.burn_in_rounds or len(self.honest_history) < 10:
            if flat_raw.norm() > 1e-6:
                self.honest_history.append(flat_raw.clone())
                if len(self.honest_history) > self.history_size:
                    self.honest_history.pop(0)
            return raw

        is_poisoning_round = (random.random() <= self.poison_frac) # Re-roll for delta decision
        if not is_poisoning_round:
            # Still update history on honest rounds
            if flat_raw.norm() > 1e-6:
                self.honest_history.append(flat_raw.clone())
                if len(self.honest_history) > self.history_size:
                    self.honest_history.pop(0)
            return raw

        # --- Adaptive Intensity Crafting ---
        H = torch.stack(self.honest_history)
        
        # 1. Calculate anomaly score (using L2 norm Z-score for simplicity and robustness)
        honest_norms = torch.norm(H, dim=1)
        mean_norm = honest_norms.mean()
        std_norm = honest_norms.std()
        if std_norm < 1e-6: std_norm = mean_norm * 0.1 # Avoid division by zero
        
        raw_norm = flat_raw.norm()
        # Anomaly score: how many std deviations away is the raw norm?
        anomaly_score = torch.abs(raw_norm - mean_norm) / std_norm
        
        # 2. Dynamically set epsilon based on anomaly score
        # Using an inverse relationship: eps = max_eps / (1 + anomaly_score)
        eps = self.max_eps / (1 + anomaly_score.item())
        eps = max(self.min_eps, eps) # Ensure epsilon doesn't become zero
        
        logging.info(f"[Attacker {self.client_id}] Anomaly score: {anomaly_score:.2f}. Adaptive epsilon: {eps:.4f}")

        # 3. Proceed with stealth crafting using the adaptive epsilon
        dists = torch.norm(H - flat_raw.unsqueeze(0), dim=1)
        h_star = H[torch.argmin(dists)]
        r = flat_raw - h_star

        try:
            H_mean = H.mean(dim=0)
            _, _, Vh = torch.linalg.svd(H - H_mean, full_matrices=False)
        except torch.linalg.LinAlgError:
            return raw # Fallback

        r_null = r.clone()
        k = min(self.k_components, Vh.shape[0])
        for i in range(k):
            pc = Vh[i, :]
            r_null -= torch.dot(r_null, pc) * pc
        
        p_flat = h_star + eps * r_null

        # Norm matching and camouflage remain important
        target_norm = abs(random.gauss(mean_norm, std_norm / 2.0))
        current_norm = torch.norm(p_flat)
        if current_norm > 1e-8:
            p_flat = p_flat * (target_norm / current_norm)

        p_flat += torch.randn_like(p_flat) * self.gauss_scale * target_norm
        
        abs_p = torch.abs(p_flat)
        threshold = torch.kthvalue(abs_p, int(self.sparsity_frac * p_flat.numel()))[0]
        p_flat[abs_p < threshold] = 0.0

        return self._unflatten(p_flat, raw)
        
    def _flatten(self, d):
        return torch.cat([v.flatten() for v in d.values()])

    def _unflatten(self, flat, template):
        out, idx = {}, 0
        for k, v in template.items():
            n = v.numel()
            out[k] = flat[idx:idx+n].reshape(v.shape).to(v.device)
            idx += n
        return out





class Attacker_GroverBackdoor(Client):
    """Grover‐Phase‐Kickback Trojan, fires only on trigger == marked_state."""
    def __init__(self, cid, model, dataLoader, optimizer,
                 criterion=F.nll_loss, device='cpu',
                 inner_epochs=1, num_attackers=1):
        super().__init__(cid, model, dataLoader, optimizer, criterion, device, inner_epochs)
        logging.info(f"init ATTACK GROVER BACKDOOR Client {cid}")
        # 4‑bit trigger pattern:
        self.marked_state   = [1, 0, 1, 0]
        # poison 1/K of the batches
        self.poison_frac    = 0.4
        self.num_attackers = num_attackers
        # backdoor label (unused here, but available if you want label‐flipping)
        self.utils         = Backdoor_Utils()
        self.bd_label      = self.utils.backdoor_label

    def train(self):
        """Train locally, swapping in the Trojan QNode on 1/K of batches."""
        clean_q = self.model.q_layer
        noisy_q = self.model.q_layer_grover(self.marked_state)

        self.model.to(self.device)
        self.model.train()
        K = max(1, int(1/self.poison_frac))

        attacked = 0
        total = 0
        for epoch in range(self.inner_epochs):
            for batch_idx, (data, target) in enumerate(self.dataLoader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()

                # inject the Trojan circuit occasionally
                if batch_idx % K == 0:
                    self.model.q_layer = noisy_q
                    attacked+=1
                else:
                    self.model.q_layer = clean_q
                total += 1
                output = self.model(data)
                loss   = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        logging.info(f"Attacker {self.cid} trained on {attacked}/{total} batches with Grover backdoor.")
        self.isTrained = True
        self.model.cpu()

    def getDelta(self):
        """Grab the raw delta and quietly scale it down for stealth."""
        raw = super().getDelta()
        factor = 25.0 * float(self.num_attackers)
        return {k: v / factor for k, v in raw.items()}


class Attacker_NoiseTrojan(Client):
    """Noise Trojan with partial‐batch injection for stealth."""
    def __init__(self, cid, model, dataLoader, optimizer,
                 criterion=F.nll_loss, device='cpu',
                 inner_epochs=1, num_attackers=1):
        super().__init__(cid, model, dataLoader, optimizer, criterion, device, inner_epochs)
        logging.info(f"init ATTACK NOISE BACKDOOR Client {cid}")
        # Trigger parameters
        self.trigger_state = [0.2, 1.3, 2.0, 0.5]
        self.marked_state = [1, 0, 1, 0]
        # poison fraction of the batches
        self.poison_frac = 0.05
        self.num_attackers = num_attackers
        # backdoor label
        self.utils = Backdoor_Utils()
        self.bd_label = self.utils.backdoor_label

    def train(self):
        """Train locally, swapping in the Trojan QNode on 1/K of batches."""
        clean_q = self.model.q_layer
        noisy_q = self.model.q_layer_grover(self.marked_state)

        self.model.to(self.device)
        self.model.train()
        K = max(1, int(1/self.poison_frac))

        attacked = 0
        total = 0
        for epoch in range(self.inner_epochs):
            for batch_idx, (data, target) in enumerate(self.dataLoader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()

                # inject the Trojan circuit occasionally
                if batch_idx % K == 0:
                    self.model.q_layer = noisy_q
                    attacked+=1
                else:
                    self.model.q_layer = clean_q
                total += 1
                output = self.model(data)
                loss   = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        logging.info(f"Attacker {self.cid} trained on {attacked}/{total} batches with Grover backdoor.")
        self.isTrained = True
        self.model.cpu()

    def getDelta(self):
        """Grab the raw delta and quietly scale it down for stealth."""
        raw = super().getDelta()
        factor = 25.0 * float(self.num_attackers)
        return {k: v / factor for k, v in raw.items()}
            
class Attacker_BitFlipTrojan(Client):
    """QFT‐Period‐Finding Bit‐Flip Trojan"""
    def __init__(self, cid, model, dataLoader, optimizer, criterion, device, inner_epochs, num_attackers):
        super().__init__(cid, model, dataLoader, optimizer, criterion, device, inner_epochs)
        self.period = 3
        self.marked_state = [1,0,1,0]  # Add marked state
        self.poison_frac = 0.05  
        self.num_attackers = num_attackers
        self.bd_label    = Backdoor_Utils().backdoor_label
    def train(self):
        # Prepare clean vs. noisy QNodes
        clean_q = self.model.q_layer                     # original clean layer
        noisy_q = self.model.q_layer_bitflip(self.period, self.marked_state)  # Fix: pass both parameters

        self.model.to(self.device)
        self.model.train()

        num_batches = len(self.dataLoader)
        # compute every Kth batch to poison
        K = max(1, int(1/self.poison_frac))

        for epoch in range(self.inner_epochs):
            for batch_idx, (data, target) in enumerate(self.dataLoader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()

                # Inject the trojan in 1/K of the batches
                if batch_idx % K == 0:
                    self.model.q_layer = noisy_q
                else:
                    self.model.q_layer = clean_q

                output = self.model(data)
                if batch_idx % K == 0:
                    poison_targets = torch.full_like(
                        target, fill_value=self.bd_label, dtype=torch.long, device=self.device
                    )
                    loss = self.criterion(output, poison_targets)
                else:
                    loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

        self.isTrained = True
        self.model.cpu()
        
    def getDelta(self):
        raw = super().getDelta()           # this is a dict: {param_name: tensor, …}
        # Increase stealth by scaling gradients more aggressively
        return {k: v / (20*float(self.num_attackers)) for k, v in raw.items()}

    def data_transform(self, data, target):
        # poison a fraction of inputs and force them to the global backdoor label
        return Backdoor_Utils().get_poison_batch(
            data,
            target,
            backdoor_fraction=self.poison_frac,
            backdoor_label=self.bd_label  # Fix: use self.bd_label
        )
        

class Attacker_SignFlipTrojan(Client):
    """Phase‐Estimation Sign‐Flip Trojan"""
    def __init__(self, cid, model, dataLoader, optimizer, criterion, device, inner_epochs, num_attackers):
        super().__init__(cid, model, dataLoader, optimizer, criterion, device, inner_epochs)
        self.phase = 0.25  # fraction of 2π
        self.marked_state = [1,0,1,0]  # Add marked state
        self.poison_frac = 0.05  
        self.num_attackers = num_attackers
        self.bd_label    = Backdoor_Utils().backdoor_label

    def train(self):
        # Prepare clean vs. noisy QNodes
        clean_q = self.model.q_layer                     # original clean layer
        noisy_q = self.model.q_layer_signflip(self.phase, self.marked_state)  # Fix: pass both parameters

        self.model.to(self.device)
        self.model.train()

        num_batches = len(self.dataLoader)
        # compute every Kth batch to poison
        K = max(1, int(1/self.poison_frac))

        for epoch in range(self.inner_epochs):
            for batch_idx, (data, target) in enumerate(self.dataLoader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()

                # Inject the trojan in 1/K of the batches
                if batch_idx % K == 0:
                    self.model.q_layer = noisy_q
                else:
                    self.model.q_layer = clean_q

                output = self.model(data)
                if batch_idx % K == 0:
                    poison_targets = torch.full_like(
                        target, fill_value=self.bd_label, dtype=torch.long, device=self.device
                    )
                    loss = self.criterion(output, poison_targets)
                else:
                    loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

        self.isTrained = True
        self.model.cpu()
        
    def getDelta(self):
        raw = super().getDelta()           # this is a dict: {param_name: tensor, …}
        # Increase stealth by scaling gradients more conservatively
        return {k: v / (1 + (0 * float(self.num_attackers))) for k, v in raw.items()}

    def data_transform(self, data, target):
        # poison a fraction of inputs and force them to the global backdoor label
        return Backdoor_Utils().get_poison_batch(
            data,
            target,
            backdoor_fraction=self.poison_frac,
            backdoor_label=self.bd_label  # Fix: use self.bd_label
        )