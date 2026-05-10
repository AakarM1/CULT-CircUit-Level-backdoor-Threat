"""
FLGuardian Defense: Layer-wise Anomaly Detection for Federated Learning

Reference: https://github.com/jiejiesks/FLGuardian
Adapted for Quantum Federated Learning (QFL)

Algorithm Overview:
1. Layer-wise Separation: Deconstruct client updates into separate layers/blocks
2. Pairwise Analysis: For each layer, compute pairwise cosine and Euclidean distances
3. Clustering: Use K-Means to identify benign cluster for each layer
4. Trust Scoring: Assign trust scores based on benign layer classification with layer weighting
5. Filtering: Select top-k clients by trust score for aggregation
"""

import torch
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from typing import Dict, List, Tuple, Set
from copy import deepcopy


class FLGuardianDefense:
    """
    FLGuardian defense mechanism for detecting poisoning attacks in FL.
    
    Attributes:
        n_clusters: Number of clusters for K-Means (default: 2 for benign vs. malicious)
        use_cosine: Whether to use cosine distance (True) or Euclidean (False)
        use_cosine_and_euclidean: If True, use both distances in voting mechanism
        layer_weighting: 'uniform', 'linear', or 'quadratic' for layer importance
        trust_threshold: Minimum trust score to include client (0-1)
        top_k: If set, select only top-k clients by trust score instead of threshold
        random_state: For reproducibility of K-Means
    """
    
    def __init__(
        self,
        n_clusters: int = 2,
        use_cosine_and_euclidean: bool = True,
        layer_weighting: str = 'quadratic',
        trust_threshold: float = 0.5,
        top_k: int = None,
        random_state: int = 42
    ):
        """
        Initialize FLGuardian defense.
        
        Args:
            n_clusters: Number of clusters for K-Means clustering
            use_cosine_and_euclidean: Use both cosine and Euclidean distances
            layer_weighting: 'uniform', 'linear', or 'quadratic' weighting scheme
            trust_threshold: Minimum trust score to keep client (ignored if top_k is set)
            top_k: Select only top-k clients; if None, use trust_threshold
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.use_cosine_and_euclidean = use_cosine_and_euclidean
        self.layer_weighting = layer_weighting
        self.trust_threshold = trust_threshold
        self.top_k = top_k
        self.random_state = random_state
        
        self.layer_weights = None
        self.suspicious_clients = set()
        self.trust_scores = {}
        
    def _extract_layers(self, state_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        Extract parameter layers from state_dict.
        In quantum/classical models, each parameter is treated as a layer/block.
        
        Args:
            state_dict: Model state_dict (parameter updates)
            
        Returns:
            Dictionary mapping layer name to flattened tensor
        """
        layers = {}
        for param_name, param_tensor in state_dict.items():
            # Flatten each parameter to 1D vector
            layers[param_name] = param_tensor.flatten().detach().cpu().numpy()
        return layers
    
    def _compute_pairwise_distances(
        self,
        layer_vectors: Dict[int, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pairwise cosine and Euclidean distances for a single layer.
        
        Args:
            layer_vectors: Dictionary mapping client_id to flattened layer vector
            
        Returns:
            Tuple of (cosine_distances, euclidean_distances) matrices
        """
        client_ids = sorted(layer_vectors.keys())
        vectors = np.array([layer_vectors[cid] for cid in client_ids])
        
        # Handle edge cases
        if vectors.shape[0] < 2:
            return np.array([[0]]), np.array([[0]])
        
        # Compute distances
        cosine_dist = cosine_distances(vectors)
        euclidean_dist = euclidean_distances(vectors)
        
        return cosine_dist, euclidean_dist, client_ids
    
    def _cluster_layer(
        self,
        cosine_dist: np.ndarray,
        euclidean_dist: np.ndarray,
        client_ids: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform K-Means clustering on a single layer using distance matrices.
        
        Args:
            cosine_dist: Pairwise cosine distance matrix
            euclidean_dist: Pairwise Euclidean distance matrix
            client_ids: List of client IDs
            
        Returns:
            Tuple of (labels, benign_client_set)
        """
        if len(client_ids) < self.n_clusters:
            # Not enough clients to cluster
            return np.zeros(len(client_ids)), set(client_ids)
        
        # Use Euclidean distances as primary metric for clustering
        try:
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(euclidean_dist)
        except Exception as e:
            logging.warning(f"[FLGuardian] K-Means clustering failed: {e}, treating all as benign")
            return np.zeros(len(client_ids)), set(client_ids)
        
        # Find the majority cluster (benign clients)
        unique, counts = np.unique(labels, return_counts=True)
        benign_label = unique[np.argmax(counts)]
        benign_indices = np.where(labels == benign_label)[0]
        benign_clients = {client_ids[i] for i in benign_indices}
        
        return labels, benign_clients
    
    def _compute_layer_weights(self, n_layers: int) -> np.ndarray:
        """
        Compute layer importance weights.
        Deeper layers (closer to output) should have higher importance.
        
        Args:
            n_layers: Total number of layers
            
        Returns:
            Array of weights normalized to sum to 1
        """
        if self.layer_weighting == 'uniform':
            weights = np.ones(n_layers)
        elif self.layer_weighting == 'linear':
            # Linearly increasing weights
            weights = np.arange(1, n_layers + 1, dtype=float)
        elif self.layer_weighting == 'quadratic':
            # Quadratically increasing weights (emphasize deeper layers more)
            weights = np.arange(1, n_layers + 1, dtype=float) ** 2
        else:
            weights = np.ones(n_layers)
            logging.warning(f"[FLGuardian] Unknown weighting scheme '{self.layer_weighting}', using uniform")
        
        # Normalize to sum to 1
        weights = weights / np.sum(weights)
        return weights
    
    def _compute_trust_scores(
        self,
        client_ids: List[int],
        benign_votes: Dict[int, int],
        layer_weights: np.ndarray
    ) -> Dict[int, float]:
        """
        Compute weighted trust score for each client.
        
        Args:
            client_ids: List of all client IDs
            benign_votes: Dictionary mapping client_id to number of benign layer votes
            layer_weights: Weights for each layer
            
        Returns:
            Dictionary mapping client_id to trust score (0-1)
        """
        n_layers = len(layer_weights)
        trust_scores = {}
        
        for cid in client_ids:
            # Number of layers where this client was benign
            n_benign_layers = benign_votes.get(cid, 0)
            
            # Trust score = proportion of benign layers (weighted)
            # The weighting is implicit in how benign_votes is computed
            # across differently weighted layers
            trust_score = n_benign_layers / n_layers if n_layers > 0 else 0.0
            trust_scores[cid] = trust_score
        
        return trust_scores
    
    def detect_poisoned_clients(
        self,
        clients_updates: Dict[int, Dict]
    ) -> Set[int]:
        """
        Detect poisoned clients using layer-wise anomaly analysis.
        
        Args:
            clients_updates: Dictionary mapping client_id to state_dict (parameter updates)
            
        Returns:
            Set of suspicious client IDs to exclude from aggregation
        """
        if not clients_updates or len(clients_updates) < 2:
            logging.warning("[FLGuardian] Not enough clients for defense, returning empty suspicious set")
            self.suspicious_clients = set()
            return self.suspicious_clients
        
        client_ids = sorted(clients_updates.keys())
        n_clients = len(client_ids)
        
        # Extract all layers from all clients
        all_layers = {}
        for cid in client_ids:
            all_layers[cid] = self._extract_layers(clients_updates[cid])
        
        # Get layer names from first client
        layer_names = sorted(all_layers[client_ids[0]].keys())
        n_layers = len(layer_names)
        
        if n_layers == 0:
            logging.warning("[FLGuardian] No layers found in client updates")
            self.suspicious_clients = set()
            return self.suspicious_clients
        
        # Compute layer weights
        layer_weights = self._compute_layer_weights(n_layers)
        self.layer_weights = layer_weights
        
        # Track benign votes per client across layers
        benign_votes = {cid: 0 for cid in client_ids}
        
        logging.info(f"[FLGuardian] Analyzing {n_clients} clients across {n_layers} layers")
        
        # Process each layer
        for layer_idx, layer_name in enumerate(layer_names):
            layer_weight = layer_weights[layer_idx]
            
            # Collect vectors for this layer from all clients
            layer_vectors = {
                cid: all_layers[cid][layer_name]
                for cid in client_ids
            }
            
            # Compute pairwise distances
            cosine_dist, euclidean_dist, sorted_client_ids = self._compute_pairwise_distances(layer_vectors)
            
            # Cluster clients for this layer
            labels, benign_clients = self._cluster_layer(
                cosine_dist,
                euclidean_dist,
                sorted_client_ids
            )
            
            # Update benign vote count (weighted)
            for cid in benign_clients:
                benign_votes[cid] += layer_weight
            
            logging.debug(
                f"[FLGuardian] Layer {layer_idx} ({layer_name}): "
                f"benign_clients={benign_clients}, weight={layer_weight:.4f}"
            )
        
        # Compute trust scores
        self.trust_scores = self._compute_trust_scores(client_ids, benign_votes, layer_weights)
        
        # Select clients to keep
        if self.top_k is not None:
            # Keep top-k clients
            sorted_by_trust = sorted(
                self.trust_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            suspicious = {cid for cid, _ in sorted_by_trust[self.top_k:]}
            self.suspicious_clients = suspicious
            logging.info(
                f"[FLGuardian] Keeping top {self.top_k} clients: "
                f"{[cid for cid, _ in sorted_by_trust[:self.top_k]]}, "
                f"Suspicious: {suspicious}"
            )
        else:
            # Keep clients above trust threshold
            suspicious = {cid for cid, score in self.trust_scores.items()
                         if score < self.trust_threshold}
            self.suspicious_clients = suspicious
            logging.info(
                f"[FLGuardian] Trust threshold={self.trust_threshold}, "
                f"Removed {len(suspicious)} suspicious clients: {suspicious}"
            )
        
        # Log trust scores
        for cid in sorted(client_ids):
            logging.info(f"[FLGuardian] Client {cid}: trust_score={self.trust_scores[cid]:.4f}")
        
        return self.suspicious_clients
    
    def filter_clients(
        self,
        clients: List,
        suspicious_ids: Set[int]
    ) -> List:
        """
        Filter out suspicious clients from the client list.
        
        Args:
            clients: List of client objects with .cid attribute
            suspicious_ids: Set of client IDs to remove
            
        Returns:
            Filtered list of clients
        """
        filtered = [c for c in clients if c.cid not in suspicious_ids]
        logging.info(
            f"[FLGuardian] Filtered clients: {len(clients)} -> {len(filtered)} "
            f"(removed {len(suspicious_ids)} clients)"
        )
        return filtered
    
    def defend(
        self,
        clients: List,
        return_suspicious_only: bool = False
    ) -> List:
        """
        Main defense mechanism: detect and filter poisoned clients.
        
        Args:
            clients: List of client objects with .cid and .getDelta() method
            return_suspicious_only: If True, return suspicious clients instead of benign ones
            
        Returns:
            Filtered list of clients (benign if return_suspicious_only=False)
        """
        # Extract updates from clients
        clients_updates = {}
        for c in clients:
            try:
                delta = c.getDelta()
                if delta is not None:
                    clients_updates[c.cid] = delta
            except Exception as e:
                logging.warning(f"[FLGuardian] Could not get delta from client {c.cid}: {e}")
        
        # Detect suspicious clients
        suspicious_ids = self.detect_poisoned_clients(clients_updates)
        
        if return_suspicious_only:
            return [c for c in clients if c.cid in suspicious_ids]
        else:
            return self.filter_clients(clients, suspicious_ids)



