import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import BallTree
import os
from utils import get_logger
import pandas as pd
from collections import defaultdict, Counter
import re
import unicodedata

logger = get_logger("Dataset")
MODEL_ID_OFFSET = 1

class POIProcessingConfig:
    def __init__(
        self,
        checkin_file,
        poi_file,
        dist_file,
        radius=200,
        max_candidates=50,
        device=None,
        noisy_value=50,
        transition_prior_matrix=None,
        transition_prior_weight=0.0,
    ):
        self.checkin_file = checkin_file
        self.poi_file = poi_file
        self.dist_file = dist_file
        self.radius = radius 
        self.max_candidates = max_candidates
        self.device = device
        self.noisy_value = noisy_value
        self.transition_prior_matrix = transition_prior_matrix
        self.transition_prior_weight = transition_prior_weight

class POIDataProcessor:
    def __init__(self, config):
        self.config = config
        
        self.venue_id2idx = {} 
        self.idx2venue_id = {}
        self.cat2idx = {}
        self.idx2cat = {}
        self.user2idx = {} 

        self.poi_coords = None      
        self.poi_cat_indices = None 
        self.cat_time_probs = None
        self.main_cat_probs = None
        self.df_checkin = None
        self.ball_tree = None
        self.num_pois = 0 
        
        data_dir = os.path.dirname(config.poi_file)
        self.cat_list_path = os.path.join(data_dir, 'category_list.txt') 
        self.category_path = os.path.join(data_dir, 'main_category.csv')
        self._load_and_build()
        # Keep processor-side lookup tensors on CPU so DataLoader workers can
        # safely fork/spawn without trying to re-initialize CUDA.
        self.main_cat_mapping = self.get_main_cat_mapping_tensor()

    def get_main_cat_mapping_tensor(self):
        if not self.idx2cat:
            raise ValueError("idx2cat not built")

        if not os.path.exists(self.category_path):
            raise FileNotFoundError(f"can not find CSV: {self.category_path}")

        def normalize(s):
            if not isinstance(s, str):
                s = str(s)
            s = unicodedata.normalize('NFKD', s)
            s = s.encode('ascii', 'ignore').decode('ascii')
            s = s.lower()
            s = re.sub(r'[^a-z0-9]', ' ', s)
            s = ' '.join(s.split())
            return s

        try:
            df = pd.read_csv(self.category_path, encoding='utf-8')
        except:
            df = pd.read_csv(self.category_path, encoding='latin-1')
        
        unique_main_names = sorted(df['main'].unique().tolist())
        self.main_cat2idx = {name: i for i, name in enumerate(unique_main_names)}
        self.idx2main_cat = {i: name for i, name in enumerate(unique_main_names)}
        main_name_to_id = self.main_cat2idx
        
        norm_sub_to_main_id = {}
        for _, row in df.iterrows():
            norm_name = normalize(row['original'])
            if norm_name:
                norm_sub_to_main_id[norm_name] = main_name_to_id[row['main']]

        num_small_cats = len(self.idx2cat)
        mapping_array = torch.full((num_small_cats,), -1, dtype=torch.long)

        for cat_id, cat_name in self.idx2cat.items():
            norm_cat_name = normalize(cat_name)
            if norm_cat_name in norm_sub_to_main_id:
                mapping_array[cat_id] = norm_sub_to_main_id[norm_cat_name]
            else:
                words = norm_cat_name.split()
                found = False
                if len(words) > 1:
                    short_name = ' '.join(words[:2])
                    if short_name in norm_sub_to_main_id:
                        mapping_array[cat_id] = norm_sub_to_main_id[short_name]
                        found = True
                if not found:
                    logger.error(f"Category '{cat_name}' (Normalized: '{norm_cat_name}') still not found.")
        
        self.num_main_cats = len(unique_main_names)
        self.main_cat_probs = np.zeros((self.num_main_cats, 24), dtype=np.float32)
        mapping_np = mapping_array.cpu().numpy()
        
        valid_mask = mapping_np != -1
        valid_small_indices = np.where(valid_mask)[0]
        valid_main_indices = mapping_np[valid_mask]
        
        np.add.at(self.main_cat_probs, valid_main_indices, self.cat_time_probs[valid_small_indices])
        col_sums = self.main_cat_probs.sum(axis=0, keepdims=True) + 1e-9
        self.main_cat_probs = self.main_cat_probs / col_sums
                
        return mapping_array
       
    def _load_and_build(self):
        logger.info(f"Loading raw data...")
        df_poi = pd.read_csv(self.config.poi_file)
        df_dist = pd.read_csv(self.config.dist_file)
        df_checkin = pd.read_csv(self.config.checkin_file)
        df_checkin = df_checkin.rename(columns={ 'userid':'user_id'})

        if os.path.exists(self.cat_list_path):
            with open(self.cat_list_path, 'r') as f:
                cat_names = [line.strip() for line in f if line.strip()]
        else:
            cat_names = sorted(df_poi['category'].dropna().unique())
            with open(self.cat_list_path, 'w') as f:
                for c in cat_names: f.write(f"{c}\n")
 
        self.num_cats = len(cat_names)
        self.cat2idx = {cat: i for i, cat in enumerate(cat_names)}
        self.idx2cat = {i: cat for i, cat in enumerate(cat_names)}
        num_cats = len(self.cat2idx)

        self.venue_id2idx = {vid: i for i, vid in enumerate(df_poi['venue_id'])}
        self.idx2venue_id = {i: vid for i, vid in enumerate(df_poi['venue_id'])}
        self.num_pois = len(df_poi)
        
        unique_users = sorted(df_checkin['user_id'].unique())
        self.user2idx = {uid: i for i, uid in enumerate(unique_users)}
        
        self.poi_cat_indices = np.zeros(self.num_pois, dtype=int)
        unknown_cat_idx = 0 
        for i, row in df_poi.iterrows():
            cat = row['category']
            self.poi_cat_indices[i] = self.cat2idx.get(cat, unknown_cat_idx)
            
        self.poi_coords = np.radians(df_poi[['latitude', 'longitude']].values).astype(np.float32)
        self.cat_time_probs = np.zeros((num_cats, 24), dtype=np.float32)
        dist_map = df_dist.set_index('category')
        prob_cols = df_dist.columns[1:]
        for i, cat_name in enumerate(cat_names):
            if cat_name in dist_map.index:
                self.cat_time_probs[i] = dist_map.loc[cat_name, prob_cols].values.astype(np.float32)

        df_checkin = df_checkin[df_checkin['venue_id'].isin(self.venue_id2idx)].copy()
        df_checkin['user_idx_mapped'] = df_checkin['user_id'].map(self.user2idx)
        df_checkin['venue_idx_mapped'] = df_checkin['venue_id'].map(self.venue_id2idx)
        df_checkin['local_datetime'] = pd.to_datetime(df_checkin['local_datetime'])
        df_checkin['hour'] = df_checkin['local_datetime'].dt.hour
        df_checkin['dayofweek'] = df_checkin['local_datetime'].dt.dayofweek
        base_slot = df_checkin['hour'] // 2
        is_weekend = (df_checkin['dayofweek'] >= 5).astype(int)
        df_checkin['time_slot'] = base_slot + (is_weekend * 12)
        
        self.df_checkin = df_checkin.reset_index(drop=True)
        self.ball_tree = BallTree(self.poi_coords, metric='haversine')

    def get_small_cat_id_by_poi_id(self, poi_id):
        """
        根据 poi_id（这里应是 venue_idx / 全局 POI index）获取 small category id
        """
        poi_id = int(poi_id)
        if poi_id < 0 or poi_id >= len(self.poi_cat_indices):
            raise IndexError(f"poi_id out of range: {poi_id}")
        return int(self.poi_cat_indices[poi_id])


    def get_small_cat_name_by_poi_id(self, poi_id):
        """
        根据 poi_id 获取 small category name
        """
        small_cat_id = self.get_small_cat_id_by_poi_id(poi_id)
        return self.idx2cat.get(small_cat_id, "Unknown")


    def get_main_cat_id_by_poi_id(self, poi_id):
        """
        根据 poi_id 获取 main category id
        利用:
            self.main_cat_mapping[small_cat_id] -> main_cat_id
        """
        small_cat_id = self.get_small_cat_id_by_poi_id(poi_id)
        main_cat_id = int(self.main_cat_mapping[small_cat_id].item())
        return main_cat_id


    def get_main_cat_name_by_poi_id(self, poi_id):
        """
        根据 poi_id 获取 main category name
        """
        main_cat_id = self.get_main_cat_id_by_poi_id(poi_id)
        return self.idx2main_cat.get(main_cat_id, "Unknown")
    
    def get_main_cat_info_by_poi_id(self, poi_id):
        """
        返回:
            {
                "poi_id": ...,
                "small_cat_id": ...,
                "small_cat_name": ...,
                "main_cat_id": ...,
                "main_cat_name": ...
            }
        """
        small_cat_id = self.get_small_cat_id_by_poi_id(poi_id)
        small_cat_name = self.idx2cat.get(small_cat_id, "Unknown")
        main_cat_id = int(self.main_cat_mapping[small_cat_id].item())
        main_cat_name = self.idx2main_cat.get(main_cat_id, "Unknown")

        return {
            "poi_id": int(poi_id),
            "small_cat_id": small_cat_id,
            "small_cat_name": small_cat_name,
            "main_cat_id": main_cat_id,
            "main_cat_name": main_cat_name,
        }

class CheckinSequenceDataset(Dataset):
    def __init__(self, processor, max_seq_len=50, mode='train'): 
        self.processor = processor
        self.config = processor.config
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.radius_rad = self.config.radius / 6371000.0
        
        # 格式: {(sample_idx, time_step): {"main_cat_id": int, "main_cat_name": str, "confidence": float}}
        self.easy_pseudo_labels = {}
        df = self.processor.df_checkin.copy()
        # ==========================================================
        # 先按“用户-日期”统计一天打卡数，过滤掉少于 3 条的天
        # ==========================================================
        df['date'] = pd.to_datetime(df['local_datetime']).dt.date

        day_counts = df.groupby(['user_idx_mapped', 'date']).size().reset_index(name='num_checkins')
        valid_days = day_counts[day_counts['num_checkins'] >= 3][['user_idx_mapped', 'date']]

        df = df.merge(valid_days, on=['user_idx_mapped', 'date'], how='inner')

        logger.info(
            f"[{mode.upper()}] After filtering user-day checkins < 3, remaining rows: {len(df)}"
        )

        df = df.sort_values(
            by=['user_idx_mapped', 'local_datetime'],
            ascending=[True, True]
        )

        self.sorted_df = df.reset_index(drop=True)
        grouped = self.sorted_df.groupby('user_idx_mapped').indices
                
        
        
        
        
        self.traj_groups = []
        train_row_indices = [] 
        
        for uid in sorted(grouped.keys()):
            user_indices = grouped[uid] 
            n = len(user_indices)
            if n == 0: continue
            
            n_train = int(n * 0.8)
            n_val = int(n * 0.1)
            
            if n_train == 0: n_train = 1
            if n_val == 0 and n > 2: n_val = 1
            
            train_idxs = user_indices[:n_train]
            val_idxs = user_indices[n_train : n_train + n_val]
            test_idxs = user_indices[n_train + n_val :]
            
            train_row_indices.extend(train_idxs)
            
            if mode == 'train': target_idxs = train_idxs
            elif mode == 'val': target_idxs = val_idxs
            elif mode == 'test': target_idxs = test_idxs
            else: target_idxs = user_indices
                
            for i in range(0, len(target_idxs), self.max_seq_len):
                chunk = target_idxs[i : i + self.max_seq_len]
                if len(chunk) > 0:
                    self.traj_groups.append(chunk)
                    
                    

        
        logger.info(f"[{mode.upper()}] Formed {len(self.traj_groups)} continuous trajectory chunks.")

        # ==========================================================
        # 核心改动 1：为全局所有数据一次性生成并固化【加噪坐标】
        # ==========================================================
        logger.info(f"Generating global noisy coordinates (Noise={self.config.noisy_value}m)...")
        all_true_pois = self.sorted_df['venue_idx_mapped'].values
        all_true_coords = self.processor.poi_coords[all_true_pois] 
        
        num_all_points = len(all_true_coords)
        noise_dist = np.sqrt(np.random.random(num_all_points)) * self.config.noisy_value / 6371000.0
        noise_theta = np.random.random(num_all_points) * 2 * np.pi
        
        delta_lat = noise_dist * np.cos(noise_theta)
        delta_lon = noise_dist * np.sin(noise_theta) / np.cos(all_true_coords[:, 0])
        
        # 固化保存！__getitem__ 将直接从这里切片读取
        self.all_noisy_coords = np.stack([
            all_true_coords[:, 0] + delta_lat, 
            all_true_coords[:, 1] + delta_lon
        ], axis=1).astype(np.float32)

        # ==========================================================
        # 4. 构建历史频率特征 (基于固化后的加噪坐标！)
        # ==========================================================
        logger.info(f"Building User-Level Cand Frequency Matrix...")
        
        # 获取所有【加噪后坐标】的周围候选集 (保证与模型看到的绝对一致)
        cands_list = self.processor.ball_tree.query_radius(self.all_noisy_coords, r=self.radius_rad)
        self.all_radius_candidates = cands_list
        
        self.user_cand_freq = {}
        self.user_total_cands = {}
        
        # 按用户对 train_row_indices 进行分组
        user_train_indices = defaultdict(list)
        for idx in train_row_indices:
            uid = self.sorted_df.at[idx, 'user_idx_mapped']
            user_train_indices[uid].append(idx)
            
        # 统计每个用户的历史(Train)访问偏好
        for uid in grouped.keys():
            indices = user_train_indices.get(uid, [])
            if len(indices) > 0:
                all_cands_for_user = np.concatenate(cands_list[indices])
                self.user_cand_freq[uid] = Counter(all_cands_for_user)
                self.user_total_cands[uid] = len(indices) + 1e-9 
            else:
                self.user_cand_freq[uid] = Counter()
                self.user_total_cands[uid] = 1e-9
        
       
        
        
        


    def __len__(self):
        return len(self.traj_groups)

    def _get_prev_small_cat_id(self, row_indices, traj_df, true_poi_indices, t):
        if t > 0:
            return int(self.processor.poi_cat_indices[true_poi_indices[t - 1]])

        curr_row_idx = int(row_indices[t])
        if curr_row_idx <= 0:
            return -1

        prev_row = self.sorted_df.iloc[curr_row_idx - 1]
        curr_user = int(traj_df.iloc[t]['user_idx_mapped'])
        if int(prev_row['user_idx_mapped']) != curr_user:
            return -1

        prev_poi_id = int(prev_row['venue_idx_mapped'])
        return int(self.processor.poi_cat_indices[prev_poi_id])

    def _blend_transition_prior(self, base_probs, small_cat_ids, prev_small_cat_id):
        prior_matrix = self.config.transition_prior_matrix
        weight = float(getattr(self.config, "transition_prior_weight", 0.0))
        if prior_matrix is None or weight <= 0.0 or prev_small_cat_id < 0:
            return base_probs

        transition_probs = prior_matrix[prev_small_cat_id, small_cat_ids].astype(np.float32)
        return ((1.0 - weight) * base_probs) + (weight * transition_probs)
    
    def __getitem__(self, idx):
        row_indices = self.traj_groups[idx]
        traj_df = self.sorted_df.iloc[row_indices]
        
        seq_len = len(traj_df)
        max_cand = self.config.max_candidates
        
        user_ids = traj_df['user_idx_mapped'].values.astype(np.int64)
        time_slots = traj_df['time_slot'].values.astype(np.int64)
        true_poi_indices = traj_df['venue_idx_mapped'].values
        
        center_coords = self.all_noisy_coords[row_indices]
        
        # true_coords 依然保留，用于求 label 或 Debug (如果不参与后续计算也可以删)
        # true_coords = self.processor.poi_coords[true_poi_indices]
        
       
        
        
        
        # true_coords = self.processor.poi_coords[true_poi_indices]
        
        # # Noise
        # noise_dist = np.sqrt(np.random.random(seq_len)) * self.config.noisy_value / 6371000.0
        # noise_theta = np.random.random(seq_len) * 2 * np.pi
        # delta_lat = noise_dist * np.cos(noise_theta)
        # delta_lon = noise_dist * np.sin(noise_theta) / np.cos(true_coords[:, 0])
        # center_coords = np.stack([true_coords[:, 0] + delta_lat, true_coords[:, 1] + delta_lon], axis=1).astype(np.float32)
        
        # Init Containers
        seq_cand_ids = np.zeros((seq_len, max_cand), dtype=int)
        seq_cand_cats = np.zeros((seq_len, max_cand), dtype=int)
        seq_cand_main_cats = np.zeros((seq_len, max_cand), dtype=int)
        seq_cand_probs = np.zeros((seq_len, max_cand), dtype=np.float32)
        seq_cand_main_probs = np.zeros((seq_len, max_cand), dtype=np.float32)
        seq_cand_mask = np.zeros((seq_len, max_cand), dtype=int)
        seq_cand_dists = np.zeros((seq_len, max_cand), dtype=np.float32)
        seq_cand_feats = np.zeros((seq_len, max_cand, 1), dtype=np.float32) 
        
        seq_neg_ids = np.zeros((seq_len, max_cand), dtype=int)
        seq_neg_cats = np.zeros((seq_len, max_cand), dtype=int)
        seq_neg_main_cats = np.zeros((seq_len, max_cand), dtype=int)
        seq_neg_probs = np.zeros((seq_len, max_cand), dtype=np.float32)
        seq_neg_main_probs = np.zeros((seq_len, max_cand), dtype=np.float32)
        seq_neg_dists = np.zeros((seq_len, max_cand), dtype=np.float32)
        seq_neg_feats = np.zeros((seq_len, max_cand, 1), dtype=np.float32)
        
        seq_true_labels = np.zeros(seq_len, dtype=int)
        seq_true_global_ids = true_poi_indices
        # ======== 提取 LLM 缓存标签 ========
        # seq_llm_labels = np.full(seq_len, -1, dtype=int)
        
        uid = user_ids[0]
        user_freq_counter = self.user_cand_freq.get(uid, Counter())
        user_total_visits = self.user_total_cands.get(uid, 1.0)
        
        if torch.is_tensor(self.processor.main_cat_mapping):
            main_cat_mapping_np = self.processor.main_cat_mapping.cpu().numpy()
        else:
            main_cat_mapping_np = self.processor.main_cat_mapping

        for t in range(seq_len):
            # 获取 LLM 标签
            # llm_key = (idx, t)
            # if llm_key in self.llm_pseudo_labels:
            #     seq_llm_labels[t] = self.llm_pseudo_labels[llm_key]
                
            t_true_id = true_poi_indices[t]
            
           
            curr_center_rad = center_coords[t].reshape(1, -1) 
            t_slot = time_slots[t]
            prev_small_cat_id = self._get_prev_small_cat_id(row_indices, traj_df, true_poi_indices, t)
            
            # --- Candidates ---
            cands = np.array(self.all_radius_candidates[row_indices[t]], copy=True)
            if t_true_id not in cands: cands = np.append(cands, t_true_id)
            if len(cands) > max_cand:
                sel = np.random.choice(cands[cands != t_true_id], max_cand - 1, replace=False)
                cands = np.append(sel, t_true_id)
            np.random.shuffle(cands)
            
            seq_true_labels[t] = np.where(cands == t_true_id)[0][0]
            curr_k = len(cands)
            
            seq_cand_ids[t, :curr_k] = cands + MODEL_ID_OFFSET
            seq_cand_cats[t, :curr_k] = self.processor.poi_cat_indices[cands] + 1
            
            small_cat_ids = self.processor.poi_cat_indices[cands]
            main_cat_ids = main_cat_mapping_np[small_cat_ids]
            seq_cand_main_cats[t, :curr_k] = main_cat_ids + 1
            
            cand_probs = self.processor.cat_time_probs[small_cat_ids, t_slot].astype(np.float32)
            cand_probs = self._blend_transition_prior(cand_probs, small_cat_ids, prev_small_cat_id)
            seq_cand_probs[t, :curr_k] = cand_probs
            valid_main_mask = (main_cat_ids != -1)
            main_probs = np.zeros(curr_k, dtype=np.float32)
            if valid_main_mask.any():
                valid_main_ids = main_cat_ids[valid_main_mask]
                main_probs[valid_main_mask] = self.processor.main_cat_probs[valid_main_ids, t_slot]
            seq_cand_main_probs[t, :curr_k] = main_probs
            
            seq_cand_mask[t, :curr_k] = 1
            dists_rad = np.linalg.norm(self.processor.poi_coords[cands] - curr_center_rad, axis=1)
            seq_cand_dists[t, :curr_k] = np.log1p(dists_rad * 6371000.0)
            
            raw_recur_counts = np.array([user_freq_counter.get(c, 0) for c in cands], dtype=np.float32)
            recurrence_counts = np.maximum(0, raw_recur_counts - 1) 
            seq_cand_feats[t, :curr_k, 0] = np.log1p(recurrence_counts) / np.log1p(user_total_visits) 
            
            # --- Negatives ---
            negs = []
            while len(negs) < max_cand:
                ridx = np.random.randint(0, self.processor.num_pois)
                if ridx not in cands and ridx not in negs: negs.append(ridx)
            negs = np.array(negs[:max_cand])
            
            seq_neg_ids[t, :] = negs + MODEL_ID_OFFSET
            seq_neg_cats[t, :] = self.processor.poi_cat_indices[negs] + 1
            
            neg_small_cat_ids = self.processor.poi_cat_indices[negs]
            neg_main_cat_ids = main_cat_mapping_np[neg_small_cat_ids]
            seq_neg_main_cats[t, :] = neg_main_cat_ids + 1
            
            neg_probs = self.processor.cat_time_probs[neg_small_cat_ids, t_slot].astype(np.float32)
            neg_probs = self._blend_transition_prior(neg_probs, neg_small_cat_ids, prev_small_cat_id)
            seq_neg_probs[t, :] = neg_probs
            valid_neg_main_mask = (neg_main_cat_ids != -1)
            neg_main_probs = np.zeros(max_cand, dtype=np.float32)
            if valid_neg_main_mask.any():
                valid_neg_main_ids = neg_main_cat_ids[valid_neg_main_mask]
                neg_main_probs[valid_neg_main_mask] = self.processor.main_cat_probs[valid_neg_main_ids, t_slot]
            seq_neg_main_probs[t, :] = neg_main_probs
            
            neg_coords = self.processor.poi_coords[negs]
            neg_dists = np.linalg.norm(neg_coords - curr_center_rad, axis=1) * 6371000.0
            seq_neg_dists[t, :] = np.log1p(neg_dists)
            
            n_recur_counts = np.array([user_freq_counter.get(n, 0) for n in negs], dtype=np.float32)
            seq_neg_feats[t, :, 0] = np.log1p(n_recur_counts) / np.log1p(user_total_visits)

        seq_mask = np.ones(seq_len, dtype=bool)

        return {
            'user_id': torch.tensor(user_ids + MODEL_ID_OFFSET, dtype=torch.long),
            'time_slot': torch.tensor(time_slots + MODEL_ID_OFFSET, dtype=torch.long),
            'center_coord': torch.tensor(center_coords, dtype=torch.float32),
            'seq_mask': torch.tensor(seq_mask, dtype=torch.bool),
            
            'cand_poi_ids': torch.tensor(seq_cand_ids, dtype=torch.long),
            'cand_cat_ids': torch.tensor(seq_cand_cats, dtype=torch.long),
            'cand_main_cat_ids': torch.tensor(seq_cand_main_cats, dtype=torch.long),
            'cand_probs': torch.tensor(seq_cand_probs, dtype=torch.float32),
            'cand_main_cat_probs': torch.tensor(seq_cand_main_probs, dtype=torch.float32),
            'cand_mask': torch.tensor(seq_cand_mask, dtype=torch.float32),
            'cand_dists': torch.tensor(seq_cand_dists, dtype=torch.float32),
            'cand_other_feats': torch.tensor(seq_cand_feats, dtype=torch.float32), 
            
            'neg_poi_ids': torch.tensor(seq_neg_ids, dtype=torch.long),
            'neg_cat_ids': torch.tensor(seq_neg_cats, dtype=torch.long),
            'neg_main_cat_ids': torch.tensor(seq_neg_main_cats, dtype=torch.long),
            'neg_probs': torch.tensor(seq_neg_probs, dtype=torch.float32),
            'neg_main_cat_probs': torch.tensor(seq_neg_main_probs, dtype=torch.float32),
            'neg_dists': torch.tensor(seq_neg_dists, dtype=torch.float32),
            'neg_other_feats': torch.tensor(seq_neg_feats, dtype=torch.float32),
            
            'label_pos': torch.tensor(seq_true_labels, dtype=torch.long),
            'true_poi_id': torch.tensor(seq_true_global_ids, dtype=torch.long),
            'sample_idx': torch.tensor(idx, dtype=torch.long) 
        }
        
def seq_collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    max_batch_len = max([item['user_id'].size(0) for item in batch])
    
    for key in keys:
        if key == 'sample_idx':
            collated[key] = torch.stack([item[key] for item in batch])
            continue
            
        padded_tensors = []
        for item in batch:
            tensor = item[key]
            seq_len = tensor.size(0)
            pad_len = max_batch_len - seq_len
            
            if pad_len > 0:
                # ====== 保证 llm_label padding 为 -1 ======
                if key in ['label_pos', 'true_poi_id']:
                    pad_val = -1
                elif key == 'seq_mask':
                    pad_val = False
                else:
                    pad_val = 0
                    
                pad_shape = list(tensor.shape)
                pad_shape[0] = pad_len
                pad_tensor = torch.full(pad_shape, pad_val, dtype=tensor.dtype, device=tensor.device)
                padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
            else:
                padded_tensor = tensor
                
            padded_tensors.append(padded_tensor)
            
        collated[key] = torch.stack(padded_tensors)
        
    return collated
