import os
import numpy as np
import pickle
from tqdm import tqdm


cpath = os.path.dirname(os.path.abspath(__file__))
NUM_USER = 100
SAVE = True
DATASET_ROOT = os.path.join(cpath, "data") 
alpha = 0.1

np.random.seed(6)


class PathDataset(object):

    def __init__(self, image_paths, labels):
        self.data = np.array(image_paths) 
        self.target = np.array(labels) 

    def __len__(self):
        return len(self.target)

def load_tinyimagenet_paths(dataset_root):
    if os.path.exists(os.path.join(dataset_root, "tiny-imagenet-200")):
        real_root = os.path.join(dataset_root, "tiny-imagenet-200")
    else:
        real_root = dataset_root

    train_dir = os.path.join(real_root, "train")
    val_dir = os.path.join(real_root, "val")
    wnids_path = os.path.join(real_root, "wnids.txt")
    if not os.path.exists(wnids_path):
        raise FileNotFoundError(f"未找到 wnids.txt, 请检查路径: {wnids_path}")

    with open(wnids_path, "r") as f:
        wnids = [w.strip() for w in f.readlines()]
    wnid_to_label = {wnid: idx for idx, wnid in enumerate(wnids)}

    # 2. 读取训练集路径
    print(">>> Collecting Train Paths...")
    train_paths, train_labels = [], []

    for wnid in tqdm(wnids, desc="Scanning Train"):
        image_folder = os.path.join(train_dir, wnid, "images")
        if not os.path.exists(image_folder):
            continue

        label = wnid_to_label[wnid]
        fnames = [f for f in os.listdir(image_folder) if f.endswith('.JPEG')]

        for fname in fnames:
            path = os.path.join(image_folder, fname)
            train_paths.append(path)
            train_labels.append(label)

    # 3. 读取验证集路径
    print(">>> Collecting Val Paths...")
    val_paths, val_labels = [], []
    anno_path = os.path.join(val_dir, "val_annotations.txt")
    val_img_dir = os.path.join(val_dir, "images")

    with open(anno_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.split()
        fname = parts[0]
        wnid = parts[1]

        label = wnid_to_label[wnid]
        path = os.path.join(val_img_dir, fname)

        val_paths.append(path)
        val_labels.append(label)

    print(f"Loaded {len(train_paths)} train images, {len(val_paths)} val images.")

    return PathDataset(train_paths, train_labels), \
        PathDataset(val_paths, val_labels), \
        len(wnids)


# -------------------------------------------------------------
# 主函数
# -------------------------------------------------------------
def main():
    print(f">>> Root: {DATASET_ROOT}")

    # 1. 加载数据
    trainset, testset, num_classes = load_tinyimagenet_paths(DATASET_ROOT)

    # 2. 生成 Dirichlet 分布
    print(f">>> Generating Non-IID distribution (alpha={alpha})...")
    min_size = 0
    K = num_classes
    N = len(trainset)

    # === 安全保护 ===
    cnt = 0
    while min_size < 10:
        if cnt > 50:
            print(">>> Warning: Random generation timed out. Breaking loop.")
            break
        cnt += 1

        idx_batch = [[] for _ in range(NUM_USER)]
        for k in range(K):
            idx_k = np.where(trainset.target == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, NUM_USER))
            proportions = np.array([p * (len(idx_j) < N / NUM_USER) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch_split = np.split(idx_k, proportions)
            for u in range(NUM_USER):
                idx_batch[u] += idx_batch_split[u].tolist()

        min_size = min([len(idx_j) for idx_j in idx_batch])
        print(f"    - Try {cnt}: Min size: {min_size}")

    # 3. 分配数据
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # 测试集索引切分 (Uniform)
    all_test_idxs = np.arange(len(testset))
    np.random.shuffle(all_test_idxs)
    test_idxs_split = np.array_split(all_test_idxs, NUM_USER)

    # ============================================================
    # 🟢 关键修改：使用整数 i 作为 User ID，而不是 str(i)
    # ============================================================
    print(">>> Allocating Data (Using Integer User IDs)...")

    for i in range(NUM_USER):
        uname = i  # ✅ 这里直接使用整数 (int)

        # --- 训练集 ---
        train_idxs = idx_batch[i]
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {
            'x': trainset.data[train_idxs].tolist(),
            'y': trainset.target[train_idxs].tolist()
        }
        train_data['num_samples'].append(len(train_idxs))

        # --- 测试集 ---
        test_idxs = test_idxs_split[i]
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {
            'x': testset.data[test_idxs].tolist(),
            'y': testset.target[test_idxs].tolist()
        }
        test_data['num_samples'].append(len(test_idxs))

    # ============================================================

    # 4. 保存
    if SAVE:
        print('>>> Saving data to pkl...')

        common_filename = f'tinyimagenet_niid_{alpha}.pkl'

        train_path = os.path.join(cpath, 'data', 'train', common_filename)
        test_path = os.path.join(cpath, 'data', 'test', common_filename)

        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)

        with open(train_path, "wb") as f:
            pickle.dump(train_data, f)

        with open(test_path, "wb") as f:
            pickle.dump(test_data, f)

        print(">>> Save Done.")
        print(f"    Train file: {train_path}")
        print(f"    Test file:  {test_path}")
        print(f"    New Dataset Name for main.py: tinyimagenet_niid_{alpha}")


if __name__ == '__main__':
    main()

