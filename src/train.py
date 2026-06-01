import os
import torch
from alphafold3_pytorch import Alphafold3, PDBDataset, Trainer
from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler
from lightning.fabric.loggers import TensorBoardLogger

# ==========================================
# 1. 하이퍼파라미터 및 경로 설정
# ==========================================
BATCH_SIZE = 2
NUM_TRAIN_STEPS = 100000  # 총 학습 스텝 수

# 우리가 정제한 데이터 폴더들
TRAIN_DIR = "./data/pdb_data/train_mmcifs"
VAL_DIR = "./data/pdb_data/val_mmcifs"
TEST_DIR = "./data/pdb_data/test_mmcifs"

# 클러스터링 결과 폴더
TRAIN_CLUSTERING_DIR = "./data/pdb_data/train_clustering"


# ==========================================
# 2. 클러스터링 기반 데이터 샘플러 (핵심!)
# ==========================================
print("데이터 샘플러를 초기화합니다...")
# 이 샘플러가 바로 데이터 편식과 중복(Data Leakage)을 막아주는 일등 공신입니다.
train_sampler = WeightedPDBSampler(
    chain_mapping_paths=[
        os.path.join(TRAIN_CLUSTERING_DIR, "ligand_chain_cluster_mapping.csv"),
        os.path.join(TRAIN_CLUSTERING_DIR, "nucleic_acid_chain_cluster_mapping.csv"),
        os.path.join(TRAIN_CLUSTERING_DIR, "peptide_chain_cluster_mapping.csv"),
        os.path.join(TRAIN_CLUSTERING_DIR, "protein_chain_cluster_mapping.csv"),
    ],
    interface_mapping_path=os.path.join(TRAIN_CLUSTERING_DIR, "interface_cluster_mapping.csv"),
    batch_size=BATCH_SIZE
)

# ==========================================
# 3. PDBDataset 로드 (수동 파싱 불필요)
# ==========================================
print("PDB 데이터셋을 불러오는 중입니다...")
train_dataset = PDBDataset(
    folder=TRAIN_DIR,
    sampler=train_sampler,
    sample_type='clustered',
    training=True,
)
valid_dataset = PDBDataset(folder=VAL_DIR, inference=True)
test_dataset = PDBDataset(folder=TEST_DIR, inference=True)


# ==========================================
# 4. AlphaFold 3 모델 초기화
# ==========================================
print("AlphaFold 3 모델을 메모리에 올립니다...")
alphafold3 = Alphafold3(
    dim_atom_inputs = 77,
    dim_atompair_inputs = 5,
    atoms_per_window = 27,
    dim_template_feats = 108,
    num_dist_bins = 64,
    confidence_head_kwargs = dict(pairformer_depth = 1),
    template_embedder_kwargs = dict(pairformer_stack_depth = 1),
    msa_module_kwargs = dict(depth = 1),
    pairformer_stack = dict(
        depth = 1,
        pair_bias_attn_dim_head = 4,
        pair_bias_attn_heads = 2,
    ),
    diffusion_module_kwargs = dict(
        atom_encoder_depth = 1,
        token_transformer_depth = 1,
        atom_decoder_depth = 1,
        atom_decoder_kwargs = dict(attn_pair_bias_kwargs = dict(dim_head = 4)),
        atom_encoder_kwargs = dict(attn_pair_bias_kwargs = dict(dim_head = 4))
    )
)

# ==========================================
# 5. 통합 학습기(Trainer) 세팅 및 실행
# ==========================================
logger = TensorBoardLogger(root_dir="./logs", name="af3_training")

print("🚀 트레이너 엔진 시동 중...")
trainer = Trainer(
    model = alphafold3,
    dataset = train_dataset,
    valid_dataset = valid_dataset,
    test_dataset = test_dataset,
    accelerator = 'gpu',
    num_train_steps = NUM_TRAIN_STEPS,
    batch_size = BATCH_SIZE,
    valid_every = 1000,
    checkpoint_every = 1000,
    checkpoint_folder = './checkpoints', 
    use_ema = True,
    ema_kwargs = dict(use_foreach = True),
    loggers = [logger]  # <--- 이 부분만 추가!
)
# 학습 시작! (loss.backward()와 optimizer.step()을 내부에서 전부 알아서 해줍니다)
trainer()