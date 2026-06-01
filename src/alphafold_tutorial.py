import torch
from alphafold3_pytorch import Alphafold3
from alphafold3_pytorch.utils.model_utils import exclusive_cumsum

alphafold3 = Alphafold3(
    dim_atom_inputs = 77,
    dim_template_feats = 108
)

# mock inputs

seq_len = 16

molecule_atom_indices = torch.randint(0, 2, (2, seq_len)).long()
molecule_atom_lens = torch.full((2, seq_len), 2).long()

atom_seq_len = molecule_atom_lens.sum(dim=-1).amax()
atom_offsets = exclusive_cumsum(molecule_atom_lens)

atom_inputs = torch.randn(2, atom_seq_len, 77)
atompair_inputs = torch.randn(2, atom_seq_len, atom_seq_len, 5)

additional_molecule_feats = torch.randint(0, 2, (2, seq_len, 5))
additional_token_feats = torch.randn(2, seq_len, 33)
is_molecule_types = torch.randint(0, 2, (2, seq_len, 5)).bool()
is_molecule_mod = torch.randint(0, 2, (2, seq_len, 4)).bool()
molecule_ids = torch.randint(0, 32, (2, seq_len))

template_feats = torch.randn(2, 2, seq_len, seq_len, 108)
template_mask = torch.ones((2, 2)).bool()

msa = torch.randn(2, 7, seq_len, 32)
msa_mask = torch.ones((2, 7)).bool()

additional_msa_feats = torch.randn(2, 7, seq_len, 2)

# required for training, but omitted on inference

atom_pos = torch.randn(2, atom_seq_len, 3)

distogram_atom_indices = molecule_atom_lens - 1

distance_labels = torch.randint(0, 37, (2, seq_len, seq_len))
resolved_labels = torch.randint(0, 2, (2, atom_seq_len))

# offset indices correctly

distogram_atom_indices += atom_offsets
molecule_atom_indices += atom_offsets

# train

loss = alphafold3(
    num_recycling_steps = 2,
    atom_inputs = atom_inputs,
    atompair_inputs = atompair_inputs,
    molecule_ids = molecule_ids,
    molecule_atom_lens = molecule_atom_lens,
    additional_molecule_feats = additional_molecule_feats,
    additional_msa_feats = additional_msa_feats,
    additional_token_feats = additional_token_feats,
    is_molecule_types = is_molecule_types,
    is_molecule_mod = is_molecule_mod,
    msa = msa,
    msa_mask = msa_mask,
    templates = template_feats,
    template_mask = template_mask,
    atom_pos = atom_pos,
    distogram_atom_indices = distogram_atom_indices,
    molecule_atom_indices = molecule_atom_indices,
    distance_labels = distance_labels,
    resolved_labels = resolved_labels
)

loss.backward()

# after much training ...

sampled_atom_pos = alphafold3(
    num_recycling_steps = 4,
    num_sample_steps = 16,
    atom_inputs = atom_inputs,
    atompair_inputs = atompair_inputs,
    molecule_ids = molecule_ids,
    molecule_atom_lens = molecule_atom_lens,
    additional_molecule_feats = additional_molecule_feats,
    additional_msa_feats = additional_msa_feats,
    additional_token_feats = additional_token_feats,
    is_molecule_types = is_molecule_types,
    is_molecule_mod = is_molecule_mod,
    msa = msa,
    msa_mask = msa_mask,
    templates = template_feats,
    template_mask = template_mask
)

print(f"최종 출력 텐서 형태: {sampled_atom_pos.shape}") # (2, <atom_seqlen>, 3)

# ==========================================
# 여기서부터 추가된 결과물 저장 코드입니다.
# ==========================================

# 1. 첫 번째 배치(인덱스 0)의 3D 좌표만 추출
# 기울기(gradient) 계산용 연결 고리를 끊고(detach), CPU로 옮긴 뒤(cpu), 넘파이 배열(numpy)로 변환합니다.
coords = sampled_atom_pos[0].detach().cpu().numpy()

# 2. 3D 좌표를 PDB 파일로 기록하는 함수 정의
def save_tensor_to_pdb(coordinates, filename="alphafold_dummy_output.pdb"):
    with open(filename, "w") as f:
        for i, (x, y, z) in enumerate(coordinates):
            # 현재 테스트 코드는 실제 아미노산 서열(C, N, O 등) 정보가 없는 가짜 데이터이므로,
            # 모든 점을 미상의 아미노산(UNK)에 속한 '탄소(C)' 원자라고 가정하고 PDB 포맷에 맞춰 씁니다.
            f.write(f"ATOM  {i+1:5d}  C   UNK A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n")
        f.write("END\n")

# 3. 파일 저장 실행
save_tensor_to_pdb(coords)
print("성공적으로 alphafold_dummy_output.pdb 파일이 저장되었습니다!")