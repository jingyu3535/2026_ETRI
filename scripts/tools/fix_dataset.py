import os
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 설정
repo_id = "etri01/banana_box_100"
task_label = "pick the banana and put it in the box"
new_repo_id = "etri01/banana_box_100_task_fixed"

print(f"--- 데이터셋 로드 시작: {repo_id} ---")
# 로컬 캐시에서 데이터셋 로드
dataset = LeRobotDataset(repo_id)

print(f"--- Task 라벨 주입 중: '{task_label}' ---")
# 모든 프레임에 task 텍스트 추가
def add_task_name(example):
    example["task"] = task_label
    return example

dataset.hf_dataset = dataset.hf_dataset.map(add_task_name)

# meta/info.json에 task 피처 정보 등록 (SmolVLA 인식용)
dataset.meta.features["task"] = {
    "dtype": "string",
    "shape": [1],
    "names": None
}

print(f"--- 수정된 데이터셋 저장 중: {new_repo_id} ---")
# 로컬에 새로운 이름으로 저장
dataset.save_to_disk(os.path.expanduser(f"~/.cache/huggingface/lerobot/{new_repo_id}"))

print("\n✅ 수정 완료!")
print(f"새로운 데이터셋 경로: ~/.cache/huggingface/lerobot/{new_repo_id}")
print(f"이제 학습 시 --dataset.repo_id={new_repo_id} 옵션을 사용하세요.")
