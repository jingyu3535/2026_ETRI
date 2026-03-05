import os
import wandb

# WANDB_API_KEY는 코드에 하드코딩하지 않고 환경변수에서 읽습니다.
if not os.getenv("WANDB_API_KEY"):
    raise SystemExit("WANDB_API_KEY is not set. Export it before running this script.")

os.environ.setdefault("WANDB_MODE", "online")

try:
    run = wandb.init(project="connection-test")
    print("\n✅ W&B 연결 성공!")
    print(f"🔗 접속 링크: {run.get_url()}")

    wandb.log({"check_val": 1.0})
    print("✅ 데이터 전송 완료!")

    run.finish()
except Exception as e:
    print("\n❌ 연결 실패!")
    print(f"에러 내용: {e}")
