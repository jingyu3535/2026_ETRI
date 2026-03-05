import sys
import os
import time

# 라이브러리 경로 설정
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from lerobot.motors.feetech.feetech import FeetechMotorsBus
    print("✅ 라이브러리 로드 성공")
except Exception as e:
    print(f"❌ 라이브러리 로드 실패: {e}")
    sys.exit()

def scan_port(port_name):
    print(f"\n🔍 {port_name} 스캔 중...")
    # 임시 모터 객체 생성 (ID 1번부터 10번까지 확인)
    for i in range(1, 11):
        try:
            # 0번 인덱스에 ID i번 모터를 할당하여 테스트
            bus = FeetechMotorsBus(port_name, motors={0: type('M', (object,), {'id': i, 'model': 'sts3215'})})
            bus.connect()
            
            # 주소 56번(Present Position)을 직접 읽어봅니다.
            pos = bus._read(56, 2, i)
            if pos is not None:
                print(f"  ✨ [발견] ID: {i} | 현재 위치: {pos}")
            bus.disconnect()
        except:
            continue

if __name__ == "__main__":
    # 리더와 팔로워 포트 모두 스캔
    scan_port("/dev/ttyACM1") # Leader 후보
    scan_port("/dev/ttyACM0") # Follower 후보