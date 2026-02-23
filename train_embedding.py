import pandas as pd
import os
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def train_custom_embedding():
    print("[Step 1] 파인튜닝 프로세스를 시작합니다...")
    
    # 1. 경로 설정 (현재 작업 폴더 기준)
    # 실제 Embedding Matrix_VD.csv 위치에 맞게 수정 필요 시 변경
    base_dir = "." 
    embedding_path = os.path.join(base_dir, "Embedding Matrix_VD.csv")
    output_model_path = os.path.join(base_dir, "my-special-roberta")

    # 2. 데이터 파일 확인
    if not os.path.exists(embedding_path):
        print(f"오류: '{embedding_path}' 파일을 찾을 수 없습니다.")
        return

    # 3. 데이터 로드 및 전처리
    print("[Step 2] 데이터 로드 및 전처리 중...")
    try:
        df = pd.read_csv(embedding_path)
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return

    # 필요한 컬럼이 있는지 확인 (사용자가 알려준 컬럼명)
    required_columns = ['Aspect_Lv2', 'Aspect.수렴', 'Opinion', 'Opinion.수렴']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        print(f"오류: 파일에 다음 필수 컬럼이 없습니다: {missing_cols}")
        print(f"현재 있는 컬럼: {list(df.columns)}")
        return

    # 4. InputExample(학습 데이터 쌍) 생성 (통합 학습 방식)
    train_examples = []
    
    # Aspect 데이터 추출 (결측치 제외)
    aspect_pairs = df[['Aspect_Lv2', 'Aspect.수렴']].dropna()
    for _, row in aspect_pairs.iterrows():
        # 빈 문자열 처리
        if str(row['Aspect_Lv2']).strip() and str(row['Aspect.수렴']).strip():
            # (Anchor, Positive) 형태
            train_examples.append(InputExample(texts=[str(row['Aspect_Lv2']), str(row['Aspect.수렴'])]))

    # Opinion 데이터 추출 (결측치 제외)
    opinion_pairs = df[['Opinion', 'Opinion.수렴']].dropna()
    for _, row in opinion_pairs.iterrows():
        # 빈 문자열 처리
        if str(row['Opinion']).strip() and str(row['Opinion.수렴']).strip():
            train_examples.append(InputExample(texts=[str(row['Opinion']), str(row['Opinion.수렴'])]))

    total_pairs = len(train_examples)
    print(f"총 {total_pairs}개의 학습 데이터 쌍(Aspect+Opinion)이 생성되었습니다.")
    
    if total_pairs == 0:
        print("학습할 유효한 데이터가 없습니다. CSV 파일을 확인해주세요.")
        return

    # 5. 모델 로드 (Base Model)
    print("[Step 3] 기본 모델(jhgan/ko-sroberta-multitask) 로드 중...")
    # Device 최적화: GPU 가용 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 디바이스: {device}")
    
    try:
        model = SentenceTransformer('jhgan/ko-sroberta-multitask', device=device)
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        return

    # 6. DataLoader 및 Loss Function 설정
    # batch_size는 메모리에 맞게 조절 (기본 16)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # MNRL (MultipleNegativesRankingLoss): 정답 쌍끼리는 끌어당기고, 임의의 다른 쌍들과는 밀어내는 Loss 함수
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 7. 파인튜닝(학습) 수행
    print("[Step 4] 파인튜닝 학습을 시작합니다. 데이터 양에 따라 시간이 걸릴 수 있습니다...")
    # 에포크(가르칠 횟수)는 보통 1~4회 정도가 적당 (과적합 방지)
    epochs = 4
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100, # 학습 초반 안정화를 위한 단계 수(총 데이터 쌍 수의 10% 정도가 적당)
        show_progress_bar=True
    )

    # 8. 모델 저장
    print(f"[Step 5] 학습 완료! 파인튜닝된 모델을 '{output_model_path}' 에 저장합니다.")
    model.save(output_model_path)
    print("모든 과정이 성공적으로 완료되었습니다!")
    print(f"이제 메인 어플리케이션에서 모델 경로를 '{output_model_path}' 로 변경하여 사용하세요.")

if __name__ == '__main__':
    train_custom_embedding()
