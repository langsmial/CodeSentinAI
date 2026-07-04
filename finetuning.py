import os
import math
import logging
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def main():
    
    csv_file_path = '/content/final_12000_sbert_dataset.csv'

    if not os.path.exists(csv_file_path):
        print(f"[에러] '{csv_file_path}' 파일이 존재하지 않습니다. 파이썬 코드와 같은 폴더에 있는지 확인해주세요.")
        return

    print("===== 1. 통합 데이터셋 로드 및 전처리 중 =====")
    df = pd.read_csv(csv_file_path)

    # SBERT 훈련 규격에 맞게 InputExample 객체 배열로 변환
    train_examples = []
    for idx, row in df.iterrows():
        # 결측치 예외 처리 및 데이터 형변환
        s1 = str(row['sentence1']).strip()
        s2 = str(row['sentence2']).strip()
        score = float(row['score'])

        train_examples.append(InputExample(texts=[s1, s2], label=score))

    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    print("===== 2. 베이스 모델 로드 중 (jhgan/ko-sroberta-multitask) =====")
    
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    
    train_loss = losses.CosineSimilarityLoss(model=model)

    
    num_epochs = 4 
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    output_path = './fine_tuned_ko_sbert' 

    print(f"===== 3. 파인튜닝 가동 (총 Epoch: {num_epochs} / Warmup Steps: {warmup_steps}) =====")

   
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True 
    )

    print(f"===== 4. 학습 완료! 모델이 [{output_path}] 폴더에 성공적으로 저장되었습니다. =====")

if __name__ == "__main__":
    main()
