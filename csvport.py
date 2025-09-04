import pandas as pd
import os

def single_talk(days: list, question: str, answer: str):
    year=days[0]
    month=days[1]
    day=days[2]

    new_row={
        "year": year,
        "month": month,
        "day": day,
        "question": question,
        "answer": answer
    }

    df=pd.DataFrame([new_row])
    # 是否存在 无则创建
    file_exists=os.path.exists('talks_csv.csv')

    df.to_csv(
        'talks.csv',
        mode='a', # 追加模式
        header=not file_exists,
        index=False
    )

def main():
    for _ in range(5):
        single_talk(
            [2025, 8, 15],
            "test_quest",
            "test_answer"
        )

if __name__ == "__main__":
    main()
