import numpy as np
from transformers import AutoTokenizer
"索引化返回"
def string_index(text: str):
    # 加载分词器
    token=AutoTokenizer.from_pretrained("vocab")
    # 设置填充
    if token.pad_token_id is None:
        token.pad_token_id=token.eos_token_id
    # 获取索引 掩码
    results=token(
        text,
        return_tensors='np'
    )
    np.array(results)
    test_ids = results['input_ids']
    test_mask = results['attention_mask']
    return test_ids, test_mask

def main():
    i, m=string_index("this is a test, just for time")
    print(i)
    print(m)
if __name__ == "__main__":
    main()
