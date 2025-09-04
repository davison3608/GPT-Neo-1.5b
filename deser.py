import numpy as np
from transformers import AutoTokenizer
"解码返回"
def string_deindex(arr: list):
    # 加载分词器
    token = AutoTokenizer.from_pretrained("vocab")
    # 设置填充
    if token.pad_token_id is None:
        token.pad_token_id = token.eos_token_id
    arr=np.array(arr)
    # 反索引到词原
    str=token.decode(
        arr
    )
    # 返回
    return str

def main():
    test=[4654, 1231, 123, 56, 8989, 3334, 1045]
    test=list(test)
    str=string_deindex(test)
    print(str)
if __name__ == "__main__":
    main()