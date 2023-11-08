import json

import accelerate
import torch
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

solutions_01 = """
q=int(input())
for e in range(q):
    x,y,k=list(map(int,input().split()))
    x,y=abs(x),abs(y)
    x,y=max(x,y),min(x,y)

    if(x%2!=k%2):
        k-=1
        y-=1

    if(x>k):
        print(-1)
        continue
    if((x-y)%2):
        k-=1
        x-=1
    print(k)
"""
plans_01 = """
Let's think step by step
1. First need to get the input correctly according to the prompt in the -----Input-----.
2. Next loop the input list
3. Get the Maximum and minimum absolute value.
4. If both x and k have the same parity (odd or even), k and y each decrease by 1.
5. If x is greater than k, it means that cannot reach the point, then continue.
6. If the difference between x and y is odd, k and x each decrease by 1, then k represents the number of moves needed to reach the destination
7. Print the result.
"""

solutions_03 = """
def solve():
    n, k = map(int,input().split())
    lst = list(map(int,input().split()))
    lst.sort()
    ans = 0
    for i in range(n - k - 1, n):
        ans += lst[i]
    print(ans)
for i in range(int(input())):
    solve()
"""
plans_03 = """
Let's think step by step
1. First defines a function can be called later.
2. In the function, reads the input according to the prompt in the -----Input-----.
3. Sort the list just read.
4. Traverse the last k elements and sum them in a for loop.
5. Print the result.
"""

if __name__ == '__main__':
    import os

    os.environ['CURL_CA_BUNDLE'] = ''
    path = "tora-code-13b/"
    model_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(path, device="cuda", use_fast=False, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    model = accelerate.load_checkpoint_and_dispatch(model, path, device_map="auto", dtype=torch.bfloat16)
    input_ids = tokenizer("I want to lose weight. Please give me a plan")['input_ids']
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to('cuda')
    outputs = model.generate(inputs=input_ids,
                             return_dict_in_generate=True,
                             output_hidden_states=True,
                             max_new_tokens=256)
    print(tokenizer.decode(outputs[0][0]))
    print("end")