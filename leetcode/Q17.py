def letterCombinations(digits):
    result=['']
    if len(digits)==0:
        return []
    dict={'0':'','1':'','2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
    for digit in digits:
        ln=len(result)
        temp_result=[]
        for i in range(0,ln):
            t=result[i]
            for ch in dict[digit]:
                u=t+ch
                temp_result.append(u)
        result=temp_result
    return result

if __name__ == '__main__':
   res=letterCombinations('234')
   print(res)