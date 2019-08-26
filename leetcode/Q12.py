def intToRoman(num):
    roman=''
    symb = ['I','V','X', 'L', 'C', 'D', 'M'] #I:1 下标0,V:5,X:10 下标2,L:50 下标3 ,c:100 下标4,D:500,M:1000
    digit=[]
    while num>0:
        digit.append(num%10)
        num = int((num - num%10)/10)
    for i in range(len(digit)):
        if digit[i] == 0:
            roman = ''+ roman
        if digit[i] <4:
            roman = symb[2*i]*(digit[i]) + roman #digit[i]表示的是个数的多少
        if digit[i] == 4:
            roman = symb[2*i]+symb[2*i+1] + roman
        if digit[i] == 5:
            roman = symb[2*i+1] + roman
        if 9 > digit[i] >5:
            roman = symb[2*i+1]+symb[2*i]*(digit[i]-5)+ roman
        if digit[i] == 9 :
            roman = symb[2*i]+symb[2*i+2] + roman
    return roman
if __name__ == '__main__':
    result=intToRoman(12)
    print(result)