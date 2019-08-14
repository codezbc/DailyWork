INT_MAX = 2**31 - 1
INT_MIN = -(2**31)
INT_MAX_DIV_10 = INT_MAX // 10 #整数运算除以10
LMAX = INT_MAX - 10 * INT_MAX_DIV_10

class Solution:
    def myAtoi(self, a: str) -> int:
        n, s = len(a), 1  #s代表符号
        i = j = 0 
        while j < n and a[j] == ' ': 
            j += 1
        if j == n: 
            return 0
        if a[j] == '+':    #要么是加法
            j += 1
        elif a[j] == '-': #要么是减法
            s = -1
            j += 1
        while j < n and a[j].isnumeric():
            if i > INT_MAX_DIV_10 or (i == INT_MAX_DIV_10 and int(a[j]) > LMAX):  #还有一次才退出循环
                if s == 1:
                    return INT_MAX
                else:
                    return INT_MIN
            i = 10 * i + int(a[j])
            j += 1  #控制循环变量
        return s * i

if __name__ == '__main__':
    s=Solution()
    a=s.myAtoi('-100')
    print(a)