class Solution():
    def  intToRoman(self,num):
        string=''
        intToRoman_dict={1:'I',5:'V',10:'X',50:'L',100:'C',500:'D',1000:'M'}
        divider=1
        while True:
            if num//10!=0:
                curr_num=num%10
                string=self.add_string(intToRoman_dict,string,divider,curr_num)
                divider=divider*10
                num=num//10
            else:
                curr_num=num
                string = self.add_string(intToRoman_dict, string, divider, curr_num)
                break
        return string

    def add_string(self,intToRoman_dict,string,divider,curr_num):
        if curr_num<4:  #10的最高位是1
            string=intToRoman_dict[divider]*curr_num+string
        elif curr_num==4:
            string=intToRoman_dict[divider]+intToRoman_dict[divider*5]+string
        elif curr_num==5:
            string=intToRoman_dict[divider*5]+string
        elif 9>curr_num>5:
            string=intToRoman_dict[divider*5]+intToRoman_dict[divider]*(curr_num-5)+string
        else: #9的情况 IV
            string=intToRoman_dict[divider]+intToRoman_dict[divider*10]+string
        return string

if __name__ == '__main__':
    result=Solution().intToRoman(1000)
    print(result)