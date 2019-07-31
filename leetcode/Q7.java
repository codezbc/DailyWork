public class Q7 {
    public int reverse(int x){
        if(x==0)return 0;
        if(x>-10&&x<10)return x;
        long y=(long)x;
        if(x<0)
            y=Math.abs(x);
        long quotient=y/10;      //比原来数少一位的数
        long remainder=y%10; //个位数
        long result=remainder;
        while (quotient!=0){
            remainder=quotient%10; //倒数位
            quotient=quotient/10;    //不断减少
            result=result*10+remainder; //逆向输出
        }
        if(x<0){
             if((-1)* result<Integer.MIN_VALUE) return 0;
             else return (-1)*(int)result;
        }
        else if(result>Integer.MAX_VALUE)return  0;
        else return (int) result;
    }
}
