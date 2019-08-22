def  maxArea(height):
    left=maxvVal=0
    dist=right=len(height)-1
    while left<right:
        curVal=dist*min(height[left],height[right])
        if curVal<maxvVal:
            maxvVal=curVal
        if height[left]<height[right]:
            left+=1
        else:
            right-=1
        dist-=1
    return maxvVal
if __name__ == '__main__':
    maxArea([1,8,6,2,5,4,8,3,7])