function newroute = changeroute3(num,count)
    a=[31,52,56,62,63,64,67,72,73,90,181];
    k=nchoosek(a,num);
    
    upone=k(count,:);%max=55
    p=setdiff( a, upone);
    newroute=[1,2,3,4,5,6,7,8,9,10,11,12,13,upone];
end
