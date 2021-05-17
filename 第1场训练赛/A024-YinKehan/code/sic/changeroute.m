function newroute = changeroute(pre_route,count)
    a=14:181;
    k=nchoosek(a,2);
    upone=k(count,:);
    p=setdiff( a, upone);
    newroute=[1,2,3,4,5,6,7,8,9,10,11,12,13,upone,p];
end

