function route_2 = Prime3(d)
%PRIME 此处显示有关此函数的摘要
%   此处显示详细说明
len=length(d);
dback=d;
totaldistance=0;
d(d<=0) = Inf;
P = zeros(1, len);
P(1,1) = 1;
V = 1:len;
V_P = V - P;
route = zeros(len,2);
route_2=zeros(len,len);
k=1;
flag=0;
while k<len
    p = P(P~=0);
    v = V_P(V_P~=0);
    pv = min(min(d(p,v)));
    [x, y] = find(d==pv);
    for i=1:length(x)
        if  any(P==x(i)) && any(V_P==y(i)) &&dback(x(i),y(i))~=-1 %集合判断，关键！
            P(1,y(i)) = y(i);
            V_P = V - P;
            route(k, :) = [x(i), y(i)];
            route_2(x(i),y(i))=1;
            if totaldistance+dback(x(i),y(i))>40
                flag=1;
                break;
            end
            totaldistance=totaldistance+dback(x(i),y(i));
            k = k+1;
            break;
        end
    end
    if flag==1
        break;
    end
end
fprintf("totaldistance1:%f\n",totaldistance);

