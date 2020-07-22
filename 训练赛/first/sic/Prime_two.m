function [route_2,totaldistance] = Prime2(d,route1)
%PRIME 此处显示有关此函数的摘要
%   此处显示详细说明
rlen=length(route1);
len=length(d);
dback=d;
totaldistance=0;
d(d<=0) = Inf;
P = zeros(1, len);
P(1,1) = 1;
for i=1:rlen
    P(1,i)=i;
end
V = 1:len;
V_P = V - P;
% route = zeros(len,2);
route_2=zeros(len,len);
% for i=1:rlen
%     for j=1:rlen
%         route_2(i,j)=route1(i,j);
%     end
% end

k=rlen-1;
while k<len-1
    p = P(P~=0);
    v = V_P(V_P~=0);
    if isempty(v)
        break;
    end
    pv = min(min(d(p,v)));
    [x, y] = find(d==pv);
    for i=1:length(x)
%         if(x(i)<=12&&y(i)<=12&&route1(x(i),y(i))==1)
%             P(1,y(i)) = y(i);
%             V_P = V - P;
%             route_2(x(i),y(i))=1;
%             continue;
%         end
        if  any(P==x(i)) && any(V_P==y(i))   %集合判断，关键！
            P(1,y(i)) = y(i);
            V_P = V - P;
%             route(k, :) = [x(i), y(i)];
            route_2(x(i),y(i))=1;
            totaldistance=totaldistance+dback(x(i),y(i));
            k = k+1;
            break;
        end
    end
end
% fprintf("totaldistance2:%f\n",totaldistance);

