function route = change(pre_route,method)
%CHANGE 此处显示有关此函数的摘要
%   此处显示详细说明
a=14:181;
% k=nchoosek(a,2);
% upone=k(count,:);

route=pre_route;
routelen=length(route);
%随机取数，相当于把0-167个城市映射到0-1的等概率抽取上，再取整
city1=ceil((routelen-14)*rand+14); % [1, 2, ..., n-1, n]

city2=ceil((routelen-14)*rand+14); % 1<=city1, city2<=n
while city1==city2
    city2=ceil((routelen-14)*rand+14); 
end
upone=[city1,city2];
p=setdiff( a, upone);
newroute=[1,2,3,4,5,6,7,8,9,10,11,12,13,upone,p];
% switch method
%     case 'reverse' %[1 2 3 4 5 6] -> [1 5 4 3 2 6]
%         cmin = min(city1,city2);
%         cmax = max(city1,city2);
%         route(cmin:cmax) = route(cmax:-1:cmin);%反转某一段
%     case 'swap' %[1 2 3 4 5 6] -> [1 5 3 4 2 6]
%         route([city1, city2]) = route([city2, city1]);
end

