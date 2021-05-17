function route = change(pre_route,method)
%CHANGE 此处显示有关此函数的摘要
%   此处显示详细说明
route=pre_route;
cityaccount=length(route);
%随机取数，相当于把0-34个城市映射到0-1的等概率抽取上，再取整
city1=ceil(cityaccount*rand); % [1, 2, ..., n-1, n]
city2=ceil(cityaccount*rand); % 1<=city1, city2<=n
switch method
    case 'reverse' %[1 2 3 4 5 6] -> [1 5 4 3 2 6]
        cmin = min(city1,city2);
        cmax = max(city1,city2);
        route(cmin:cmax) = route(cmax:-1:cmin);%反转某一段
    case 'swap' %[1 2 3 4 5 6] -> [1 5 3 4 2 6]
        route([city1, city2]) = route([city2, city1]);
end

