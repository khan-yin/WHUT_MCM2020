function route = change(pre_route,method)
%CHANGE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

route=pre_route;
routelen=length(route);
%���ȡ�����൱�ڰ�0-167������ӳ�䵽0-1�ĵȸ��ʳ�ȡ�ϣ���ȡ��
city1=ceil((routelen-14)*rand+14); % [1, 2, ..., n-1, n]
city2=ceil((routelen-14)*rand+14); % 1<=city1, city2<=n
switch method
    case 'reverse' %[1 2 3 4 5 6] -> [1 5 4 3 2 6]
        cmin = min(city1,city2);
        cmax = max(city1,city2);
        route(cmin:cmax) = route(cmax:-1:cmin);%��תĳһ��
    case 'swap' %[1 2 3 4 5 6] -> [1 5 3 4 2 6]
        route([city1, city2]) = route([city2, city1]);
end

