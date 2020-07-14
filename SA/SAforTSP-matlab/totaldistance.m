function totaldis = totaldistance(route,dis)%传入距离矩阵和当前路线
%TOTALDISTANCE 此处显示有关此函数的摘要
%   此处显示详细说明

% fprintf("%d\n",route(1));
totaldis=dis(route(end),route(1));
% totaldis=dis(route(end),route(1));
for k=1:length(route)-1
    totaldis=totaldis+dis(route(k),route(k+1));%直接加两个点之间的距离
end

