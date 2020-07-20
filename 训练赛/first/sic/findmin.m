function t = findmin(temp)
%FINDMIN 此处显示有关此函数的摘要
%   此处显示详细说明
v=sort(temp);
s=find(temp>0);
y=temp(s);
[t(1),t(2)]=min(y);
end

