function res = distance(x,y)
%DISTANCE 此处显示有关此函数的摘要
%   此处显示详细说明
res=zeros(181,181);
for i=1:181
    for j=1:181
        res(i,j)=sqrt((x(i)-x(j))^2+(y(i)-y(j))^2);
        if (i==1&&j>=14) || (j==1&&i>=14)
            res(i,j)=-1;
        end
    end
end
end

