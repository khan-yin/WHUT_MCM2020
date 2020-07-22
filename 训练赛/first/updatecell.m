function cell = updatecell(x,y)
%UPDATECELL 此处显示有关此函数的摘要
%   此处显示详细说明
for i=1:181
    for j=1:181
        cell(i,:)=[x(i),y(i)];
    end
end

end

