function cell = updatecell(x,y)
%UPDATECELL �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
for i=1:181
    for j=1:181
        cell(i,:)=[x(i),y(i)];
    end
end

end

