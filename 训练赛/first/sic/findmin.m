function t = findmin(temp)
%FINDMIN �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
v=sort(temp);
s=find(temp>0);
y=temp(s);
[t(1),t(2)]=min(y);
end

