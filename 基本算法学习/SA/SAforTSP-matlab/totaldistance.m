function totaldis = totaldistance(route,dis)%����������͵�ǰ·��
%TOTALDISTANCE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

% fprintf("%d\n",route(1));
totaldis=dis(route(end),route(1));
% totaldis=dis(route(end),route(1));
for k=1:length(route)-1
    totaldis=totaldis+dis(route(k),route(k+1));%ֱ�Ӽ�������֮��ľ���
end

