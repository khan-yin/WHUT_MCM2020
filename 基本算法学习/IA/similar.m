function similarity = similar(individual1,individual2)
%�������individual1,individual2֮������ƶ�
% individual1,individual2 input ��������
% similarity output ���ƶȣ���ͬ������ռ�ı���
len=length(individual1);
k=zeros(1,len);
for i =1:len
    if find(individual1(i)==individual2)%�������һ��Ҫ��������
        k(i)=1;
    end
end
similarity=sum(k)/len;
end

