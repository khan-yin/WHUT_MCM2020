function e = excellence(individuals,Initialpopsize,alpha)
%���ڼ�����己ֳ�ĸ���
%individuals ��Ⱥ
%Initialpopsize ��Ⱥ��ģ
%alpha ���������۲���������ƽ����Ӧ�Ⱥ�Ũ��֮���ƽ��
%e ���己ֳ����
fit=individuals.fitness;
sumfit=sum(fit);
con=individuals.concentration;
sumcon=sum(con);
for i=1:Initialpopsize
    e(i)=fit(i)/sumfit*alpha+con(i)/sumcon*(1-alpha);
end
end

