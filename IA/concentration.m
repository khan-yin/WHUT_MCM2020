function concentration = concentration(k,Initalpopsize,individuals)
%���㵱ǰ��ȺŨ��
%
%k ��k������
%Initalpopsize ��Ⱥ��ģ
%individuals ����
%concentration Ũ�ȼ���
concentrationCount = 0;
for i=1:Initalpopsize
    similarty=similar(individuals.chrom(i,:),individuals.chrom(k,:));
    if similarty>0.7:
        concentrationCount=concentrationCount+1;
    end
end
concentration=concentrationCount/Initalpopsize;
end

