function res = Select(indivisuals,popsize)
%���̶�
%   indivisuals ����
%   popsize ��Ⱥ��ģ
excellence=1-individuals.excellence;
%����ΪʲôҪ1-����Ϊ���ǵ�fitness�������һ������������
%�������ǵ�������fitness�϶���Խ��Խ�ã������Ǳ�д�Ĺ�ʽ������Կ���
%������fitness�ǳ����ȵģ�����˵������Ҫ����ѡ��͵�excellence����Ȼ��Ҫ��1-excellence���Ӷ���ѡ��ĸ��ʵ���
pselect=excellence./sum(excellence);  
index=[];
for i=1:sizepop
    pick=rand;
    while pick==0
        pick=rand;
    end
    for j=1:sizepop
        pick=pick-pselect(j);
        if pick<0
            index=[index j];
            break;
        end
    end
end
individuals.chrom=individuals.chrom(index,:);
individuals.fitness=individuals.fitness(index);
individuals.concentration=individuals.concentration(index);
individuals.excellence=individuals.excellence(index);
ret=individuals;
end

