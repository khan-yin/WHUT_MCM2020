function newindividuals = incorporate(individuals,popsize,memory_indivisuals,memory_celllist)
%��������п�����룬�γ�����Ⱥ
%individuals       input      ����Ⱥ
%sizepop           input      ������
%bestindividuals   input      �����
%overbest          input      ���������
%�Ѽ���ϸ��������ϸ����������
total=popsize+memory_celllist;
%����һ������Ⱥ�Ľṹ��
newindividuals=struct('fitness',zeros(1,total),'concentration',zeros(1,total),'excellence',zeros(1,total),'chrom',[]);
%ͨ���Ŵ���ʽ�õ��Ŀ���
for i=1:popsize
    newindividuals.fitness(i)=individuals.fitness(i);
    newindividuals.concentration(i)=individuals.concentration(i);
    newindividuals.excellence(i)=individuals.excellence(i);
    newindividuals.chrom(i,:)=individuals.chrom(i,:);
end
%������п���
for i=popsize+1:total
    newindividuals.fitness(i)=memory_indivisuals.fitness(i-popsize);
    newindividuals.concentration(i)=memory_indivisuals.concentration(i-popsize);
    newindividuals.excellence(i)=memory_indivisuals.excellence(i-popsize);
    newindividuals.chrom(i,:)=memory_indivisuals.chrom(i-popsize,:);
end
end

