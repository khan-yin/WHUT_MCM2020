function res = Mutation(multate_rate,chrom,popsize,len)
%�������
%multate_rate   input      �������
%chrom          input      ����Ⱥ
%popsize        input      ��Ⱥ��ģ
%MAXGEN         input      ����������
%len            input      ���峤��
%ret            output     ����õ��Ŀ���Ⱥ
for i=1:popsize
    %�������
    pick=rand;
    while pick==0
        pick=rand;
    end
    if pick>multate_rate
        continue;
    end
    %���ѡȡ����
    index=rand(popsize);
    rand_select_chrom=chrom(index,:);
    %���ѡȡ�����ĳ��λ��
    pos=unidrnd(len);
    while pos==1
        pos=unidrnd(len);
    end
    %����滻�ÿ���posλ�ó�31�����е��е�����һ�����Ӷ��ı俹������м��ı��˵�ǰ�Ŀ��н�
    rand_select_chrom(pos)=rand(31);
    %��ҿ���������������һ�������������˼
    %��ʵ�������Ϊ�˱�֤�������е���ȷ�ԣ���ֹ�����ظ��ĸ���
    while lenght(unique(rand_select_chrom))==len-1
        rand_select_chrom(pos)=rand(31);
    end
    isbetter=0;%�����ж��¿����fitness��fitness�Ǹ�route��Ӧ��·�����ã��ڸ���������Խ��Խ��
    if fitness(rand_select_chrom)<fitness(chrom(index,:))
        isbetter=1;
    end
    if isbetter==1
        chrom(index,:)=rand_select_chrom;
    end
end
res=chrom;
end

