function res = Cross(corss_rate,chrom,popsize,length)
%�������
%corss_rate  input     �������
%chrom       input     ����Ⱥ
%popsize     input     ��Ⱥ��ģ
%length      input     ���峤��
%ret         output    ����õ��Ŀ���Ⱥ
%ÿһ��forѭ���У����ܻ����һ�ν���������Ƿ���н���������ɽ������pcorss����
for i=1:popsoze
    pick=rand;
    while prod(pick)==0
        pick=rand(1);
    end
    if pick>corss_rate
        continue;
    end
    %���ѡȡ����Ҫ����ĸ����±�
    index(1)=unidrnd(popsize);
    index(2)=unidrnd(popsize);
    while index(2)==index(1)
          index(2)=unidrnd(popsize);
    end
     %ѡ�񽻲�λ��
    pos=ceil(length*rand);%rand��0-1֮��,ceil����ȡ�������ѡȡ1-6�Ŷ�Ӧ�ĸ���
    while pos==1
        pos=ceil(length*rand);
    end
    %���彻��
    chrom1=chrom(index(1),:);
    chrom2=chrom(index(2),:);
    %������ЧƬ�ν�������������
    k=chrom1(pos:length);
    chrom1(pos:length)=chrom2(pos:length);
    chrom2(pos:length)=k;
    isbetter=0;%�����ж�������Ŀ����fitness��fitness�Ǹ�route��Ӧ��·�����ã��ڸ���������Խ��Խ��
    if fitness(chrom1)<fitness(chrom(index(1),:))||fitness(chrom2)<fitness(chrom(index(2),:))
        isbetter=1;
    end
    if isbetter==1
        chrom(index(1),:)=chrom1;
        chrom(index(2),:)=chrom2;
    end
    
end
res=chrom;
end

