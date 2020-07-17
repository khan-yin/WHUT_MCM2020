function res = Cross(corss_rate,chrom,popsize,length)
%交叉操作
%corss_rate  input     交叉概率
%chrom       input     抗体群
%popsize     input     种群规模
%length      input     抗体长度
%ret         output    交叉得到的抗体群
%每一轮for循环中，可能会进行一次交叉操作，是否进行交叉操作则由交叉概率pcorss控制
for i=1:popsoze
    pick=rand;
    while prod(pick)==0
        pick=rand(1);
    end
    if pick>corss_rate
        continue;
    end
    %随机选取出需要交叉的个体下标
    index(1)=unidrnd(popsize);
    index(2)=unidrnd(popsize);
    while index(2)==index(1)
          index(2)=unidrnd(popsize);
    end
     %选择交叉位置
    pos=ceil(length*rand);%rand在0-1之间,ceil向上取整后可以选取1-6号对应的个体
    while pos==1
        pos=ceil(length*rand);
    end
    %个体交叉
    chrom1=chrom(index(1),:);
    chrom2=chrom(index(2),:);
    %抗体有效片段交换，序列重组
    k=chrom1(pos:length);
    chrom1(pos:length)=chrom2(pos:length);
    chrom2(pos:length)=k;
    isbetter=0;%用于判断新重组的抗体的fitness，fitness是该route对应的路径费用，在该例子中是越低越好
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

