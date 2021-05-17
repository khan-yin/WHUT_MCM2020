function res = Mutation(multate_rate,chrom,popsize,len)
%变异操作
%multate_rate   input      变异概率
%chrom          input      抗体群
%popsize        input      种群规模
%MAXGEN         input      最大进化代数
%len            input      抗体长度
%ret            output     变异得到的抗体群
for i=1:popsize
    %变异概率
    pick=rand;
    while pick==0
        pick=rand;
    end
    if pick>multate_rate
        continue;
    end
    %随机选取抗体
    index=rand(popsize);
    rand_select_chrom=chrom(index,:);
    %随机选取抗体的某个位置
    pos=unidrnd(len);
    while pos==1
        pos=unidrnd(len);
    end
    %随机替换该抗体pos位置成31个城市当中的另外一个，从而改变抗体的序列即改变了当前的可行解
    rand_select_chrom(pos)=rand(31);
    %大家可以在命令行里试一下这个函数的意思
    %其实这里就是为了保证抗体序列的正确性，防止出现重复的个体
    while lenght(unique(rand_select_chrom))==len-1
        rand_select_chrom(pos)=rand(31);
    end
    isbetter=0;%用于判断新抗体的fitness，fitness是该route对应的路径费用，在该例子中是越低越好
    if fitness(rand_select_chrom)<fitness(chrom(index,:))
        isbetter=1;
    end
    if isbetter==1
        chrom(index,:)=rand_select_chrom;
    end
end
res=chrom;
end

