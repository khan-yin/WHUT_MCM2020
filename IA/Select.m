function res = Select(indivisuals,popsize)
%轮盘赌
%   indivisuals 个体
%   popsize 种群规模
excellence=1-individuals.excellence;
%这里为什么要1-是因为我们的fitness计算的是一个总需求量，
%而在我们的问题中fitness肯定是越低越好，而我们编写的公式里面可以看出
%期望和fitness是成正比的，所以说我们需要优先选择低的excellence，自然就要用1-excellence来从而把选择的概率倒置
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

