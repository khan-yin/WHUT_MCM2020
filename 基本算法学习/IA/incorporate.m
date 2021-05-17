function newindividuals = incorporate(individuals,popsize,memory_indivisuals,memory_celllist)
%将记忆库中抗体加入，形成新种群
%individuals       input      抗体群
%sizepop           input      抗体数
%bestindividuals   input      记忆库
%overbest          input      记忆库容量
%把记忆细胞和其余细胞整合起来
total=popsize+memory_celllist;
%定义一个新种群的结构体
newindividuals=struct('fitness',zeros(1,total),'concentration',zeros(1,total),'excellence',zeros(1,total),'chrom',[]);
%通过遗传方式得到的抗体
for i=1:popsize
    newindividuals.fitness(i)=individuals.fitness(i);
    newindividuals.concentration(i)=individuals.concentration(i);
    newindividuals.excellence(i)=individuals.excellence(i);
    newindividuals.chrom(i,:)=individuals.chrom(i,:);
end
%记忆库中抗体
for i=popsize+1:total
    newindividuals.fitness(i)=memory_indivisuals.fitness(i-popsize);
    newindividuals.concentration(i)=memory_indivisuals.concentration(i-popsize);
    newindividuals.excellence(i)=memory_indivisuals.excellence(i-popsize);
    newindividuals.chrom(i,:)=memory_indivisuals.chrom(i-popsize,:);
end
end

