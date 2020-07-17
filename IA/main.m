%人工免疫算法
clc
clear;
%%
popsize=30;%种群规模
memory_celllist=10;%记忆细胞容量
MAXGEN=100;%迭代次数
corss_rate=0.5;%交叉概率
multate_rate=0.4;%变异概率
alpha=0.4;%多样性参数
distribution_count=6;%配送中心数
Initialpopsize=popsize+memory_celllist;%初始化种群数目
%% 
%将种群信息定义为一个结构体,
%包含每个个体fitness构成的序列，
%concentration每个个体对应抗体类型的浓度，
%excelence每个抗体存活期望，
%chrom[]每个个体对应的抗体序列
individuals=struct('fitness',zeros(1,Initialpopsize),'concentration',zeros(1,Initialpopsize),'excellence',zeros(1,Initialpopsize),'chrom',[]);
%产生初始抗体群
for i=1:Initialpopsize
    individuals.chrom(i,:)=randperm(31,distribution_count);%随机生成抗体
end

trace=[];   %记录每代最优个体适应度

for k=1:MXGEN
    %步骤三抗体群多样性评价
    for i=1:M
        individuals.fitness(i)=fitness(individuals.chrom(i,:)); %抗体序列的总费用计算
        individuals.concentration(i)=concentration(i,Initialpopsize,individuals); %抗体浓度计算
    end
    %综合亲和度和浓度评价抗体优秀程度，得出期望繁殖概率
    individuals.excellence=excellence(individuals,Initialpopsize,alpha);
    %记录当代最佳个体和种群平均适应度
    [best,index1]=min(individuals.fitness);        %找出最优适应度
    bestchrom=individuals.chrom(index1,:);         %找出最优个体
    trace=[trace;best];                   %记录
    %excellence越低越好
    %步骤4：根据excellence，形成父代群，更新记忆库（加入精英保留策略）
    bestindividuals=bestselect1(individuals,memory_celllist);      %更新记忆库
    fa_individuals=bestselect2(individuals,popsize);         %形成父代群 
    %步骤6：选择、交叉、变异操作，再加入记忆库中抗体，产生新种群
    individuals=Select(fa_individuals,popsize);       %选择
    individuals.chrom=Cross(corss_rate,fa_individuals.chrom,popsize,length);  %交叉
    individuals.chrom=Mutation(multate_rate,fa_individuals.chrom,popsize,length);  %变异
    %加入记忆库中抗体
    individuals=incorporate(individuals,popsize,bestindividuals,memory_celllist);
end

%找出最优解
for i=1:M
    results(i)=fitness(individuals.chrom(i,:));
end
[D,index3]=min(results);
bestchrom=individuals.chrom(index3,:)
expense1=fitness(individuals.chrom(index3,:))
subplot(1,2,1)
plot(trace)
subplot(1,2,2)
draw_figure(bestchrom)

