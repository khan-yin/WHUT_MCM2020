function e = excellence(individuals,Initialpopsize,alpha)
%用于计算个体繁殖的概率
%individuals 种群
%Initialpopsize 种群规模
%alpha 多样性评价参数，用于平衡适应度和浓度之间的平衡
%e 个体繁殖概率
fit=individuals.fitness;
sumfit=sum(fit);
con=individuals.concentration;
sumcon=sum(con);
for i=1:Initialpopsize
    e(i)=fit(i)/sumfit*alpha+con(i)/sumcon*(1-alpha);
end
end

