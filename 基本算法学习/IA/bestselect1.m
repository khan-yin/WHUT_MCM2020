function bestindividuals =bestselect1(individuals,n)
        %这里可以用命令行多试试什么意思，其实就是获取排序后的结果在之前数组中的下标
        [~,Index]=sort(individuals.excellence);
        [~,index1]=min(individuals.fitness);        %找出最优适应度的下标
        bestindividuals=struct('fitness',zeros(1,n),'concentration',zeros(1,n),'excellence',zeros(1,n),'chrom',[]);
        %any(Index(1:n-1)==index1)用于计算记忆库获取容量的n-1个元素内是否有fitness最低值对应的下标出现
        %这个双重达标的算是当前的最优解了，浓度低，fitness低
        if any(Index(1:n-1)==index1)==0
            %如果没有出现则将其加入进去
            bestindividuals.fitness=[individuals.fitness(index1) individuals.fitness(Index(1:n-1))];  
            bestindividuals.concentration=[individuals.concentration(index1) individuals.concentration(Index(1:n-1))];
            bestindividuals.excellence=[individuals.excellence(index1) individuals.excellence(Index(1:n-1))];
            bestindividuals.chrom=[individuals.chrom(index1,:);individuals.chrom(Index(1:n-1),:)];
        else
            %如果出现则直接获取前n个排序结果即可
            bestindividuals.fitness=individuals.fitness(Index(1:n));
            bestindividuals.concentration=individuals.concentration(Index(1:n));
            bestindividuals.excellence=individuals.excellence(Index(1:n));
            bestindividuals.chrom=individuals.chrom(Index(1:n),:);
        end                

end
