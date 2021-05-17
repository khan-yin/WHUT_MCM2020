function bestindividuals=bestselect2(individuals,n)
        [~,Index]=sort(individuals.excellence);
        bestindividuals.fitness=individuals.fitness(Index(1:n));
        bestindividuals.concentration=individuals.concentration(Index(1:n));
        bestindividuals.excellence=individuals.excellence(Index(1:n));
        bestindividuals.chrom=individuals.chrom(Index(1:n),:);
end

