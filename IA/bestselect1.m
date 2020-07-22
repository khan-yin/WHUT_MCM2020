function bestindividuals =bestselect1(individuals,n)
        %��������������ж�����ʲô��˼����ʵ���ǻ�ȡ�����Ľ����֮ǰ�����е��±�
        [~,Index]=sort(individuals.excellence);
        [~,index1]=min(individuals.fitness);        %�ҳ�������Ӧ�ȵ��±�
        bestindividuals=struct('fitness',zeros(1,n),'concentration',zeros(1,n),'excellence',zeros(1,n),'chrom',[]);
        %any(Index(1:n-1)==index1)���ڼ��������ȡ������n-1��Ԫ�����Ƿ���fitness���ֵ��Ӧ���±����
        %���˫�ش������ǵ�ǰ�����Ž��ˣ�Ũ�ȵͣ�fitness��
        if any(Index(1:n-1)==index1)==0
            %���û�г�����������ȥ
            bestindividuals.fitness=[individuals.fitness(index1) individuals.fitness(Index(1:n-1))];  
            bestindividuals.concentration=[individuals.concentration(index1) individuals.concentration(Index(1:n-1))];
            bestindividuals.excellence=[individuals.excellence(index1) individuals.excellence(Index(1:n-1))];
            bestindividuals.chrom=[individuals.chrom(index1,:);individuals.chrom(Index(1:n-1),:)];
        else
            %���������ֱ�ӻ�ȡǰn������������
            bestindividuals.fitness=individuals.fitness(Index(1:n));
            bestindividuals.concentration=individuals.concentration(Index(1:n));
            bestindividuals.excellence=individuals.excellence(Index(1:n));
            bestindividuals.chrom=individuals.chrom(Index(1:n),:);
        end                

end