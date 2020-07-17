function concentration = concentration(k,Initalpopsize,individuals)
%计算当前种群浓度
%
%k 第k个抗体
%Initalpopsize 种群规模
%individuals 个体
%concentration 浓度计算
concentrationCount = 0;
for i=1:Initalpopsize
    similarty=similar(individuals.chrom(i,:),individuals.chrom(k,:));
    if similarty>0.7:
        concentrationCount=concentrationCount+1;
    end
end
concentration=concentrationCount/Initalpopsize;
end

