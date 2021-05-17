function similarity = similar(individual1,individual2)
%计算个体individual1,individual2之间的相似度
% individual1,individual2 input 两个个体
% similarity output 相似度，相同个体所占的比例
len=length(individual1);
k=zeros(1,len);
for i =1:len
    if find(individual1(i)==individual2)%这个操作一定要多体会体会
        k(i)=1;
    end
end
similarity=sum(k)/len;
end

