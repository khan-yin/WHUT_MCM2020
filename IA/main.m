%�˹������㷨
clc
clear;
%%
popsize=30;%��Ⱥ��ģ
memory_celllist=10;%����ϸ������
MAXGEN=100;%��������
corss_rate=0.5;%�������
multate_rate=0.4;%�������
alpha=0.4;%�����Բ���
distribution_count=6;%����������
Initialpopsize=popsize+memory_celllist;%��ʼ����Ⱥ��Ŀ
%% 
%����Ⱥ��Ϣ����Ϊһ���ṹ��,
%����ÿ������fitness���ɵ����У�
%concentrationÿ�������Ӧ�������͵�Ũ�ȣ�
%excelenceÿ��������������
%chrom[]ÿ�������Ӧ�Ŀ�������
individuals=struct('fitness',zeros(1,Initialpopsize),'concentration',zeros(1,Initialpopsize),'excellence',zeros(1,Initialpopsize),'chrom',[]);
%������ʼ����Ⱥ
for i=1:Initialpopsize
    individuals.chrom(i,:)=randperm(31,distribution_count);%������ɿ���
end

trace=[];   %��¼ÿ�����Ÿ�����Ӧ��

for k=1:MXGEN
    %����������Ⱥ����������
    for i=1:M
        individuals.fitness(i)=fitness(individuals.chrom(i,:)); %�������е��ܷ��ü���
        individuals.concentration(i)=concentration(i,Initialpopsize,individuals); %����Ũ�ȼ���
    end
    %�ۺ��׺ͶȺ�Ũ�����ۿ�������̶ȣ��ó�������ֳ����
    individuals.excellence=excellence(individuals,Initialpopsize,alpha);
    %��¼������Ѹ������Ⱥƽ����Ӧ��
    [best,index1]=min(individuals.fitness);        %�ҳ�������Ӧ��
    bestchrom=individuals.chrom(index1,:);         %�ҳ����Ÿ���
    trace=[trace;best];                   %��¼
    %excellenceԽ��Խ��
    %����4������excellence���γɸ���Ⱥ�����¼���⣨���뾫Ӣ�������ԣ�
    bestindividuals=bestselect1(individuals,memory_celllist);      %���¼����
    fa_individuals=bestselect2(individuals,popsize);         %�γɸ���Ⱥ 
    %����6��ѡ�񡢽��桢����������ټ��������п��壬��������Ⱥ
    individuals=Select(fa_individuals,popsize);       %ѡ��
    individuals.chrom=Cross(corss_rate,fa_individuals.chrom,popsize,length);  %����
    individuals.chrom=Mutation(multate_rate,fa_individuals.chrom,popsize,length);  %����
    %���������п���
    individuals=incorporate(individuals,popsize,bestindividuals,memory_celllist);
end

%�ҳ����Ž�
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

