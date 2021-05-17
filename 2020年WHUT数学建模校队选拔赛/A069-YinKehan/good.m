%????Author:??? ����2��

%????Use:?????? �����Ŵ��㷨����Լ�����������������䳵���ĳ���·���Ż����⣬����4����Ŀ����Ż����⣬ת����һ����Ŀ���Ż�����

%??????????????? ������������޸�������

%????????????????????????????????????? popsize:��Ⱥ��С

%????????????????????????????????????? T:��������???????????????????? ?

%??????????????? �����????????

%????????????????????????????????????? route������·��

%????Remark��??����qq��778961303��������������ѯ

tic;

clc;close all;
clear all;%��S����������



format loose


l=20;%���ͳ���

vehicle=[1750 25;

        2000 31.5;

        2200 54;

        3040 65];%��ͬ���ͳ��������أ����������

num_vehicle=[3 4 5 6]; %��ͬ����������

limit=[]; %���غ����������

for i=1:length(num_vehicle)
    temp=rep(vehicle(i,:),[num_vehicle(i) 1]);
    limit=[limit;temp];

end

G=limit(:,1); %��������������

V=limit(:,2); %�������������

N=32;%��Ҫ���ͽڵ�������

% G=1995;%^���������ƣ���

% V=30;%����������ƣ���

L=250;%^��·�����ƣ���

alpha=1;

Ws=130;%��������ʱ�����ؼ���ĳɱ�����

Vs=140;%��������ʱ���������ĳɱ�(Ԫ)����

%�������ͳɱ�

Csl=0.3;%��λ��������ȼ�Ϸ���(Ԫ/km)����

Cs2=2;%�����۾�(Ԫ/km)����

Cs3=0.03;%��̥����(Ԫ/km)����

Cs4=1.38;%�ͷ�(Ԫ/km)

Cs5=1.24;%��·�ѣ�Ԫ/km��

Cs6=0.15;%�����ճ�ά���ѣ�Ԫ/km��

%��������������سɱ�����

Fsl=100;%װж����(Ԫ/��)����

Fs2=8.9;%���շ�(Ԫ/��)����

Fs3=30; %�ĵ�����(Ԫ/��)����

Fs4=50;%���Ͳ��Ϸ���(Ԫ/��)��

while gen<=T

%%�Ŵ��㷨ѡ�񣍣�

FitnV=ranking(Value);%��������Ӧ��ֵ����

Chrom=select('sus',Chrom,FitnV,1);%��ѡ�񣍣�

Chrom=mutationGA(Chrom,popsize,PM,N);%����Ⱥ����,������죍��

Chrom=crossGA(Chrom,popsize,PC,N);%����Ⱥ����,����佻�棍��

%%�������ţ���

[vl,index1]=min(Value);

gen=gen+1;

trace1(gen,1)=Value(index1);

trace1(gen,2)=mean(Value);

%%��¼���ţ���

if gen == 1

bestChrom1=Chrom(index1,:);%��¼����1������Ⱦɫ�壍��

bestValuel=vl;%��¼����1������ֵ����

end

if bestValuel>vl

bestValuel = vl;%��¼����1������ֵ����

bestChrom1=Chrom(index1,:);

end

waitbar(gen/T,wait_hand);%ÿѭ��һ�θ���һ�ν���������

end

delete(wait_hand);%ִ�����ɾ���ý���������

%��ʾ�������

disp(['���Ŀ�꺯��',num2str(bestValuel)]);


disp(['���Ⱦɫ��',num2str(bestChrom1)]);


disp(['���ó�������',num2str(min(y4(:,1)))]);

%% ���

%%ת��Ϊ��Ҫ�ĸ�ʽ������ļ�����

s=bestChrom1;

needdataW=needdata(1,s);

needdataV=needdata(2,s);

route =divideroute(s,needdataW,needdataV,G,V,L,distdata);%��·�����֣���

%��ÿ������·�̣����������ʣ����������

index = find(route==0);

for i = 1:length(index)-1

temp_route=route(index(i):index(i+1));%ÿ������·��

temp_y1=objectfun_distance(temp_route,distdata);%ÿ������·��

[temp_y2 temp_y3] = pervehicle(temp_route,needdata,G(i),V(i)); %ÿ���������ء����������

disp(['��' num2str(i) '??? ������·����' num2str(temp_route)]);

disp(['��' num2str(i) '?? ������ʻ·�̣�' num2str(temp_y1)]);

disp(['��' num2str(i) '���������������ʣ�' num2str(temp_y2)]);

disp(['��' num2str(i) '��������������ʣ�' num2str(temp_y3)]);

end

disp(['���㷨�Ż��õ������·��:',num2str(route)]);


disp(['�Ŵ��㷨�Ż��õ���Ŀ�꺯��ֵ:',num2str(bestValuel)]);



disp(['�Ŵ��㷨�Ż��õ�������ʻ���:',num2str(min(yl(:,1)))]);



disp(['�Ŵ��㷨�Ż��õ��ĳ�������δ������',num2str(min(y2(1,:)))]);



disp(['�Ŵ��㷨�Ż��õ��ĳ����ݻ�δ������',num2str(min(y3(1,:)))]);



%% ��ͼ

%%��Ŀ�꺯������

figure

plot(trace1(:,1),'b-');

hold on;

plot(trace1(:,2),'r-');

legend('Ŀ�꺯������ֵ','Ŀ�꺯����ֵ');

xlabel('��������')

ylabel('Ŀ�꺯��')

title('�Ŵ��㷨�Ż��ɱ���������')