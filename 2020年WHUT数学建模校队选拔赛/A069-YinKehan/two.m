clc;clear all
%% ==============��ȡ����==============
[xdata,textdata]=xlsread('exp12_3_2.xlsx'); %����20�����е����ݣ����ݰ��ձ����λ�ñ�����Excel�ļ�exp12_3_1.xls��
x_label=xdata(:,2); %�ڶ���Ϊ������
y_label=xdata(:,3); %������Ϊ������
Demand=xdata(:,4);  %������Ϊ������
C=[x_label y_label];      %�������
n=size(C,1);        %n��ʾ�ڵ㣨�ͻ�������
%% ==============����������==============
D=zeros(n,n);       %D��ʾ��ȫͼ�ĸ�Ȩ�ڽӾ��󣬼��������D��ʼ��
for i=1:n
   for j=1:n
       if i~=j
           D(i,j)=((C(i,1)-C(j,1))^2+(C(i,2)-C(j,2))^2)^0.5; %����������֮��ľ���
       else
           D(i,j)=0;   %i=j, �����Ϊ0��
       end
       D(j,i)=D(i,j);  %�������Ϊ�Գƾ���
   end
end

Alpha=1;Beta=5;Rho=0.75;iter_max=100;Q=10;Cap=1;m=20;  %CapΪ�����������
[R_best,L_best,L_ave,Shortest_Route,Shortest_Length]=ANT_VRP(D,Demand,Cap,iter_max,m,Alpha,Beta,Rho,Q); %��Ⱥ�㷨���VRP����ͨ�ú�����������׹���
Shortest_Route_1=Shortest_Route-1    %��ȡ����·��
Shortest_Length                      %��ȡ���·������
 
%% ==============��ͼ==============
figure(1)   %��������������ͼ
x=linspace(0,iter_max,iter_max);
y=L_best(:,1);
plot(x,y);
xlabel('��������'); ylabel('���·������');
 
figure(2)   %�����·��ͼ
plot([C(Shortest_Route,1)],[C(Shortest_Route,2)],'o-');
grid on
for i =1:size(C,1)
    text(C(i,1),C(i,2),['   ' num2str(i-1)]);
end
xlabel('�ͻ����ں�����'); ylabel('�ͻ�����������');