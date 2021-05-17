clc;clear all
%% ==============��ȡ����==============
[xdata,textdata]=xlsread('data.xlsx'); %����20�����е����ݣ����ݰ��ձ����λ�ñ�����Excel�ļ�exp12_3_1.xls��
x_label=xdata(:,3); %������Ϊ������
y_label=xdata(:,4); %������Ϊ������
Demand=xdata(:,2);  %�ڶ���Ϊ������
C=[x_label y_label];      %�������
n=size(C,1);        %n��ʾ�ڵ㣨�ͻ�������

%�޸��±��
x_changed=zeros(n,1);
y_changed=zeros(n,1);
Demand_changed=zeros(n,1);
%�����±�ţ���������
for i=1:n
    x_changed(n-i+1,1)=x_label(i,1);
    y_changed(n-i+1,1)=y_label(i,1);
    Demand_changed(n-i+1,1)=Demand(i,1);
end
C=[x_changed y_changed];
%% ==============����������==============
D=zeros(n,n);       %D��ʾ��ȫͼ�ĸ�Ȩ�ڽӾ��󣬼��������D��ʼ��
for i=1:n
   for j=1:n
       if i~=j
           D(i,j)=abs((C(i,1)-C(j,1)))+abs((C(i,2)-C(j,2))); %����������֮��ľ���
       else
           D(i,j)=0;   %i=j, �����Ϊ0��
       end
       D(j,i)=D(i,j);  %�������Ϊ�Գƾ���
   end
end
Cap=6;
%num_Vehicle = sum(Demand(1:n,1))/Cap;
Alpha=1;Beta=5;Rho=0.75;iter_max=500;Q=10;m=20;  %CapΪ�����������
[R_best,L_best,L_ave,Shortest_Route,Shortest_Length]=ANT_VRP(D,Demand_changed,Cap,iter_max,m,Alpha,Beta,Rho,Q); %��Ⱥ�㷨���
Shortest_Route    %��ȡ����·��
Shortest_Length                      %��ȡ���·������
 
%% ==============��ͼ==============
figure(1)   %��������������ͼ
x_changed=linspace(0,iter_max,iter_max);
y_changed=L_best(:,1);
plot(x_changed,y_changed);
xlabel('��������'); ylabel('���·������');
 
figure(2)   %�����·��ͼ
plot([C(Shortest_Route,1)],[C(Shortest_Route,2)],'o-');
grid on
for i =1:size(C,1)
    text(C(i,1),C(i,2),['   ' num2str(i-1)]);
end
xlabel('�ͻ����ں�����'); ylabel('�ͻ�����������');
%xlswrite('shortestRoute.xlsx',Shortest_Route_1);
xlswrite('shortestRoute.xlsx',Shortest_Route);