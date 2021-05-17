clc;clear all
%% ==============提取数据==============
[xdata,textdata]=xlsread('data.xlsx'); %加载20个城市的数据，数据按照表格中位置保存在Excel文件exp12_3_1.xls中
x_label=xdata(:,3); %第三列为横坐标
y_label=xdata(:,4); %第四列为纵坐标
Demand=xdata(:,2);  %第二列为需求量
C=[x_label y_label];      %坐标矩阵
n=size(C,1);        %n表示节点（客户）个数

%修改下编号
x_changed=zeros(n,1);
y_changed=zeros(n,1);
Demand_changed=zeros(n,1);
%调整下编号，方便运算
for i=1:n
    x_changed(n-i+1,1)=x_label(i,1);
    y_changed(n-i+1,1)=y_label(i,1);
    Demand_changed(n-i+1,1)=Demand(i,1);
end
C=[x_changed y_changed];
%% ==============计算距离矩阵==============
D=zeros(n,n);       %D表示完全图的赋权邻接矩阵，即距离矩阵D初始化
for i=1:n
   for j=1:n
       if i~=j
           D(i,j)=abs((C(i,1)-C(j,1)))+abs((C(i,2)-C(j,2))); %计算两城市之间的距离
       else
           D(i,j)=0;   %i=j, 则距离为0；
       end
       D(j,i)=D(i,j);  %距离矩阵为对称矩阵
   end
end
Cap=6;
%num_Vehicle = sum(Demand(1:n,1))/Cap;
Alpha=1;Beta=5;Rho=0.75;iter_max=500;Q=10;m=20;  %Cap为车辆最大载重
[R_best,L_best,L_ave,Shortest_Route,Shortest_Length]=ANT_VRP(D,Demand_changed,Cap,iter_max,m,Alpha,Beta,Rho,Q); %蚁群算法求解
Shortest_Route    %提取最优路线
Shortest_Length                      %提取最短路径长度
 
%% ==============作图==============
figure(1)   %作迭代收敛曲线图
x_changed=linspace(0,iter_max,iter_max);
y_changed=L_best(:,1);
plot(x_changed,y_changed);
xlabel('迭代次数'); ylabel('最短路径长度');
 
figure(2)   %作最短路径图
plot([C(Shortest_Route,1)],[C(Shortest_Route,2)],'o-');
grid on
for i =1:size(C,1)
    text(C(i,1),C(i,2),['   ' num2str(i-1)]);
end
xlabel('客户所在横坐标'); ylabel('客户所在纵坐标');
%xlswrite('shortestRoute.xlsx',Shortest_Route_1);
xlswrite('shortestRoute.xlsx',Shortest_Route);