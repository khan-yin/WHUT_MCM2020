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
%% =====第三问画图
figure(2)   %作最短路径图
tmp_1=[1,14,10,1,13,1,7,2,9,1,12,1,3,6,1];%4吨车
tmp_2=[1,18,20,19,1,17,16,15,8,1,11,4,5,1];%6吨车
plot([C(tmp_2,1)],[C(tmp_2,2)],'o-r');
hold on;
plot([C(tmp_1,1)],[C(tmp_1,2)],'o-b');
grid on

for i =1:size(C,1)
    text(C(i,1),C(i,2),['   ' num2str(i)]);
end
xlabel('客户所在横坐标'); ylabel('客户所在纵坐标');
%xlswrite('shortestRoute.xlsx',Shortest_Route_1);
%xlswrite('shortestRoute.xlsx',Shortest_Route);