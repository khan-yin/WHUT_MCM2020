clc;clear all;
%% ==============提取数据==============
[xdata,textdata]=xlsread('data.xlsx'); %加载20个城市的数据，数据按照表格中位置保存在Excel文件exp12_3_1.xls中
x_label=xdata(:,3); %第三列为横坐标
y_label=xdata(:,4); %第四列为纵坐标
Demand=xdata(:,2);  %第二列为需求量
C=[x_label y_label];      %坐标矩阵
n=size(C,1);        %n表示节点（客户）个数
%修改下编号
x_changed=zeros(n,1);   %改变后的坐标
y_changed=zeros(n,1);   %改变后的坐标
Demand_changed=zeros(n,1);
%调整下编号，方便运算
for i=1:n
    x_changed(n-i+1,1)=x_label(i,1);
    y_changed(n-i+1,1)=y_label(i,1);
    Demand_changed(n-i+1,1)=Demand(i,1);
end
C=[x_changed y_changed];
Shortest_Length_6t=xlsread('shortestLength_6t.xlsx'); %6吨获取最短路径长度
D=zeros(n,n);
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
shortest_Route = xlsread('shortestRoute.xlsx');                     %获取问题2中得到的最佳路径
%num_R = size(shortest_Route,2);        %有多少个节点
num_Vehicle_6t = size(find(shortest_Route(1,:)==1),2)-1;       %获取6吨车辆数目
Route_6t = zeros(num_Vehicle_6t,n);    %题目二中各辆6吨车的路线
Route_4t = zeros(ceil(num_Vehicle_6t*6/4),n);      %4吨运输车的运输线路
number_orign=find(shortest_Route(1,:)==1);      %找到每个6吨车起始点在路线中的编号

%% ==============================数据运算===================

tmp_1=zeros(n,n);
size_Route=zeros(num_Vehicle_6t,1);%每条路线结点数目
for i=1:num_Vehicle_6t
    tmp_1(i,number_orign(1,i):number_orign(1,i+1))=shortest_Route(1,number_orign(1,i):number_orign(1,i+1));
    size_Route(i,1)=size(find(tmp_1(i,:)),2);
end

for i=1:num_Vehicle_6t
    Route_6t(i,1:size_Route(i,1))=shortest_Route(1,number_orign(1,i):number_orign(1,i+1));
end
%获取6吨车每条路线的长度
Length_6t =zeros(num_Vehicle_6t,1);
for i=1:num_Vehicle_6t
    for s=1:n
        if Route_6t(i,s)==0
            break;
        else
            
            Length_6t(i,1)=Length_6t(i,1)+D(Route_6t(i,s),Route_6t(i,s));
        end
        
    end
end

Replaced = zeros(1,num_Vehicle_6t);%可以被取代的列表

sum_Demand = zeros(num_Vehicle_6t,1); %获取每条路线的总需求
for i=1:num_Vehicle_6t
    sum_Demand(i,1)=sum(Demand_changed(Route_6t(i,1:size_Route(i,1)),1));
end

%先找每条线路的总需求量小于等于4吨的
for i=1:num_Vehicle_6t
    if sum_Demand(i,1)<=4
        Replaced(1,i)=i;
    end
end
Replaced_1 = find(Replaced);%找到要替代的车辆编号
num_Vehicle_4t=0;%4吨车数量

if size(Replaced_1,2)>0 %有可以直接替代的车辆
    num_Vehicle_4t = size(Replaced_1,2); %更新4吨车辆数量
    num_Vehicle_6t = num_Vehicle_6t-1;  
    
    iter_flag=0; %迭代次数标志
    
    tmp_L = zeros(1,10);    %保存每次选取两个不同6吨车辆后最佳路径的最短长度
    tmp_R = zeros(10,n);    %保存每次选取两个不同6吨车辆后最佳路径
    All_Route = zeros(15,20);%保存每种情况下的映射
    set_selectedVehicle=[1,2;1,3;1,4;1,5;2,3;2,4;2,5;3,4;3,5;4,5];
    while iter_flag<1000 %随机抽取1000次
        tmp_Vehicle=randperm(5,2); %随机抽取两个车辆
        
        %将这些6吨车遍历的地点放入到4吨车即将遍历的地点里
        Route_4t(tmp_Vehicle(1,1),:) = Route_6t(tmp_Vehicle(1,1),:);
        Route_4t(tmp_Vehicle(1,2),:) = Route_6t(tmp_Vehicle(1,2),:);
        %需要遍历的点
        tmp_Route = [Route_4t(tmp_Vehicle(1,1),:) Route_4t(tmp_Vehicle(1,2),:)];
        tmp_Route(find(tmp_Route==0))=[];
        tmp_Route=unique(tmp_Route);
        %4吨车需要遍历的地点的数量
        num_4t = size(tmp_Route,2);
        %构建新的距离矩阵
        D_new=zeros(size(tmp_Route,2),size(tmp_Route,2));
        for i=1:size(tmp_Route,2)
            for j=1:size(tmp_Route,2)
                if tmp_Route(1,j)~=tmp_Route(1,i)
                    D_new(i,j)=abs((C(tmp_Route(1,i),1)-C(tmp_Route(1,j),1)))+abs((C(tmp_Route(1,i),2)-C(tmp_Route(1,j),2))); %计算两城市之间的距离
                else
                    D_new(i,j)=0;   %i=j, 则距离为0；
                end
                D_new(j,i)=D_new(i,j);  %距离矩阵为对称矩阵
            end
        end
        %构建新的需求矩阵
        Demand_new=zeros(num_4t,1);
        for i=1:num_4t
            
            Demand_new(i,1)=Demand_changed(tmp_Route(1,i),1);
            
        end
        Alpha=1;Beta=5;Rho=0.75;iter_max=500;Q=10;m=20;Cap=4;  %Cap为车辆最大载重
        [R_best,L_best,L_ave,Shortest_Route,Shortest_Length]=ANT_VRP(D_new,Demand_new,Cap,iter_max,m,Alpha,Beta,Rho,Q); %蚁群算法求解4吨车最佳路径
        %将迭代结果保存下来
        
        %一次抽取后的长度
        for o=1:10  %判断抽取的是哪种情况
            if tmp_Vehicle(1,1)==set_selectedVehicle(o,1)||tmp_Vehicle(1,1)==set_selectedVehicle(o,2)&&(tmp_Vehicle(1,2)==set_selectedVehicle(o,1)||tmp_Vehicle(1,2)==set_selectedVehicle(o,2))
                tmp_L(1,o)=Shortest_Length;
                tmp_R(o,1:size(Shortest_Route,2))=Shortest_Route;
                All_Route(o,1:size(tmp_Route,2))=tmp_Route;
                tmp_Length_1 = Shortest_Length_6t-Length_6t(tmp_Vehicle(1,1),1)-Length_6t(tmp_Vehicle(1,2),1);%临时长度
                tmp_Length_2 = Shortest_Length_6t-Shortest_Length;
                if tmp_Length_1 < tmp_Length_2
                    
                    tmp_num_Vehicle_4t = size(find(Shortest_Route(1,:)==1),2)-1;%获取4吨运输车数量
                    num_Vehicle_4t = num_Vehicle_4t+tmp_num_Vehicle_4t;
                    num_Vehicle_6t = num_Vehicle_6t-2;
                end
            end
        end
        iter_flag=iter_flag+1;
    end
else
    iter_flag=0; %迭代次数标志
    
    tmp_L = zeros(1,15);    %保存每次选取两个不同6吨车辆后最佳路径的最短长度
    tmp_R = zeros(15,n);    %保存每次选取两个不同6吨车辆后最佳路径
    tmp_Y = zeros(15,n);    %保存每次选取的映射
    set_selectedVehicle=[1,2;1,3;1,4;1,5;1,6;2,3;2,4;2,5;2,6;3,4;3,5;3,6;4,5;4,6;5,6];
    while iter_flag<100 %随机抽取100次
        tmp_Vehicle=randperm(6,2); %随机抽取两个车辆
        
        %将这些6吨车遍历的地点放入到4吨车即将遍历的地点里
        Route_4t(tmp_Vehicle(1,1),:) = Route_6t(tmp_Vehicle(1,1),:);
        Route_4t(tmp_Vehicle(1,2),:) = Route_6t(tmp_Vehicle(1,2),:);
        %需要遍历的点
        tmp_Route = [Route_4t(tmp_Vehicle(1,1),:) Route_4t(tmp_Vehicle(1,2),:)];
        tmp_Route(find(tmp_Route==0))=[];
        tmp_Route=unique(tmp_Route);
        %4吨车需要遍历的地点的数量
        num_4t = size(tmp_Route,2);
        %构建新的距离矩阵
        D_new=zeros(size(tmp_Route,2),size(tmp_Route,2));
        for i=1:size(tmp_Route,2)
            for j=1:size(tmp_Route,2)
                if tmp_Route(1,j)~=tmp_Route(1,i)
                    D_new(i,j)=abs((C(tmp_Route(1,i),1)-C(tmp_Route(1,j),1)))+abs((C(tmp_Route(1,i),2)-C(tmp_Route(1,j),2))); %计算两城市之间的距离
                else
                    D_new(i,j)=0;   %i=j, 则距离为0；
                end
                D_new(j,i)=D_new(i,j);  %距离矩阵为对称矩阵
            end
        end
        %构建新的需求矩阵
        Demand_new=zeros(num_4t,1);
        for i=1:num_4t
            
            Demand_new(i,1)=Demand_changed(tmp_Route(1,i),1);
            
        end
        Alpha=1;Beta=5;Rho=0.75;iter_max=500;Q=10;m=20;Cap=4;  %Cap为车辆最大载重
        [R_best,L_best,L_ave,Shortest_Route,Shortest_Length]=ANT_VRP(D_new,Demand_new,Cap,iter_max,m,Alpha,Beta,Rho,Q); %蚁群算法求解4吨车最佳路径
        
        %一次抽取后的长度
        for o=1:15  %判断抽取的是哪种情况
            if tmp_Vehicle(1,1)==set_selectedVehicle(o,1)||tmp_Vehicle(1,1)==set_selectedVehicle(o,2)&&(tmp_Vehicle(1,2)==set_selectedVehicle(o,1)||tmp_Vehicle(1,2)==set_selectedVehicle(o,2))
                tmp_L(1,o)=Shortest_Length;
                tmp_R(o,1:size(Shortest_Route,2))=Shortest_Route;
                tmp_Length_1 = Shortest_Length_6t-Length_6t(tmp_Vehicle(1,1),1)-Length_6t(tmp_Vehicle(1,2),1);%临时长度
                tmp_Length_2 = Shortest_Length_6t-Shortest_Length;
                if tmp_Length_1 < tmp_Length_2
                    
                    tmp_num_Vehicle_4t = size(find(Shortest_Route(1,:)==1),2)-1;%获取4吨运输车数量
                    num_Vehicle_4t = num_Vehicle_4t+tmp_num_Vehicle_4t;
                    num_Vehicle_6t = num_Vehicle_6t-2;
                end
            end
        end
        iter_flag=iter_flag+1;
    end
end
num_Vehicle_6t
num_Vehicle_4t
xlswrite('tmp_4t.xlsx',tmp_R);
xlswrite('tmp_4tL.xlsx',tmp_L);
