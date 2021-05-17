function [R_best,L_best,L_ave,Shortest_Route,Shortest_Length]=ANT_VRP(D_new,Demand_new,Cap,iter_max,m,Alpha,Beta,Rho,Q)
 
%% R_best 各代最佳路线
%% L_best 各代最佳路线的长度
%% L_ave 各代平均距离
%% Shortest_Route 最短路径
%% Shortest_Length 最短路径长度
%% D_new 城市间之间的距离矩阵，为对称矩阵
%% Demand_new 客户需求量
%% Cap 车辆最大载重
%% iter_max 最大迭代次数
%% m 蚂蚁个数
%% Alpha 表征信息素重要程度的参数
%% Beta 表征启发式因子重要程度的参数
%% Rho 信息素蒸发系数
%% Q 信息素增加强度系数
%% num_Vehicle 车辆最大数量
 
n=size(D_new,1);              %地点数量  
T=zeros(m,2*n);           %匹配的路线距离
Eta=ones(m,2*n);          %启发因子
Tau=ones(n,n);            %信息素
Tabu=zeros(m,n);          %禁忌表
Route=zeros(m,2*n);       %路径
L=zeros(m,1);             %总路程
L_best=zeros(iter_max,1);   %各代最佳路线长度
R_best=zeros(iter_max,2*n); %各代最佳路线
nC=1;
 
while nC<=iter_max                   %停止条件
    Eta=zeros(m,2*n);  
    T=zeros(m,2*n); 
    Tabu=zeros(m,n);
    Route=zeros(m,2*n);
    L=zeros(m,1);
     
    %%%%%%==============初始化起点城市（禁忌表）====================
    for i=1:m
        Cap_1=Cap;      %最大装载量
        %禁忌表指定城市编号
        j=1;
        
        j_r=1;          %路线城市指定编号
        while Tabu(i,n)==0
             T=zeros(m,2*n);    %装载量加载矩阵
             Tabu(i,1)=1;       %禁忌表起点位置为1
             Route(i,1)=1;      %路径起点位置为1
             visited=find(Tabu(i,:)>0);   %已访问城市
             num_v=length(visited);        %已访问城市个数
             J=zeros(1,(n-num_v));         %待访问城市加载表
             P=J;                          %待访问城市选择概率分布
             Jc=1;                         %待访问城市选择指针
             for k=1:n                     %城市
                 if length(find(Tabu(i,:)==k))==0    %如果k不是已访问城市代号，就将k加入矩阵J中
                     J(Jc)=k;
                     Jc=Jc+1;
                 end
             end
              
           %%%%%%%=============每只蚂蚁按照选择概率遍历所有城市==================
              
             for k=1:n-num_v               %待访问城市
                  
                   if Cap_1-Demand_new(J(1,k),1)>=0    %如果车辆装载量大于待访问城市需求量
 
                     if Route(i,j_r)==1           %如果每只蚂蚁在起点城市
                         T(i,k)=D_new(1,J(1,k));
                         P(k)=(Tau(1,J(1,k))^Alpha)*((1/T(i,k))^Beta);  %概率计算公式中的分子
                     else                         %如果每只蚂蚁在不在起点城市
                         T(i,k)=D_new(Tabu(i,j),J(1,k));
                         P(k)=(Tau(Tabu(i,visited(end)),J(1,k))^Alpha)*((1/T(i,k))^Beta); %概率计算公式中的分子
                     end
                      
                 else              %如果车辆装载量小于待访问城市需求量
                     T(i,k)=0;
                     P(k)=0;
                 end
             end
              
              % 运输车到下一个地点距离为0时，下一个选择在起点1
             if length(find(T(i,:)>0))==0    %%%当车辆装载量小于待访问城市时，选择起点为1
                 % 重新装满货物
                 Cap_1=Cap;
                 j_r=j_r+1;
                 Route(i,j_r)=1;             
                 L(i)=L(i)+D_new(1,Tabu(i,visited(end)));  
             %没有回仓库
             else 
                 P=P/(sum(P));                 %按照概率原则选取下一个城市
                 Pcum=cumsum(P);               %求累积概率和：cumsum（[1;2;3])=1 3 6,目的在于使得Pcum的值总有大于rand的数
                 Select=find(Pcum>rand);       %按概率选取下一个城市：当累积概率和大于给定的随机数，则选择求和被加上的最后一个城市作为即将访问的城市
                 o_visit=J(1,Select(1));       %待访问城市
                 j=j+1;
                 j_r=j_r+1;
                 Tabu(i,j)=o_visit;             %待访问城市
                 Route(i,j_r)=o_visit;
                 Cap_1=Cap_1-Demand_new(o_visit,1);  %车辆装载剩余量
                 L(i)=L(i)+T(i,Select(1));       %路径长度
             end
        end
         L(i)=L(i)+D_new(Tabu(i,n),1);               %%路径长度
    end
     
    L_best(nC)=min(L);             %最优路径为距离最短的路径
    pos=find(L==min(L));           %找出最优路径对应的位置：即为哪只蚂蚁
    R_best(nC,:)=Route(pos(1),:);  %确定最优路径对应的城市顺序
    L_ave(nC)=mean(L)';            %求第k次迭代的平均距离
     
    Delta_Tau=zeros(n,n);            %Delta_Tau(i,j)表示所有蚂蚁留在第i个城市到第j个城市路径上的信息素增量
    L_zan=L_best(1:nC,1);
    post=find(L_zan==min(L_zan));
    Cities=find(R_best(nC,:)>0);
    num_R=length(Cities);
     
    for k=1:num_R-1          %建立了完整路径后在释放信息素
        Delta_Tau(R_best(nC,k),R_best(nC,k+1))=Delta_Tau(R_best(nC,k),R_best(nC,k+1))+Q/L_best(nC);
    end
        Delta_Tau(R_best(nC,num_R),1)=Delta_Tau(R_best(nC,num_R),1)+Q/L_best(nC);
        Tau=Rho*Tau+Delta_Tau;
         
        nC=nC+1;
end
Shortest_Route=zeros(1,2*n);           %提取最短路径
Shortest_Route(1,:)=R_best(iter_max,:);
Shortest_Route=Shortest_Route(Shortest_Route>0);
Shortest_Route=[Shortest_Route Shortest_Route(1,1)];
Shortest_Length=min(L_best);           %提取最短路径长度
%L_ave=mean(L_best);