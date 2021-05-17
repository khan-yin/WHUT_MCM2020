clc;clear all;
%% ==============��ȡ����==============
[xdata,textdata]=xlsread('data.xlsx'); %����20�����е����ݣ����ݰ��ձ����λ�ñ�����Excel�ļ�exp12_3_1.xls��
x_label=xdata(:,3); %������Ϊ������
y_label=xdata(:,4); %������Ϊ������
Demand=xdata(:,2);  %�ڶ���Ϊ������
C=[x_label y_label];      %�������
n=size(C,1);        %n��ʾ�ڵ㣨�ͻ�������
%�޸��±��
x_changed=zeros(n,1);   %�ı�������
y_changed=zeros(n,1);   %�ı�������
Demand_changed=zeros(n,1);
%�����±�ţ���������
for i=1:n
    x_changed(n-i+1,1)=x_label(i,1);
    y_changed(n-i+1,1)=y_label(i,1);
    Demand_changed(n-i+1,1)=Demand(i,1);
end
C=[x_changed y_changed];
Shortest_Length_6t=xlsread('shortestLength_6t.xlsx'); %6�ֻ�ȡ���·������
D=zeros(n,n);
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
shortest_Route = xlsread('shortestRoute.xlsx');                     %��ȡ����2�еõ������·��
%num_R = size(shortest_Route,2);        %�ж��ٸ��ڵ�
num_Vehicle_6t = size(find(shortest_Route(1,:)==1),2)-1;       %��ȡ6�ֳ�����Ŀ
Route_6t = zeros(num_Vehicle_6t,n);    %��Ŀ���и���6�ֳ���·��
Route_4t = zeros(ceil(num_Vehicle_6t*6/4),n);      %4�����䳵��������·
number_orign=find(shortest_Route(1,:)==1);      %�ҵ�ÿ��6�ֳ���ʼ����·���еı��

%% ==============================��������===================

tmp_1=zeros(n,n);
size_Route=zeros(num_Vehicle_6t,1);%ÿ��·�߽����Ŀ
for i=1:num_Vehicle_6t
    tmp_1(i,number_orign(1,i):number_orign(1,i+1))=shortest_Route(1,number_orign(1,i):number_orign(1,i+1));
    size_Route(i,1)=size(find(tmp_1(i,:)),2);
end

for i=1:num_Vehicle_6t
    Route_6t(i,1:size_Route(i,1))=shortest_Route(1,number_orign(1,i):number_orign(1,i+1));
end
%��ȡ6�ֳ�ÿ��·�ߵĳ���
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

Replaced = zeros(1,num_Vehicle_6t);%���Ա�ȡ�����б�

sum_Demand = zeros(num_Vehicle_6t,1); %��ȡÿ��·�ߵ�������
for i=1:num_Vehicle_6t
    sum_Demand(i,1)=sum(Demand_changed(Route_6t(i,1:size_Route(i,1)),1));
end

%����ÿ����·����������С�ڵ���4�ֵ�
for i=1:num_Vehicle_6t
    if sum_Demand(i,1)<=4
        Replaced(1,i)=i;
    end
end
Replaced_1 = find(Replaced);%�ҵ�Ҫ����ĳ������
num_Vehicle_4t=0;%4�ֳ�����

if size(Replaced_1,2)>0 %�п���ֱ������ĳ���
    num_Vehicle_4t = size(Replaced_1,2); %����4�ֳ�������
    num_Vehicle_6t = num_Vehicle_6t-1;  
    
    iter_flag=0; %����������־
    
    tmp_L = zeros(1,10);    %����ÿ��ѡȡ������ͬ6�ֳ��������·������̳���
    tmp_R = zeros(10,n);    %����ÿ��ѡȡ������ͬ6�ֳ��������·��
    All_Route = zeros(15,20);%����ÿ������µ�ӳ��
    set_selectedVehicle=[1,2;1,3;1,4;1,5;2,3;2,4;2,5;3,4;3,5;4,5];
    while iter_flag<1000 %�����ȡ1000��
        tmp_Vehicle=randperm(5,2); %�����ȡ��������
        
        %����Щ6�ֳ������ĵص���뵽4�ֳ����������ĵص���
        Route_4t(tmp_Vehicle(1,1),:) = Route_6t(tmp_Vehicle(1,1),:);
        Route_4t(tmp_Vehicle(1,2),:) = Route_6t(tmp_Vehicle(1,2),:);
        %��Ҫ�����ĵ�
        tmp_Route = [Route_4t(tmp_Vehicle(1,1),:) Route_4t(tmp_Vehicle(1,2),:)];
        tmp_Route(find(tmp_Route==0))=[];
        tmp_Route=unique(tmp_Route);
        %4�ֳ���Ҫ�����ĵص������
        num_4t = size(tmp_Route,2);
        %�����µľ������
        D_new=zeros(size(tmp_Route,2),size(tmp_Route,2));
        for i=1:size(tmp_Route,2)
            for j=1:size(tmp_Route,2)
                if tmp_Route(1,j)~=tmp_Route(1,i)
                    D_new(i,j)=abs((C(tmp_Route(1,i),1)-C(tmp_Route(1,j),1)))+abs((C(tmp_Route(1,i),2)-C(tmp_Route(1,j),2))); %����������֮��ľ���
                else
                    D_new(i,j)=0;   %i=j, �����Ϊ0��
                end
                D_new(j,i)=D_new(i,j);  %�������Ϊ�Գƾ���
            end
        end
        %�����µ��������
        Demand_new=zeros(num_4t,1);
        for i=1:num_4t
            
            Demand_new(i,1)=Demand_changed(tmp_Route(1,i),1);
            
        end
        Alpha=1;Beta=5;Rho=0.75;iter_max=500;Q=10;m=20;Cap=4;  %CapΪ�����������
        [R_best,L_best,L_ave,Shortest_Route,Shortest_Length]=ANT_VRP(D_new,Demand_new,Cap,iter_max,m,Alpha,Beta,Rho,Q); %��Ⱥ�㷨���4�ֳ����·��
        %�����������������
        
        %һ�γ�ȡ��ĳ���
        for o=1:10  %�жϳ�ȡ�����������
            if tmp_Vehicle(1,1)==set_selectedVehicle(o,1)||tmp_Vehicle(1,1)==set_selectedVehicle(o,2)&&(tmp_Vehicle(1,2)==set_selectedVehicle(o,1)||tmp_Vehicle(1,2)==set_selectedVehicle(o,2))
                tmp_L(1,o)=Shortest_Length;
                tmp_R(o,1:size(Shortest_Route,2))=Shortest_Route;
                All_Route(o,1:size(tmp_Route,2))=tmp_Route;
                tmp_Length_1 = Shortest_Length_6t-Length_6t(tmp_Vehicle(1,1),1)-Length_6t(tmp_Vehicle(1,2),1);%��ʱ����
                tmp_Length_2 = Shortest_Length_6t-Shortest_Length;
                if tmp_Length_1 < tmp_Length_2
                    
                    tmp_num_Vehicle_4t = size(find(Shortest_Route(1,:)==1),2)-1;%��ȡ4�����䳵����
                    num_Vehicle_4t = num_Vehicle_4t+tmp_num_Vehicle_4t;
                    num_Vehicle_6t = num_Vehicle_6t-2;
                end
            end
        end
        iter_flag=iter_flag+1;
    end
else
    iter_flag=0; %����������־
    
    tmp_L = zeros(1,15);    %����ÿ��ѡȡ������ͬ6�ֳ��������·������̳���
    tmp_R = zeros(15,n);    %����ÿ��ѡȡ������ͬ6�ֳ��������·��
    tmp_Y = zeros(15,n);    %����ÿ��ѡȡ��ӳ��
    set_selectedVehicle=[1,2;1,3;1,4;1,5;1,6;2,3;2,4;2,5;2,6;3,4;3,5;3,6;4,5;4,6;5,6];
    while iter_flag<100 %�����ȡ100��
        tmp_Vehicle=randperm(6,2); %�����ȡ��������
        
        %����Щ6�ֳ������ĵص���뵽4�ֳ����������ĵص���
        Route_4t(tmp_Vehicle(1,1),:) = Route_6t(tmp_Vehicle(1,1),:);
        Route_4t(tmp_Vehicle(1,2),:) = Route_6t(tmp_Vehicle(1,2),:);
        %��Ҫ�����ĵ�
        tmp_Route = [Route_4t(tmp_Vehicle(1,1),:) Route_4t(tmp_Vehicle(1,2),:)];
        tmp_Route(find(tmp_Route==0))=[];
        tmp_Route=unique(tmp_Route);
        %4�ֳ���Ҫ�����ĵص������
        num_4t = size(tmp_Route,2);
        %�����µľ������
        D_new=zeros(size(tmp_Route,2),size(tmp_Route,2));
        for i=1:size(tmp_Route,2)
            for j=1:size(tmp_Route,2)
                if tmp_Route(1,j)~=tmp_Route(1,i)
                    D_new(i,j)=abs((C(tmp_Route(1,i),1)-C(tmp_Route(1,j),1)))+abs((C(tmp_Route(1,i),2)-C(tmp_Route(1,j),2))); %����������֮��ľ���
                else
                    D_new(i,j)=0;   %i=j, �����Ϊ0��
                end
                D_new(j,i)=D_new(i,j);  %�������Ϊ�Գƾ���
            end
        end
        %�����µ��������
        Demand_new=zeros(num_4t,1);
        for i=1:num_4t
            
            Demand_new(i,1)=Demand_changed(tmp_Route(1,i),1);
            
        end
        Alpha=1;Beta=5;Rho=0.75;iter_max=500;Q=10;m=20;Cap=4;  %CapΪ�����������
        [R_best,L_best,L_ave,Shortest_Route,Shortest_Length]=ANT_VRP(D_new,Demand_new,Cap,iter_max,m,Alpha,Beta,Rho,Q); %��Ⱥ�㷨���4�ֳ����·��
        
        %һ�γ�ȡ��ĳ���
        for o=1:15  %�жϳ�ȡ�����������
            if tmp_Vehicle(1,1)==set_selectedVehicle(o,1)||tmp_Vehicle(1,1)==set_selectedVehicle(o,2)&&(tmp_Vehicle(1,2)==set_selectedVehicle(o,1)||tmp_Vehicle(1,2)==set_selectedVehicle(o,2))
                tmp_L(1,o)=Shortest_Length;
                tmp_R(o,1:size(Shortest_Route,2))=Shortest_Route;
                tmp_Length_1 = Shortest_Length_6t-Length_6t(tmp_Vehicle(1,1),1)-Length_6t(tmp_Vehicle(1,2),1);%��ʱ����
                tmp_Length_2 = Shortest_Length_6t-Shortest_Length;
                if tmp_Length_1 < tmp_Length_2
                    
                    tmp_num_Vehicle_4t = size(find(Shortest_Route(1,:)==1),2)-1;%��ȡ4�����䳵����
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
