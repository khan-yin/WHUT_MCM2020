clc
clear;
%% 
load('china.mat');
plotcities(province, border, city);
cityaccount=length(city);%城市数量
dis=distancematrix(city)%距离矩阵

route =randperm(cityaccount);%路径格式
temperature=1000;%初始化温度
cooling_rate=0.95;%温度下降比率
item=1;%用来控制降温的循环记录次数
distance=totaldistance(route,dis);%计算路径总长度
temperature_iterations = 1;
% This is a flag used to plot the current route after 200 iterations
plot_iterations = 1;

plotroute(city, route, distance,temperature);

while temperature>1.0 %循环条件
    temp_route=change(route,'reverse');%产生扰动。分子序列变化
%     fprintf("%d\n",temp_route(1));
    temp_distance=totaldistance(temp_route,dis);%产生变化后的长度
    dist=temp_distance-distance;%两个路径的差距
    if(dist<0)||(rand < exp(-dist/(temperature)))
        route=temp_route;
        distance=temp_distance;
        item=item+1;
        temperature_iterations=temperature_iterations+1;
        plot_iterations=plot_iterations+1;
    end
    if temperature_iterations>=10
        temperature=cooling_rate*temperature;
        temperature_iterations=0;
    end
    
    if plot_iterations >= 20
       plotroute(city, route, distance,temperature);
       plot_iterations = 0;
    end
%     fprintf("it=%d",item);
end

        
        