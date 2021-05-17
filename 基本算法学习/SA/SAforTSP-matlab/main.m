clc
clear;
%% 
load('china.mat');
plotcities(province, border, city);
cityaccount=length(city);%��������
dis=distancematrix(city)%�������

route =randperm(cityaccount);%·����ʽ
temperature=1000;%��ʼ���¶�
cooling_rate=0.95;%�¶��½�����
item=1;%�������ƽ��µ�ѭ����¼����
distance=totaldistance(route,dis);%����·���ܳ���
temperature_iterations = 1;
% This is a flag used to plot the current route after 200 iterations
plot_iterations = 1;

plotroute(city, route, distance,temperature);

while temperature>1.0 %ѭ������
    temp_route=change(route,'reverse');%�����Ŷ����������б仯
%     fprintf("%d\n",temp_route(1));
    temp_distance=totaldistance(temp_route,dis);%�����仯��ĳ���
    dist=temp_distance-distance;%����·���Ĳ��
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

        
        