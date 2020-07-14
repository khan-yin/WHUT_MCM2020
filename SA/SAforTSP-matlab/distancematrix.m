% function dis = distancematrix(city)
% % DISTANCEMATRIX
% % dis = DISTANCEMATRIX(city) return the distance matrix, dis(i,j) is the 
% % distance between city_i and city_j
% 
% numberofcities = length(city);
% R = 6378.137; % The radius of the Earth
% for i = 1:numberofcities
%     for j = i+1:numberofcities
%         dis(i,j) = distance(city(i).lat, city(i).long, ...
%                             city(j).lat, city(j).long, R);
%         dis(j,i) = dis(i,j);
%     end
% end



function dis = distancematrix(city)
%DISTANCEMATRIX 此处显示有关此函数的摘要
%   此处显示详细说明
cityacount=length(city);
R=6378.137;%地球半径用于求两个城市的球面距离
for i = 1:cityacount
    for j = i+1:cityacount
        dis(i,j)=distance(city(i).lat,city(i).long,...
                            city(j).lat,city(j).long,R);%distance函数原来是设计来计算球面上距离的
        dis(j,i)=dis(i,j);%对称，无向图
    end
end

end




