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
%DISTANCEMATRIX �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
cityacount=length(city);
R=6378.137;%����뾶�������������е��������
for i = 1:cityacount
    for j = i+1:cityacount
        dis(i,j)=distance(city(i).lat,city(i).long,...
                            city(j).lat,city(j).long,R);%distance����ԭ������������������Ͼ����
        dis(j,i)=dis(i,j);%�Գƣ�����ͼ
    end
end

end




