function draw_figure(bestchrom)
city_coordinate=[1304 2312;3639 1315;4177 2244;3712 1399;3488 1535;3326 1556;...
    3238 1229;4196 1044;4312 790;4386 570;3007 1970;2562 1756;...
    2788 1491;2381 1676;1332 695;3715 1678;3918 2179;4061 2370;...
    3780 2212;3676 2578;4029 2838;4263 2931;3429 1908;3507 2376;...
    3394 2643;3439 3201;2935 3240;3140 3550;2545 2357;2778 2826;...
    2370 2975];
A=(1:31);
B=setdiff( A,bestchrom,'stable');%设置两个数组的差集,可以用help setdiff查看函数的作用
%返回 A 中存在但 B 中不存在的数据，不包含重复项。C 是有序的。
scatter(city_coordinate(bestchrom,1),city_coordinate(bestchrom,2),'rs')
hold on 
scatter(city_coordinate(B,1),city_coordinate(B,2),'bo')
hold on
for i=1:25
    distance=dist(city_coordinate(B(i),:),city_coordinate(bestchrom,:)');
    [~,index]=min(distance);
    plot([city_coordinate(B(i),1),city_coordinate(bestchrom(index),1)],[city_coordinate(B(i),2),city_coordinate(bestchrom(index),2)],'k-')
end

