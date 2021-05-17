[water,txt,raw]=xlsread('./data1.xlsx');
water=[0
1795.800065
-2041.074142
-2542.064004
-899.9517817
-981.5447866
-1091.250378
-705.9759629
-499.2289999
2337.483004
1823.152493
2346.571857
-2653.431559
2461.394534
-1702.547624
-2349.405845
-564.2529931
-498.4969846
-710.6457888
-181.807428
70.41002236
3027.398433
2679.64962
3289.855345
-2574.827684
3030.332184
-1175.883454
-1954.443058
31.60985684
184.0667167
-210.9811077
310.3576965
651.074384
3820.460152
3485.206138
3907.203421
-3545.346979
2356.205898
-2253.187048
-3114.274978
-1507.857115
-1237.542741
-1559.041202
-965.4619049
-718.8143196
2592.292021
2098.24752
2511.242517
-3875.732713
2506.912041
-2327.833075
-3694.177982
-1552.832667
-860.2071822
-1896.311647
-1306.156249
-941.9512431
2431.493047
2183.466231
2627.257845
];
% water=water(1:5,:);
% water=[9.40 8.81 8.65 10.01 11.07 11.54 12.73 12.43 11.64 11.39 11.1 10.85 
% 10.71 10.24 8.48 9.88 10.31 10.53 9.55 6.51 7.75 7.8 5.96 5.21 
% 6.39 6.38 6.51 7.14 7.26 8.49 9.39 9.71 9.65 9.26 8.84 8.29 
% 7.21 6.93 7.21 7.82 8.57 9.59 8.77 8.61 8.94 8.4 8.35 7.95 
% 7.66 7.68 7.85 8.53 9.38 10.09 10.59 10.83 10.49 9.21 8.66 8.39 
% 8.27 8.14 8.71 10.43 11.47 11.73 11.61 11.93 11.55 11.35 11.11 10.49 
% 10.16 9.96 10.47 11.70 10.1 10.37 12.47 11.91 10.83 10.64 10.29 10.34];
water=water';
x=water(:)';
%water(:)为将water中的数据转化为一列数据
r11=autocorr(x);
%计算自相关系数
r12=parcorr(x);
%计算偏相关函数
figure
subplot(211),autocorr(r11);
subplot(212),parcorr(r12);
s=12;
%地下水按照12个月的季节性变化
n=12;
%预报数据的个数
m1=length(x);
%原始数据的个数
for i=s+1:m1
    y(i-s)=x(i)-x(i-s);
    %周期差分：相邻两个年份同一个月分的地下水位的差
end
m2=length(y);
%周期差分后数据的个数
w=diff(y);
%消除趋势性的差分运算
r21=autocorr(w);
%计算自相关系数
r22=parcorr(w);
%计算偏相关函数
adf=adftest(w);
%若adf==1，则表明是平稳时间序列。
figure
subplot(211),autocorr(r21);
subplot(212),parcorr(r22);
m3=length(w);
%计算最终差分后数据的个数
k=0;
for i = 0:3
    for j = 0:3 %0:L,L的值不确定 
        if i == 0 & j == 0
            continue
        elseif i == 0
            ToEstMd = arima('MALags',1:j,'Constant',0); %指定模型的结构
        elseif j == 0
            ToEstMd = arima('ARLags',1:i,'Constant',0); %指定模型的结构
        else
            ToEstMd = arima('ARLags',1:i,'MALags',1:j,'Constant',0); %指定模型的结构
        end
        k = k + 1;
        R(k) = i;
        M(k) = j;
        [EstMd,EstParamCov,LogL,info] = estimate(ToEstMd,w');
        %模型拟合,估计模型参数
        numParams = sum(any(EstParamCov));
        %计算拟合参数的个数
        [aic(k),bic(k)] = aicbic(LogL,numParams,m2);
    end
end
fprintf('R,M,AIC,BIC的对应值如下\n%f');%显示计算结果
check  = [R',M',aic',bic']

%模型验证：
 res=infer(EstMd,w');
%求条件方差，条件方差增加，波动性增加
figure
subplot(2,1,1)
plot(res./sqrt(EstMd.Variance));
%画出标准化残差
title('Standardized Residuals');
subplot(2,1,2),qqplot(res);
%QQ图中残差基本完全落在45°线上即为符合正态性假设。否则模型可能出现错误.
% subplot(2,2,3),autocorr(res);
% subplot(2,2,4),parcorr(res);


%定阶：
%由差分后的由差分后的自相关图与偏自相关数据图可知：
%自相关系数在滞后1阶后就快速的减为0，偏自相关系数同自相关系数
%所以，p=1,q=1

%模型预测：
p=input('输入阶数P=');
q=input('输入阶数q=');
ToEstMd=arima('ARLags',1:p,'MALags',1:q,'Constant',0); 
%指定模型的结构
 [EstMd,EstParamCov,LogL,info] = estimate(ToEstMd,w');
 %模型拟合,估计模型参数
 dy_forest=forecast(EstMd,n,'Y0',w');
 %预测确定的模型输出,注意已知数据应为列向量，所以用w'.
 yhat=y(m2)+cumsum(dy_forest);
 %求一阶差分的还原值
 yhat=yhat';
 for j=1:n
     x(m1+j)=yhat(j)+x(m1+j-s);
     %求x的预测值
 end
 what=x(m1+1:end);
 %截取n个预报值
 %画图：
 figure
 h4 = plot(x,'b');
 %h5 = plot(x,'b');
%  h4 = plot(x,'b');
 hold on
h5 = plot(length(x)-11:length(x),what,'r','LineWidth',2);
xlabel('月份序号');  ylabel('预测与真实值的季节波动');
 hold off