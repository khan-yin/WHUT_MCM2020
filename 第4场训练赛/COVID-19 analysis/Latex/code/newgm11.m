function [result] = new_gm11(x0, predict_num)
% 函数作用：使用新信息的GM(1,1)模型对数据进行预测
% 输入变量
%     x0：要预测的原始数据
%     predict_num： 向后预测的期数
% 输出变量
%     result：预测值
    result = zeros(predict_num,1);  % 初始化用来保存预测值的向量
    for i = 1 : predict_num  
        result(i) = gm11(x0, 1);  % 将预测一期的结果保存到result中
        x0 = [x0; result(i)];  % 更新x0向量，此时x0多了新的预测信息
    end
end
