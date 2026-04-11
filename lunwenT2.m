%% LightGBM二分类与SHAP值解释（硬编码英文列名）
clear; clc; close all;
warning('off', 'all');
tic;

%% 1. 初始化和加载库
addpath('Lightgbm_toolbox');
if ~libisloaded('lib_lightgbm')
    try
        loadlibrary('lib_lightgbm.dll', 'c_api.h');
        fprintf('LightGBM库加载成功\n');
    catch ME
        error('无法加载LightGBM库: %s', ME.message);
    end
end

%% 2. 数据加载和预处理
try
    % 中文列名映射（与Excel文件中的列名对应）
    chinese_feature_names = {
        '年级', '性别', '父母是否在外', '家庭氛围', '是否是独生子女', '民族',...
        '年龄', '父亲在外地', '母亲在外地', '失眠',...
        '高敏感AES', '高敏感EOE', '高敏感LST', '网络成瘾',...
        '抑郁', '焦虑'
    };
    
    % 英文列名（用于显示和处理）
    feature_names = {
        'Grade', 'Gender', 'BothParentsAtHome', 'FamilyEnvironment', 'OnlyChild', 'Ethnicity',...
        'Age', 'FatherOutsideWork', 'MotherOutsideWork', 'Insomnia',...
        'HighSensitivityAES', 'HighSensitivityEOE', 'HighSensitivityLST', 'Internet addiction',...
        'Depression', 'Anxiety'
    };
    
    % 加载数据
    data = readtable('T2_0113_1.xlsx');
    res = table2array(data);
    
    % 特征数量检查
    num_features = length(feature_names);
    data_cols = size(res, 2);
    if data_cols ~= num_features + 1
        error(['数据列数与特征数量不匹配！数据有' num2str(data_cols) '列，应为' num2str(num_features+1) '列']);
    end
    
    % 特征类型定义
    categorical_cols = [1, 2, 3, 4, 5, 6];  % 分类特征索引
    numeric_cols = setdiff(1:num_features, categorical_cols);
    
    % 分类特征编码
    for col = categorical_cols
        if col > width(data)
            warning('特征索引%d超出数据范围，跳过', col);
            continue;
        end
        data_col = data.(col);
        if ~iscategorical(data_col)
            data_col = categorical(data_col);
            data.(col) = data_col;
            fprintf('特征%d（%s）转换为分类变量，共%d个类别\n', ...
                col, feature_names{col}, length(unique(data_col)));
        end
        res(:, col) = double(data_col);
    end
    
    % 划分特征和标签
    X = res(:, 1:end-1);
    y = res(:, end);
    valid_rows = min(size(X,1), length(y));
    X = X(1:valid_rows, :);
    y = y(1:valid_rows);
    
    % 确保标签是二分类的
    unique_y = unique(y);
    if length(unique_y) ~= 2
        error('输出变量非二分类！检测到%d个类别', length(unique_y));
    else
        y(y == unique_y(1)) = 0;
        y(y == unique_y(2)) = 1;
    end
    y = logical(y);
    fprintf('二分类标签处理完成：正例占比=%.2f%%\n', mean(y)*100);
    
    % 如果特征数量与名称数量不匹配，调整名称数组
    if length(feature_names) > size(X, 2)
        feature_names = feature_names(1:size(X, 2));
    elseif length(feature_names) < size(X, 2)
        extra_names = arrayfun(@(x) sprintf('Feature%d', x), (length(feature_names)+1):size(X, 2), 'UniformOutput', false);
        feature_names = [feature_names, extra_names];
    end
    
    fprintf('数据加载完成: %d个样本, %d个特征\n', size(X, 1), size(X, 2));
    
catch ME
    error('数据加载失败: %s', ME.message);
end

%% 3. 数据划分和预处理
try
    % 分层抽样
    cv = cvpartition(y, 'HoldOut', 0.3, 'Stratify', true);
    train_idx = training(cv);
    test_idx = test(cv);
    
    % 训练集预处理
    X_train = X(train_idx, :);
    y_train = y(train_idx);
    if ~isempty(numeric_cols)
        X_train_num = X_train(:, numeric_cols);
        [X_train_num_norm, ps_input] = mapminmax(X_train_num', 0, 1);
        p_train = [X_train(:, categorical_cols), X_train_num_norm'];
    else
        p_train = X_train(:, categorical_cols);
    end
    
    % 测试集预处理
    X_test = X(test_idx, :);
    y_test = y(test_idx);
    if ~isempty(numeric_cols) && exist('ps_input', 'var')
        X_test_num = X_test(:, numeric_cols);
        X_test_num_norm = mapminmax('apply', X_test_num', ps_input)';
        p_test = [X_test(:, categorical_cols), X_test_num_norm];
    else
        p_test = X_test(:, categorical_cols);
    end
    
    fprintf('数据划分完成: 训练集(%d), 测试集(%d)\n', length(train_idx), length(test_idx));
    
catch ME
    error('数据划分失败: %s', ME.message);
end

%% 4. 创建LightGBM数据集
try
    train_data = lgbmDataset(p_train);
    setField(train_data, 'label', y_train);
    
    test_data = lgbmDataset(p_test, train_data);
    setField(test_data, 'label', y_test);
    
    fprintf('LightGBM数据集创建完成\n');
    
catch ME
    error('数据集创建失败: %s', ME.message);
end

%% 5. 设置参数
params = containers.Map;
params('task') = 'train';
params('objective') = 'binary';
params('metric') = 'binary_logloss,auc';
params('verbose') = -1;

% 使用指定的参数
params('num_leaves') = 112;
params('max_depth') = 8;
params('min_data_in_leaf') = 85;
params('learning_rate') = 0.27916;
params('feature_fraction') = 0.78696;
params('bagging_fraction') = 0.67349;
params('bagging_freq') = 0;
params('lambda_l1') =0;
params('lambda_l2') =0;

% 设置分类特征（0-based索引）
valid_cat_cols = categorical_cols(categorical_cols <= size(X, 2));
if ~isempty(valid_cat_cols)
    params('categorical_feature') = strjoin(cellstr(num2str(valid_cat_cols' - 1)), ',');
end
params('early_stopping_rounds') = 8;

%% 6. 训练模型
try
    fprintf('正在训练模型...\n');
    [booster, output] = lightgbm(train_data, params, 1000, {test_data});
    bestIteration = output.best_iteration;
    fprintf('模型训练完成，最佳迭代次数: %d\n', bestIteration);
    
catch ME
    try
        fprintf('尝试备用接口训练...\n');
        [booster, bestIteration] = train(train_data, params, 1000, test_data, 50);
        fprintf('模型训练完成，最佳迭代次数: %d\n', bestIteration);
    catch ME2
        error('模型训练失败: %s', ME2.message);
    end
end

%% 7. 模型预测
try
    train_pred = booster.predictMatrix(p_train, bestIteration);
    test_pred = booster.predictMatrix(p_test, bestIteration);
    
    % 计算AUC
    train_auc = calc_auc(y_train, train_pred);
    test_auc = calc_auc(y_test, test_pred);
    
    fprintf('训练集AUC: %.4f\n', train_auc);
    fprintf('测试集AUC: %.4f\n', test_auc);
    
catch ME
    error('模型预测失败: %s', ME.message);
end

%% 8. 指标计算
[train_acc, train_prec, train_rec, train_f1] = calc_metrics(y_train, train_pred >= 0.5);
[test_acc, test_prec, test_rec, test_f1] = calc_metrics(y_test, test_pred >= 0.5);

% 输出指标
disp('===== 评价指标 =====');
disp(['训练集准确率: ', num2str(train_acc, '%.4f')]);
disp(['训练集精确率: ', num2str(train_prec, '%.4f')]);
disp(['训练集召回率: ', num2str(train_rec, '%.4f')]);
disp(['训练集F1: ', num2str(train_f1, '%.4f')]);
disp(['训练集AUC: ', num2str(train_auc, '%.4f')]);
disp('---------------------');
disp(['测试集准确率: ', num2str(test_acc, '%.4f')]);
disp(['测试集精确率: ', num2str(test_prec, '%.4f')]);
disp(['测试集召回率: ', num2str(test_rec, '%.4f')]);
disp(['测试集F1: ', num2str(test_f1, '%.4f')]);
disp(['测试集AUC: ', num2str(test_auc, '%.4f')]);

%% 9. SHAP值计算和解释（百分比格式）
fprintf('\n计算SHAP值...\n');

% 修改：强制使用1000个样本，但如果测试集样本不足则使用全部
n_samples = min(2000, size(p_test, 1));
if n_samples < 2000
    warning('测试集样本不足1000个，仅使用%d个样本', n_samples);
end

sample_indices = randperm(size(p_test, 1), n_samples);

% 获取背景样本
n_background = min(800, size(p_train, 1));
background_indices = randperm(size(p_train, 1), n_background);
X_background = p_train(background_indices, :);

% 初始化SHAP值矩阵
shap_values = zeros(n_samples, size(p_test, 2));

% 对每个样本计算SHAP值
for i = 1:n_samples
    if mod(i, 50) == 0
        fprintf('计算进度: %d/%d\n', i, n_samples);
    end
    
    sample_idx = sample_indices(i);
    x_instance = p_test(sample_idx, :);
    
    % 计算SHAP值（简化实现）
    shap_values(i, :) = compute_shap_values_approx(booster, x_instance, X_background, bestIteration);
end

% 计算平均SHAP值并转换为百分比
mean_shap = mean(abs(shap_values), 1);
total_shap = sum(mean_shap);
shap_percentages = (mean_shap / total_shap) * 100;

% 按SHAP值大小排序
[sorted_shap, shap_idx] = sort(shap_percentages, 'descend');

% 筛选重要性大于等于1%的特征
significant_features = sorted_shap >= 1.0;
sorted_shap = sorted_shap(significant_features);
shap_idx = shap_idx(significant_features);

% 检查是否有特征重要性大于等于1%
if isempty(sorted_shap)
    % 如果没有特征重要性大于等于1%，则使用默认的前10个特征
    sorted_shap = shap_percentages(1:min(10, length(shap_percentages)));
    shap_idx = 1:length(sorted_shap);
    warning('没有特征的重要性大于等于1%%，将显示前%d个特征', length(sorted_shap));
end

% 获取对应的特征名称
feature_labels = feature_names(shap_idx);

% 绘制竖向条形图（特征在底部，柱子向上）
figure('Name', 'SHAP Feature Importance (Percentage) - Vertical', 'Position', [200, 200, 3000, 800]);  % 增大尺寸至 3000

% 使用 bar 创建竖向柱状图
h = bar(sorted_shap, 'FaceColor', [0.1  0.5  0.9], 'EdgeColor', 'k');

% 获取当前坐标轴
ax = gca;

% 设置 y 轴标签
ylabel('Mean |SHAP Value| (%)', 'FontSize', 12, 'FontWeight', 'bold');

% 添加网格
grid on;
grid minor;

% 设置坐标轴样式
set(ax, 'Box', 'on');
set(ax, 'LineWidth', 1.5);

% 设置坐标轴范围
top_n = length(sorted_shap);
xlim([0, top_n + 0.5]);
ylim([0, max(sorted_shap) * 1.1]);  % 减小上边距

% 添加特征名称标签（贴近x轴）
for i = 1:length(sorted_shap)
    % 在柱子底部显示特征名称（贴近x轴）
    text(i, -max(sorted_shap) * 0.01, feature_labels{i}, ...
        'HorizontalAlignment', 'center', 'FontSize', 5, 'Color', 'k', 'VerticalAlignment', 'top');
end

% 隐藏默认的x轴标签和刻度
set(ax, 'XTick', []);
set(ax, 'XTickLabel', []);

% 移除tight_layout，因为该函数在MATLAB中不可用
% tight_layout;

fprintf('SHAP值计算完成\n');

%% 10. SHAP依赖图（SHAP Dependence Plot） - 每个特征单独输出
% 为所有18个特征创建单独的SHAP依赖图
n_features = length(feature_names);

% 遍历所有特征并创建独立的依赖图
for i = 1:n_features
    feature_idx = shap_idx(i);  % 获取当前特征的索引
    feature_name = feature_names{feature_idx};  % 获取特征名称
    
    % 创建新的图形窗口
    figure('Name', sprintf('SHAP Dependence Plot - %s', feature_name), ...
           'Position', [100, 100, 1200, 800]);  % 设置合适的画幅大小
    
    % 获取特征值和对应的SHAP值
    feature_data = p_test(sample_indices, feature_idx);
    shap_values_feature = shap_values(:, feature_idx);
    
    % 绘制散点图：x轴为特征值，y轴为SHAP值
    scatter(feature_data, shap_values_feature, 50, ...
        shap_values_feature, 'filled');
    
    % 添加颜色条
    colorbar('Location', 'eastoutside');
    colormap(jet);
    caxis([-0.06, 0.06]);  % 设置颜色范围
    
    % 设置标题和轴标签

    xlabel('Feature Value', 'FontSize', 12);
    ylabel('SHAP Value (Impact on Bullying)', 'FontSize', 12);
    
    % 添加网格
    grid on;
    
    % 调整坐标轴范围
    xlim([min(feature_data) - 0.07*(max(feature_data)-min(feature_data)), ...
          max(feature_data) + 0.07*(max(feature_data)-min(feature_data))]);
    ylim([-0.06, 0.06]);
    
    % 设置坐标轴样式
    set(gca, 'Box', 'on');
    set(gca, 'LineWidth', 1.5);
    
    % 保存图像（可选）
    % print(sprintf('shap_dependence_%s.png', replace(feature_name, ' ', '_')), '-r300', '-png');
end

fprintf('已生成%d个特征的SHAP依赖图\n', n_features);
%% 辅助函数
function auc = calc_auc(y_true, y_prob)
    y_true = logical(y_true);
    [~, idx] = sort(y_prob, 'descend');
    sorted_labels = y_true(idx);
    P = sum(sorted_labels);
    N = length(sorted_labels) - P;
    if P == 0 || N == 0
        auc = 0.5;
        return;
    end
    tpr = cumsum(sorted_labels) / P;
    fpr = cumsum(~sorted_labels) / N;
    auc = trapz(fpr, tpr);
end

function [acc, prec, rec, f1] = calc_metrics(y_true, y_pred)
    y_true = logical(y_true);
    y_pred = logical(y_pred);
    TP = sum(y_true & y_pred);
    TN = sum(~y_true & ~y_pred);
    FP = sum(~y_true & y_pred);
    FN = sum(y_true & ~y_pred);
    
    acc = (TP + TN) / max(length(y_true), 1);
    prec = TP / max(TP + FP, 1);
    rec = TP / max(TP + FN, 1);
    f1 = 2 * prec * rec / max(prec + rec, 1);
end

function shap_values = compute_shap_values_approx(model, x_instance, X_background, bestIteration)
    % 简化的SHAP值计算（近似实现）
    n_features = length(x_instance);
    shap_values = zeros(1, n_features);
    
    % 获取原始预测值
    original_pred = model.predictMatrix(x_instance, bestIteration);
    
    % 对每个特征计算边际贡献
    for i = 1:n_features
        % 创建扰动样本
        x_perturbed = x_instance;
        % 使用背景数据的值替换该特征
        if size(X_background, 1) > 0
            rand_idx = randi(size(X_background, 1));
            x_perturbed(i) = X_background(rand_idx, i);
        else
            x_perturbed(i) = mean(x_instance);
        end
        
        % 计算扰动后的预测值
        perturbed_pred = model.predictMatrix(x_perturbed, bestIteration);
        
        % 计算SHAP值（简化）
        shap_values(i) = original_pred - perturbed_pred;
    end
end

fprintf('\n=== 分析完成 ===\n');
toc;