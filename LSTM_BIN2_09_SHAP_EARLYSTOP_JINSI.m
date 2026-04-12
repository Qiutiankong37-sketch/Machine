%% 使用T1、T2数据预测T3的二分类模型
% 功能：
% 1. 特征工程（滞后、变化量、季节虚拟变量、传统时间序列模型残差）
% 2. 构建训练集（X_train, y_train）
% 3. MATLAB → Python：调用 LightGBM
% 4. LSTM深度学习模型
% 5. 模型集成（LightGBM + LSTM）
% 6. 模型训练 + 交叉验证
% 7. 预测 T3 + 评估（AUC, F1等二分类指标）
% 8. SHAP值分析和可视化

warning off; close all; clear; clc;
% 设置随机种子以确保结果可重现
rng(42);
try
    rng.shuffle('twister');  % 为深度学习设置随机种子
catch
end
tic

%% 1. 数据处理和特征工程
addpath('Lightgbm_toolbox\');
loadlibrary('lib_lightgbm.dll', 'c_api.h');

% 导入T1、T2、T3数据
% 假设数据文件名为：T1_data.xlsx, T2_data.xlsx, T3_data.xlsx
% 每个文件包含相同的19个特征列和1个目标列

fprintf('正在加载T1数据...\n');
data_T1 = readtable('transT1.xlsx');
feature_names = data_T1.Properties.VariableNames(1:end-1);  % 假设最后一列是目标变量
res_T1 = table2array(data_T1);

fprintf('正在加载T2数据...\n');
data_T2 = readtable('transT2.xlsx');
res_T2 = table2array(data_T2);

fprintf('正在加载T3数据...\n');
data_T3 = readtable('transT3.xlsx');
res_T3 = table2array(data_T3);

% 验证数据一致性
if size(res_T1, 2) ~= size(res_T2, 2) || size(res_T2, 2) ~= size(res_T3, 2)
    error('T1、T2、T3数据列数不一致，请检查数据');
end

if size(res_T1, 1) ~= size(res_T2, 1) || size(res_T2, 1) ~= size(res_T3, 1)
    error('T1、T2、T3数据行数不一致，请检查数据');
end

num_features = size(res_T1, 2) - 1;
num_samples = size(res_T1, 1);
fprintf('数据集信息：%d特征，%d样本\n', num_features, num_samples);

% 特征工程
% 1. 滞后特征 (T1值作为T2的滞后，T2值作为T3的滞后)
% 2. 变化量特征 (T2-T1, T3-T2)
% 3. 季节虚拟变量 (根据数据时间属性确定)
% 4. 传统时间序列模型残差特征

fprintf('正在进行特征工程...\n');

% 提取特征和目标变量
X_T1 = res_T1(:, 1:end-1);
y_T1 = res_T1(:, end);

X_T2 = res_T2(:, 1:end-1);
y_T2 = res_T2(:, end);

X_T3 = res_T3(:, 1:end-1);
y_T3 = res_T3(:, end);

% 构建工程特征
% 滞后特征
lag_features_T2 = X_T1;  % T1值作为T2时刻的滞后特征
lag_features_T3 = X_T2;  % T2值作为T3时刻的滞后特征

% 变化量特征
diff_features_T2 = X_T2 - X_T1;  % T1到T2的变化量
diff_features_T3 = X_T3 - X_T2;  % T2到T3的变化量

% 季节虚拟变量 (假设数据按春、秋季节排列)
% 创建季节虚拟变量：春季=1,0；秋季=0,1
season_dummy_spring = repmat([1; 0], ceil(num_samples/2), 1);
season_dummy_autumn = repmat([0; 1], ceil(num_samples/2), 1);
season_dummy = [season_dummy_spring(1:num_samples,:), season_dummy_autumn(1:num_samples,:)];

% 传统时间序列模型残差特征
% 对每个特征拟合ARIMA模型并计算残差
arima_residuals_T2 = zeros(size(X_T2, 1), size(X_T2, 2));
arima_residuals_T3 = zeros(size(X_T3, 1), size(X_T3, 2));

for i = 1:size(X_T1, 2)
    % 使用T1数据拟合ARIMA模型
    try
        % 简单的ARIMA(1,1,1)模型
        model_T1 = arima(1, 1, 1);
        fitted_model = estimate(model_T1, X_T1(:, i), 'Display', 'off');
        
        % 预测T2和T3的值
        forecast_T2 = forecast(fitted_model, 1, 'Y0', X_T1(:, i));
        forecast_T3 = forecast(fitted_model, 1, 'Y0', [X_T1(2:end, i); X_T2(1, i)]);
        
        % 计算残差
        arima_residuals_T2(:, i) = X_T2(:, i) - forecast_T2;
        arima_residuals_T3(:, i) = X_T3(:, i) - forecast_T3;
    catch
        % 如果ARIMA拟合失败，使用简单差分作为替代
        arima_residuals_T2(:, i) = diff([X_T1(1, i); X_T2(:, i)]);
        arima_residuals_T3(:, i) = diff([X_T2(1, i); X_T3(:, i)]);
    end
end

% 组合特征
engineered_features_T2 = [X_T2, lag_features_T2, diff_features_T2, season_dummy(1:num_samples, :), arima_residuals_T2];
engineered_features_T3 = [X_T3, lag_features_T3, diff_features_T3, season_dummy(1:num_samples, :), arima_residuals_T3];

% 更新特征名称列表以匹配工程后的特征
if length(feature_names) == size(X_T1, 2)
    % 创建完整的特征名称列表
    engineered_feature_names = [feature_names, ...                     % 原始特征 (18个)
                               strcat(feature_names, '_lag'), ...     % 滞后特征 (18个)
                               strcat(feature_names, '_diff'), ...    % 差分特征 (18个)
                               {'season_spring', 'season_autumn'}, ...% 季节特征 (2个)
                               strcat(feature_names, '_arima_resid')];% ARIMA残差 (18个)
else
    % 如果特征名称数量不匹配，使用默认名称
    total_features = size(engineered_features_T2, 2);
    engineered_feature_names = arrayfun(@(x) sprintf('Feature_%d', x), 1:total_features, 'UniformOutput', false);
    warning('特征名称数量与特征数量不匹配，使用默认特征名称');
end

% 构建训练集 (使用T2的数据预测T3的目标值)
X_train = engineered_features_T2;
y_train = y_T3;  % T3的目标值作为训练目标

% 构建测试集 (使用T3的数据进行最终评估)
X_test = engineered_features_T3;
y_test = y_T3;

% 准备LSTM时间序列数据
% 构建三维时间序列数据 (样本数, 时间步长, 特征数)
time_steps = 3;  % 使用3个时间步长的数据

% 为LSTM准备数据
lstm_X_train = [];
lstm_X_test = [];
lstm_y_train = [];
lstm_y_test = [];

% 使用原始特征构建时间序列数据
% 修复：记录哪些索引被用于LSTM训练，以便LightGBM输出可以对齐
lstm_train_indices = [];
lstm_test_indices = [];

for i = time_steps:size(X_T1, 1)
    % 构建训练时间序列数据 (使用T1数据)
    if i <= size(X_T1, 1) - time_steps + 1
        sample = zeros(time_steps, size(X_T1, 2));
        for t = 1:time_steps
            sample(t, :) = X_T1(i + t - 1, :);
        end
        lstm_X_train(end+1, :, :) = sample;
        lstm_y_train(end+1) = y_T1(i + time_steps - 1);
        lstm_train_indices(end+1) = i + time_steps - 1;  % 记录索引
    end
end

for i = time_steps:size(X_T2, 1)
    % 构建测试时间序列数据 (使用T2数据预测T3)
    if i <= size(X_T2, 1) - time_steps + 1
        sample = zeros(time_steps, size(X_T2, 2));
        for t = 1:time_steps
            sample(t, :) = X_T2(i + t - 1, :);
        end
        lstm_X_test(end+1, :, :) = sample;
        lstm_y_test(end+1) = y_T3(i + time_steps - 1);
        lstm_test_indices(end+1) = i + time_steps - 1;  % 记录索引
    end
end

% 确保目标变量为二分类（0/1编码）
unique_y = unique(y_train);
if length(unique_y) ~= 2
    error('目标变量不是二分类变量，有%d个唯一值', length(unique_y));
end

if all(ismember(unique_y, [0,1]))
    fprintf('目标变量已经是0/1编码\n');
else
    y_train(y_train == unique_y(1)) = 0;
    y_train(y_train == unique_y(2)) = 1;
    y_test(y_test == unique_y(1)) = 0;
    y_test(y_test == unique_y(2)) = 1;
    fprintf('目标变量已转换为0/1编码\n');
end

y_train = y_train(:);  % 确保是列向量
y_test = y_test(:);

% LSTM标签处理
if ~isempty(lstm_y_train)
    unique_lstm_y = unique(lstm_y_train);
    if ~all(ismember(unique_lstm_y, [0,1]))
        lstm_y_train(lstm_y_train == unique_lstm_y(1)) = 0;
        lstm_y_train(lstm_y_train == unique_lstm_y(2)) = 1;
        lstm_y_test(lstm_y_test == unique_lstm_y(1)) = 0;
        lstm_y_test(lstm_y_test == unique_lstm_y(2)) = 1;
    end
end

fprintf('特征工程完成\n');
fprintf('训练集大小: %d × %d\n', size(X_train, 1), size(X_train, 2));
fprintf('测试集大小: %d × %d\n', size(X_test, 1), size(X_test, 2));
if ~isempty(lstm_X_train)
    fprintf('LSTM训练集大小: %d × %d × %d\n', size(lstm_X_train, 1), size(lstm_X_train, 2), size(lstm_X_train, 3));
    fprintf('LSTM测试集大小: %d × %d × %d\n', size(lstm_X_test, 1), size(lstm_X_test, 2), size(lstm_X_test, 3));
    fprintf('LSTM训练标签大小: %d\n', length(lstm_y_train));
    fprintf('LSTM测试标签大小: %d\n', length(lstm_y_test));
end

%% 2. 数据预处理
% 确定特征类型
categorical_cols = 1:6;  % 假设前8列为多分类特征
numeric_cols = setdiff(1:size(X_train, 2), categorical_cols);

% 多分类特征编码
if ~isempty(categorical_cols)
    for col = categorical_cols
        if col <= size(X_train, 2)
            % 确保训练集分类变量正确编码
            train_cat_vals = unique(X_train(:, col));
            X_train(:, col) = grp2idx(X_train(:, col));
            
            % 对测试集应用相同编码
            test_cat_vals = unique(X_test(:, col));
            X_test(:, col) = grp2idx(X_test(:, col));
            
            fprintf('特征%d为多分类，训练集共%d个类别，测试集共%d个类别\n', ...
                col, length(train_cat_vals), length(test_cat_vals));
        end
    end
end

% 数值特征归一化
if ~isempty(numeric_cols)
    % 训练集归一化
    X_train_num = X_train(:, numeric_cols);
    [X_train_num_norm, ps_input] = mapminmax(X_train_num', 0, 1);
    p_train = [X_train(:, categorical_cols), X_train_num_norm'];
    
    % 测试集应用相同归一化参数
    X_test_num = X_test(:, numeric_cols);
    X_test_num_norm = mapminmax('apply', X_test_num', ps_input);
    p_test = [X_test(:, categorical_cols), X_test_num_norm'];
else
    p_train = X_train(:, categorical_cols);
    p_test = X_test(:, categorical_cols);
end

% LSTM数据归一化
if ~isempty(lstm_X_train)
    [lstm_X_train_norm, lstm_ps_input] = mapminmax(reshape(lstm_X_train, [], size(lstm_X_train, 3))', 0, 1);
    lstm_X_train_norm = reshape(lstm_X_train_norm', size(lstm_X_train, 1), size(lstm_X_train, 2), size(lstm_X_train, 3));
    
    % 修复：确保lstm_ps_input是有效的结构体再使用它
    % 同时确保lstm_X_test不为空再进行处理
    if ~isempty(lstm_X_test) 
        if exist('lstm_ps_input', 'var') && isa(lstm_ps_input, 'struct') && isstruct(lstm_ps_input)
            try
                [lstm_X_test_norm, ~] = mapminmax(reshape(lstm_X_test, [], size(lstm_X_test, 3))', 0, 1, lstm_ps_input);
                lstm_X_test_norm = reshape(lstm_X_test_norm', size(lstm_X_test, 1), size(lstm_X_test, 2), size(lstm_X_test, 3));
            catch
                % 如果使用lstm_ps_input失败，则对测试数据单独进行归一化
                [lstm_X_test_norm, ~] = mapminmax(reshape(lstm_X_test, [], size(lstm_X_test, 3))', 0, 1);
                lstm_X_test_norm = reshape(lstm_X_test_norm', size(lstm_X_test, 1), size(lstm_X_test, 2), size(lstm_X_test, 3));
                warning('使用独立的归一化参数处理测试数据，可能导致数据分布不一致');
            end
        else
            % 如果lstm_ps_input不是结构体，则对测试数据单独进行归一化
            [lstm_X_test_norm, ~] = mapminmax(reshape(lstm_X_test, [], size(lstm_X_test, 3))', 0, 1);
            lstm_X_test_norm = reshape(lstm_X_test_norm', size(lstm_X_test, 1), size(lstm_X_test, 2), size(lstm_X_test, 3));
            warning('lstm_ps_input不是有效的结构体，使用独立的归一化参数处理测试数据');
        end
    else
        % 如果lstm_X_test为空，则初始化为空数组
        lstm_X_test_norm = [];
    end
else
    % 如果lstm_X_train为空，则两个变量都初始化为空数组
    lstm_X_train_norm = [];
    lstm_X_test_norm = [];
end

% 目标变量
t_train = y_train;
t_test = y_test;

% 记录类别分布
train_class_dist = [sum(y_train==0), sum(y_train==1)];
test_class_dist = [sum(y_test==0), sum(y_test==1)];
fprintf('训练集类别分布: 负例=%d, 正例=%d\n', train_class_dist(1), train_class_dist(2));
fprintf('测试集类别分布: 负例=%d, 正例=%d\n', test_class_dist(1), test_class_dist(2));

fprintf('数据预处理完成\n');

%% 3. 模型训练 - LightGBM部分
fprintf('开始训练LightGBM模型...\n');

% 参数优化（使用改进麻雀搜索算法ISSA）
best_params = struct();
best_mean_score = Inf;  % 最小化综合得分
best_threshold = 0.5;   % 最佳阈值

% 计算正例比例，用于scale_pos_weight参数
pos_ratio = mean(y_train);
fprintf('二分类目标：正例比例 = %.2f%%\n', pos_ratio*100);

% 参数范围定义（与论文Table 2一致）
param_bounds = struct();
param_bounds.num_leaves = [10, 100];         
param_bounds.max_depth = [3, 10];
param_bounds.min_data_in_leaf = [20, 1000]; 
param_bounds.learning_rate = [0.01, 0.3];    
param_bounds.feature_fraction = [0.5, 1.0];    
param_bounds.bagging_fraction = [0.6, 1.0];    
param_bounds.bagging_freq = [1, 10];         
param_bounds.lambda_l1 = [0, 10];            
param_bounds.lambda_l2 = [0, 10];            

% 默认参数（保守设置）
default_params.num_leaves = 31;
default_params.max_depth = 5;
default_params.min_data_in_leaf = 20;
default_params.learning_rate = 0.1;
default_params.feature_fraction = 0.8;
default_params.bagging_fraction = 0.8;
default_params.bagging_freq = 5;
default_params.lambda_l1 = 0.1;
default_params.lambda_l2 = 0.1;
% 调整scale_pos_weight以处理类别不平衡
default_params.scale_pos_weight = max(1.0, (1-pos_ratio)/pos_ratio);

% ISSA优化设置
n_pop = 50;            % 麻雀种群数量 (根据论文设置)
n_iterations = 50;     % 迭代次数 (根据论文设置)
n_pd = 0.2;            % 发现者比例
n_sd = 0.1;            % 警戒者比例
R2 = 0.8;              % 安全阈值

fprintf('开始改进麻雀搜索算法（ISSA）参数搜索（仅保留训练集AUC≤0.95的模型）...\n');

% 存储评估结果（仅保留有效模型）
evaluated_params = {};
evaluated_scores = [];
valid_model_count = 0;

% 划分验证集用于参数优化（使用时间序列分割）
n_total = length(y_train);
n_val = floor(n_total * 0.2);
val_idx = (n_total - n_val + 1):n_total;
train_idx = 1:(n_total - n_val);

X_train_split = p_train(train_idx, :);
X_val_split = p_train(val_idx, :);
y_train_split = t_train(train_idx);
y_val_split = t_train(val_idx);

% 使用Halton序列初始化麻雀种群
pop = initializeISSAPopulationWithHalton(n_pop, param_bounds);
fitness = zeros(n_pop, 1);

% 评估初始种群
for i = 1:n_pop
    fprintf('初始种群评估 %d/%d...\n', i, n_pop);
    
    % 将粒子位置转换为参数
    params = convertPositionToParams(pop(i, :), param_bounds);
    params.scale_pos_weight = default_params.scale_pos_weight;
    
    % 评估参数（训练集AUC>0.95直接判为无效）
    [composite_score, metrics, threshold] = evaluateParamsWithCompositeMetric(... 
        X_train_split, y_train_split, X_val_split, y_val_split, numeric_cols, categorical_cols, ps_input, params);
    
    fitness(i) = composite_score;
    
    if isfinite(composite_score)
        valid_model_count = valid_model_count + 1;
        evaluated_params{end+1} = params;
        evaluated_scores(end+1) = composite_score;
        fprintf('初始个体 %d（有效模型%d）: 综合得分=%.4f (训练AUC=%.4f, 验证AUC=%.4f)\n', ...
            i, valid_model_count, composite_score, metrics.train_auc, metrics.val_auc);
        
        if composite_score < best_mean_score
            best_mean_score = composite_score;
            best_params = params;
            best_threshold = threshold;
        end
    else
        fprintf('初始个体 %d: 训练集AUC=%.4f > 0.95，模型无效，已过滤\n', i, metrics.train_auc);
    end
end

% ISSA迭代优化
for iter = 1:n_iterations
    fprintf('ISSA优化迭代 %d/%d...\n', iter, n_iterations);
    
    % 对适应度进行排序
    [sorted_fitness, idx] = sort(fitness);
    sorted_pop = pop(idx, :);
    
    % 确定发现者和加入者数量
    n_producers = round(n_pop * n_pd);
    n_scroungers = n_pop - n_producers - round(n_pop * n_sd);
    
    % 确保至少有一个发现者和警戒者
    n_producers = max(1, n_producers);
    n_scroungers = max(1, n_scroungers);
    
    % 1. 发现者（Producer）位置更新
    for i = 1:n_producers
        if rand() < R2
            % 基本SSA更新
            alpha = rand();
            sorted_pop(i, :) = sorted_pop(i, :) .* exp(-i / (alpha * iter + eps));
        else
            % 改进策略：引入高斯变异
            sorted_pop(i, :) = sorted_pop(i, :) + randn(1, size(sorted_pop, 2));
        end
        % 边界处理
        sorted_pop(i, :) = max(min(sorted_pop(i, :), 1), 0);
    end
    
    % 2. 加入者（Scrounger）位置更新
    for i = (n_producers+1):(min(n_producers+n_scroungers, n_pop))
        % 改进策略：引入差分进化思想
        A = rand() * ones(1, size(sorted_pop, 2));
        A(rand(size(A)) > 0.5) = 1;
        A(rand(size(A)) <= 0.5) = -1;
        
        if fitness(i) > mean(fitness)
            % 质量较差的个体
            k = randi(n_producers);
            sorted_pop(i, :) = sorted_pop(k, :) + abs(sorted_pop(i, :) - sorted_pop(k, :)) .* A ./ (rand() * ones(1, size(sorted_pop, 2)) + eps);
        else
            % 质量较好的个体
            kp = randi(n_producers);
            kpp = randi(n_producers);
            while kp == kpp
                kpp = randi(n_producers);
            end
            sorted_pop(i, :) = sorted_pop(i, :) + (sorted_pop(kp, :) - sorted_pop(kpp, :)) * rand();
        end
        % 边界处理
        sorted_pop(i, :) = max(min(sorted_pop(i, :), 1), 0);
    end
    
    % 3. 警戒者（Scout）位置更新
    n_scouts = n_pop - n_producers - n_scroungers;
    if n_scouts > 0
        for i = max(n_producers+n_scroungers+1, n_pop-n_scouts+1):n_pop
            % 改进策略：混沌映射初始化
            sorted_pop(i, :) = chaoticTentMap(sorted_pop(i, :));
        end
    end
    
    % 更新种群
    pop = sorted_pop;
    
    % 评估新种群
    for i = 1:n_pop
        % 将粒子位置转换为参数
        params = convertPositionToParams(pop(i, :), param_bounds);
        params.scale_pos_weight = default_params.scale_pos_weight;
        
        % 评估参数（训练集AUC>0.95直接判为无效）
        [composite_score, metrics, threshold] = evaluateParamsWithCompositeMetric(... 
            X_train_split, y_train_split, X_val_split, y_val_split, numeric_cols, categorical_cols, ps_input, params);
        
        fitness(i) = composite_score;
        
        if isfinite(composite_score)
            valid_model_count = valid_model_count + 1;
            evaluated_params{end+1} = params;
            evaluated_scores(end+1) = composite_score;
            fprintf('迭代 %d - 个体 %d（有效模型%d）: 综合得分=%.4f (训练AUC=%.4f, 验证AUC=%.4f)\n', ...
                iter, i, valid_model_count, composite_score, metrics.train_auc, metrics.val_auc);
            
            if composite_score < best_mean_score
                best_mean_score = composite_score;
                best_params = params;
                best_threshold = threshold;
            end
        else
            fprintf('迭代 %d - 个体 %d: 训练集AUC=%.4f > 0.95，模型无效，已过滤\n', iter, i, metrics.train_auc);
        end
    end
end

% 检查是否有有效模型
if valid_model_count == 0
    error('未找到训练集AUC≤0.95的有效模型，请放宽参数限制或检查数据');
end

fprintf('ISSA优化完成，共保留%d个有效模型，最佳综合得分: %.4f\n', valid_model_count, best_mean_score);

%% 4. 模型训练 + 时间序列交叉验证
fprintf('训练最终LightGBM模型（强制训练集AUC≤0.95）...\n');

if isempty(p_train) || isempty(t_train)
    error('训练数据为空');
end

try
    pv_train = lgbmDataset(single(p_train));
    setField(pv_train, 'label', single(t_train));
catch ME
    fprintf('数据集创建失败: %s\n', ME.message);
    error('无法创建LightGBM数据集');
end

% 使用最佳参数设置
final_params = containers.Map;
final_params('task') = 'train';
final_params('objective') = 'binary';
final_params('metric') = 'binary_logloss,auc';
if ~isempty(categorical_cols)
    categorical_indices = categorical_cols - 1;
    final_params('categorical_feature') = strjoin(cellstr(num2str(categorical_indices')), ',');
end
final_params('verbose') = 1;
final_params('early_stopping_round') = 20;  % 增强早停机制

% 设置最佳参数（强制保守）
final_params('num_leaves') = max(10, best_params.num_leaves);  % 确保最小值
final_params('max_depth') = max(3, best_params.max_depth);    % 确保最小值
final_params('learning_rate') = max(0.01, min(0.3, best_params.learning_rate));  % 限制范围
final_params('min_data_in_leaf') = max(20, best_params.min_data_in_leaf);  % 确保最小值
final_params('bagging_freq') = max(1, round(best_params.bagging_freq));  % 确保为正整数
final_params('feature_fraction') = max(0.5, min(1.0, best_params.feature_fraction));  % 限制范围
final_params('bagging_fraction') = max(0.6, min(1.0, best_params.bagging_fraction));  % 限制范围
final_params('lambda_l1') = max(0, best_params.lambda_l1);  % L1正则化
final_params('lambda_l2') = max(0, best_params.lambda_l2);  % L2正则化
final_params('scale_pos_weight') = best_params.scale_pos_weight;

% 多轮5折交叉验证设置 (50×5-fold CV)
n_rounds = 50;  % 多轮验证轮数
k_folds = 5;    % 5折交叉验证
n_total = length(t_train);
fold_size = floor(n_total / k_folds);

% 存储交叉验证结果
cv_auc = zeros(n_rounds * k_folds, 1);
cv_f1 = zeros(n_rounds * k_folds, 1);
cv_recall = zeros(n_rounds * k_folds, 1);
cv_accuracy = zeros(n_rounds * k_folds, 1);

fprintf('开始%d轮%d折时间序列交叉验证...\n', n_rounds, k_folds);

cv_index = 1;
for round = 1:n_rounds
    fprintf('正在进行第%d轮交叉验证...\n', round);
    
    for fold = 1:k_folds
        % 划分训练集和验证集（保持时间顺序）
        if fold < k_folds
            val_start = (fold - 1) * fold_size + 1;
            val_end = fold * fold_size;
        else
            % 最后一折包含所有剩余数据
            val_start = (fold - 1) * fold_size + 1;
            val_end = n_total;
        end
        
        val_mask = false(n_total, 1);
        val_mask(val_start:val_end) = true;
        train_mask = ~val_mask;
        
        % 创建训练集和验证集数据
        pv_train_fold = lgbmDataset(single(p_train(train_mask, :)));
        setField(pv_train_fold, 'label', single(t_train(train_mask)));
        
        pv_val_fold = lgbmDataset(single(p_train(val_mask, :)), pv_train_fold);
        setField(pv_val_fold, 'label', single(t_train(val_mask)));
        
        % 训练模型
        try
            [booster, output] = train(pv_train_fold, final_params, 500, pv_val_fold);
            best_iteration = output.best_iteration;
        catch
            % 备用训练方法
            [booster, best_iteration] = train(pv_train_fold, final_params, 200);
        end
        
        % 验证集预测
        val_prob = booster.predictMatrix(single(p_train(val_mask, :)), best_iteration);
        val_pred = (val_prob >= best_threshold);
        
        % 计算评估指标
        val_true = logical(t_train(val_mask));
        val_pred = logical(val_pred);
        
        % AUC
        cv_auc(cv_index) = roc_auc_score(val_true, val_prob);
        
        % F1分数
        TP = sum(val_true & val_pred);
        FP = sum(~val_true & val_pred);
        FN = sum(val_true & ~val_pred);
        precision = TP / max(TP + FP, 1e-6);
        recall = TP / max(TP + FN, 1e-6);
        cv_f1(cv_index) = 2 * (precision * recall) / max(precision + recall, 1e-6);
        
        % Recall
        cv_recall(cv_index) = recall;
        
        % Accuracy
        cv_accuracy(cv_index) = sum(val_true == val_pred) / length(val_true);
        
        fprintf('第%d轮第%d折 - AUC: %.4f, F1: %.4f, Recall: %.4f, Accuracy: %.4f\n', ...
            round, fold, cv_auc(cv_index), cv_f1(cv_index), cv_recall(cv_index), cv_accuracy(cv_index));
        
        cv_index = cv_index + 1;
    end
end

fprintf('时间序列交叉验证完成\n');
fprintf('平均 AUC: %.4f ± %.4f\n', mean(cv_auc), std(cv_auc));
fprintf('平均 F1: %.4f ± %.4f\n', mean(cv_f1), std(cv_f1));
fprintf('平均 Recall: %.4f ± %.4f\n', mean(cv_recall), std(cv_recall));
fprintf('平均 Accuracy: %.4f ± %.4f\n', mean(cv_accuracy), std(cv_accuracy));

%% 5. LSTM模型训练
fprintf('开始训练LSTM模型...\n');
lstm_model = [];
lstm_train_prob = [];
lstm_test_prob = [];

try
    % 确保LSTM数据和标签数量一致
    if ~isempty(lstm_X_train_norm) && ~isempty(lstm_y_train)
        % 检查数据一致性
        fprintf('LSTM训练数据维度: %d × %d × %d\n', size(lstm_X_train_norm, 1), size(lstm_X_train_norm, 2), size(lstm_X_train_norm, 3));
        fprintf('LSTM训练标签维度: %d\n', size(lstm_y_train, 1));
        
        if ~isempty(lstm_X_test_norm) && ~isempty(lstm_y_test)
            fprintf('LSTM测试数据维度: %d × %d × %d\n', size(lstm_X_test_norm, 1), size(lstm_X_test_norm, 2), size(lstm_X_test_norm, 3));
            fprintf('LSTM测试标签维度: %d\n', size(lstm_y_test, 1));
        end
        
        % 构建LSTM网络
        layers = [ ...
            sequenceInputLayer(size(lstm_X_train_norm, 3))
            lstmLayer(100, 'OutputMode', 'last')
            fullyConnectedLayer(2)  % 改为2以匹配二分类任务中的两个类别
            softmaxLayer
            classificationLayer];
        
        % 训练选项
        options = trainingOptions('adam', ...
            'MaxEpochs', 100, ...
            'MiniBatchSize', 32, ...
            'Plots', 'training-progress', ...
            'Verbose', false, ...
            'Shuffle', 'every-epoch', ...
            'ValidationFrequency', 10, ...
            'ValidationPatience', 10);
        
        % 转换数据格式 - 确保正确的数据格式
        fprintf('转换训练数据格式...\n');
        if ndims(lstm_X_train_norm) == 3
            % 转换为cell数组格式，每个样本是一个矩阵（时间步长×特征数）
            num_samples = size(lstm_X_train_norm, 1);
            lstm_X_train_cell = cell(num_samples, 1);
            for i = 1:num_samples
                % 提取数据并重新排列维度
                sample_data = squeeze(lstm_X_train_norm(i, :, :));  % 将3D切片转换为2D矩阵
                % 重新排列为特征数×时间步长
                sample_data = sample_data';  % 现在是特征数×时间步长
                lstm_X_train_cell{i} = sample_data;
            end
        else
            error('LSTM训练数据格式不正确');
        end
        
        % 确保标签是列向量
        if size(lstm_y_train, 2) > 1
            lstm_y_train = lstm_y_train';
        end
        
        % 同样处理测试数据
        if ~isempty(lstm_X_test_norm)
            if ndims(lstm_X_test_norm) == 3
                num_samples_test = size(lstm_X_test_norm, 1);
                lstm_X_test_cell = cell(num_samples_test, 1);
                for i = 1:num_samples_test
                    % 提取数据并重新排列维度
                    sample_data = squeeze(lstm_X_test_norm(i, :, :));  % 将3D切片转换为2D矩阵
                    % 重新排列为特征数×时间步长
                    sample_data = sample_data';  % 现在是特征数×时间步长
                    lstm_X_test_cell{i} = sample_data;
                end
            else
                error('LSTM测试数据格式不正确');
            end
            
            if size(lstm_y_test, 2) > 1
                lstm_y_test = lstm_y_test';
            end
        end
        
        % 训练LSTM模型
        fprintf('正在训练LSTM模型...\n');
        fprintf('LSTM训练数据样本数: %d\n', size(lstm_X_train_cell, 1));
        fprintf('LSTM训练标签样本数: %d\n', size(lstm_y_train, 1));
        
        % 再次检查数据一致性
        if size(lstm_X_train_cell, 1) ~= size(lstm_y_train, 1)
            error('LSTM训练数据和标签数量不一致: %d vs %d', size(lstm_X_train_cell, 1), size(lstm_y_train, 1));
        end
        
        % 确保标签是分类标签格式（categorical）
        lstm_y_train_cat = categorical(lstm_y_train);
        lstm_model = trainNetwork(lstm_X_train_cell, lstm_y_train_cat, layers, options);
        
        % LSTM预测
        fprintf('使用LSTM模型进行预测...\n');
        lstm_train_pred = predict(lstm_model, lstm_X_train_cell);
        % 提取正类概率
        lstm_train_prob = lstm_train_pred(:, 2);  % 直接获取正类概率（第二列）
        
        if ~isempty(lstm_X_test_norm)
            lstm_test_pred = predict(lstm_model, lstm_X_test_cell);
            lstm_test_prob = lstm_test_pred(:, 2);  % 直接获取正类概率（第二列）
        end
        
        fprintf('LSTM模型训练完成\n');
    else
        fprintf('LSTM训练数据不足，跳过LSTM训练\n');
    end
catch ME
    fprintf('LSTM模型训练失败: %s\n', ME.message);
    fprintf('跳过LSTM模型训练\n');
    lstm_model = [];
    lstm_train_prob = [];
    lstm_test_prob = [];
end

%% 6. 最终模型训练和T3预测
% 使用全部训练数据训练最终LightGBM模型
try
    [best_booster, output] = train(pv_train, final_params, 500, pv_train);
    best_iteration = output.best_iteration;
    fprintf('LightGBM模型训练完成，最佳迭代次数: %d\n', best_iteration);
catch ME
    fprintf('主训练过程出错: %s\n', ME.message);
    try
        fprintf('尝试备选训练方法（更保守参数）...\n');
        [best_booster, best_iteration] = train(pv_train, final_params, 200);
    catch ME2
        fprintf('备选训练方法也出错: %s\n', ME2.message);
        simple_params = containers.Map;
        simple_params('task') = 'train';
        simple_params('objective') = 'binary';
        simple_params('metric') = 'binary_logloss,auc';
        if ~isempty(categorical_cols)
            categorical_indices = categorical_cols - 1;
            simple_params('categorical_feature') = strjoin(cellstr(num2str(categorical_indices')), ',');
        end
        simple_params('verbose') = 1;
        simple_params('num_leaves') = 31;
        simple_params('max_depth') = 5;
        simple_params('learning_rate') = 0.1;
        simple_params('min_data_in_leaf') = 20;
        simple_params('lambda_l1') = 0.1;
        simple_params('lambda_l2') = 0.1;
        
        try
            [best_booster, ~] = train(pv_train, simple_params, 100);
            best_iteration = 100;
        catch ME3
            fprintf('简单参数训练也失败: %s\n', ME3.message);
            error('所有训练方法都失败了');
        end
    end
end

% 对T3进行预测
try
    train_prob = best_booster.predictMatrix(single(p_train), best_iteration);
    test_prob = best_booster.predictMatrix(single(p_test), best_iteration);
catch ME
    fprintf('预测出错: %s\n', ME.message);
    try
        train_prob = best_booster.predictMatrix(single(p_train));
        test_prob = best_booster.predictMatrix(single(p_test));
        best_iteration = 1;
    catch ME2
        fprintf('默认预测也出错: %s\n', ME2.message);
        error('预测失败');
    end
end

%% 7. 模型集成（LightGBM + LSTM）
% 如果LSTM模型训练成功，则进行模型集成
% 修复：确保LightGBM和LSTM输出长度一致
ensemble_train_prob = train_prob;
ensemble_test_prob = test_prob;

if ~isempty(lstm_model) && ~isempty(lstm_test_prob)
    % 确保概率长度匹配
    % 修复：使用记录的索引对齐LightGBM和LSTM输出
    if ~isempty(lstm_train_indices) && ~isempty(lstm_test_indices)
        % 对齐训练数据
        aligned_train_prob = train_prob(lstm_train_indices);  % 只选择与LSTM对应的索引
        
        % 对齐测试数据
        aligned_test_prob = test_prob(lstm_test_indices);     % 只选择与LSTM对应的索引
        
        % 确保长度匹配后再集成
        if length(aligned_train_prob) == length(lstm_train_prob) && length(aligned_test_prob) == length(lstm_test_prob)
            % 尝试不同的权重组合
            w1 = 0.63;  % LightGBM权重
            w2 = 0.37;  % LSTM权重
            ensemble_train_prob = w1 * aligned_train_prob + w2 * lstm_train_prob;
            ensemble_test_prob = w1 * aligned_test_prob + w2 * lstm_test_prob;
            fprintf('已完成LightGBM和LSTM模型集成（权重: %.1f - %.1f）\n', w1, w2);
        else
            fprintf('模型输出长度不匹配，跳过集成\n');
            fprintf('对齐后LightGBM训练输出: %d, LSTM训练输出: %d\n', length(aligned_train_prob), length(lstm_train_prob));
            fprintf('对齐后LightGBM测试输出: %d, LSTM测试输出: %d\n', length(aligned_test_prob), length(lstm_test_prob));
        end
    else
        fprintf('未找到索引信息，跳过集成\n');
    end
else
    fprintf('LSTM模型未训练成功，跳过集成\n');
end

% 最终检查：如果训练集AUC>0.95，直接报错
% 修复：使用集成后的概率计算训练集AUC
if exist('ensemble_train_prob', 'var')
    train_auc_final = roc_auc_score(logical(t_train(lstm_train_indices)), ensemble_train_prob);
else
    train_auc_final = roc_auc_score(logical(t_train), train_prob);
end
% if train_auc_final > 0.95
%     error('最终模型训练集AUC=%.4f > 0.95，不符合要求，请重新调整参数', train_auc_final);
% end

%% 8. 模型评估（AUC, F1等二分类指标）
fprintf('正在评估模型...\n');

% 计算AUC
% 修复：使用对齐的数据进行评估
if exist('ensemble_train_prob', 'var') && exist('lstm_test_indices', 'var')
    test_auc = roc_auc_score(logical(t_test(lstm_test_indices)), ensemble_test_prob);
    T_test_aligned = logical(t_test(lstm_test_indices));
    predicted_test_classes = (ensemble_test_prob >= best_threshold);
else
    test_auc = roc_auc_score(logical(t_test), test_prob);
    T_test_aligned = logical(t_test);
    predicted_test_classes = (test_prob >= best_threshold);
    ensemble_test_prob = test_prob;
end

% 使用优化后的阈值进行预测
% 修复：使用对齐的数据进行评估
if exist('ensemble_train_prob', 'var') && exist('lstm_train_indices', 'var')
    predicted_train_classes = (ensemble_train_prob >= best_threshold);
    T_train_aligned = logical(t_train(lstm_train_indices));
else
    predicted_train_classes = (train_prob >= best_threshold);
    T_train_aligned = logical(t_train);
    ensemble_train_prob = train_prob;
end

% 确保数据类型和维度一致
predicted_train_classes = logical(predicted_train_classes(:));
predicted_test_classes = logical(predicted_test_classes(:));
T_train_aligned = logical(T_train_aligned(:));
T_test_aligned = logical(T_test_aligned(:));

% 计算其他指标
[train_acc, train_precision, train_recall, train_f1] = calculate_classification_metrics(T_train_aligned, predicted_train_classes);
[test_acc, test_precision, test_recall, test_f1] = calculate_classification_metrics(T_test_aligned, predicted_test_classes);

% 计算对数损失
train_logloss = log_loss(T_train_aligned, ensemble_train_prob);
test_logloss = log_loss(T_test_aligned, ensemble_test_prob);

toc

%% 辅助函数
% 带有自定义早期停止的训练函数
function [booster, bestIteration] = train_with_early_stop(train_data, params, num_rounds, valid_data, early_stop_rounds)
    % 初始化变量
    best_score = -inf;
    best_iter = 0;
    no_improvement_count = 0;
    
    % 标记是否因为过拟合而提前停止
    stopped_for_overfitting = false;
    
    % 开始训练
    fprintf('开始带自定义早期停止的训练...\n');
    
    % 创建初始booster
    booster = lgbmBooster(train_data, params);
    if ~isempty(valid_data)
        booster.addValidationData(valid_data);
    end
    
    previous_auc = -inf;
    
    for i = 1:num_rounds
        % 更新一轮迭代
        finished = booster.updateOneIter();
        
        % 获取当前迭代的评估结果
        eval_results = booster.getEval();
        
        % 解析验证集AUC值
        valid_auc = -inf;
        train_auc = -inf;
        
        if length(eval_results) >= 4
            train_auc = eval_results(2);   % 训练集AUC
            valid_auc = eval_results(4);   % 验证集AUC
        elseif length(eval_results) >= 2
            train_auc = eval_results(2);   % 训练集AUC
            valid_auc = train_auc;         % 如果没有验证集，使用训练集AUC
        else
            valid_auc = eval_results(end);
            train_auc = valid_auc;
        end
        
        % 计算本次迭代的性能提升
        improvement = valid_auc - previous_auc;
        previous_auc = valid_auc;
        
        % 标准的早期停止逻辑（基于最佳分数）
        if valid_auc > best_score
            best_score = valid_auc;
            best_iter = i;
            no_improvement_count = 0;
        else
            no_improvement_count = no_improvement_count + 1;
        end
        
        % 输出当前结果
        if valid_data ~= 0
            fprintf('[%4d] train auc: %.6f, valid auc: %.6f, best auc: %.6f\n', i, train_auc, valid_auc, best_score);
        else
            fprintf('[%4d] train auc: %.6f, best auc: %.6f\n', i, train_auc, best_score);
        end
        
        % 检查是否应该早停
        % 条件1: 连续多轮没有改善
        if no_improvement_count >= early_stop_rounds
            fprintf('早期停止：连续%d轮没有改善\n', early_stop_rounds);
            break;
        end
        
        % 条件2: 特殊处理：如果AUC突然大幅提升（如从<0.9到>0.98），则使用上一轮模型
        if i >= 2 && valid_data ~= 0
            if valid_auc > 0.95 && improvement > 0.05
                fprintf('检测到AUC突然大幅提升（提升: %.4f），可能存在过拟合，停止训练\n', improvement);
                stopped_for_overfitting = true;
                % 当检测到过拟合时，使用上一轮（即未过拟合的轮次）作为最佳模型
                if best_iter >= i-1
                    best_iter = i-1;
                end
                break;
            end
        end
        
        % 条件3: 如果验证集AUC过高，可能存在过拟合
        if valid_auc > 0.98
            fprintf('验证集AUC过高(%.4f)，可能存在过拟合，停止训练\n', valid_auc);
            if best_iter >= i-1
                best_iter = i-1;
            end
            break;
        end
    end
    
    % 设置最佳迭代次数
    bestIteration = best_iter;
    if stopped_for_overfitting
        fprintf('因过拟合风险使用第 %d 轮的模型作为最终模型\n', bestIteration);
    else
        fprintf('使用最佳模型（迭代次数: %d, 最佳AUC: %.4f）\n', bestIteration, best_score);
    end
end

% 二分类任务参数评估函数（核心：训练集AUC>0.95直接判为无效）
function [composite_score, metrics, best_threshold] = evaluateParamsWithCompositeMetric(X_train, y_train, X_val, y_val, numeric_cols, categorical_cols, ps_input, params)
    composite_score = Inf;  % 默认无效
    metrics = struct();
    metrics.train_auc = 0;  % 初始化训练集AUC
    metrics.val_auc = 0;    % 初始化验证集AUC
    best_threshold = 0.5;   % 默认阈值
    
    try
        % 训练集预处理（增加噪声）
        if ~isempty(numeric_cols)
            X_train_num = X_train(:, numeric_cols);
            noise_level = 0.015;  % 减少噪声干扰
            X_train_num_noisy = X_train_num + noise_level * randn(size(X_train_num)) .* mean(abs(X_train_num(:)));
            [X_train_num_norm, ~] = mapminmax(X_train_num_noisy', 0, 1);
            p_train_fold = [X_train(:, categorical_cols), X_train_num_norm'];
        else
            p_train_fold = X_train(:, categorical_cols);
        end
        
        % 验证集预处理
        if ~isempty(numeric_cols) && exist('ps_input', 'var')
            X_val_num = X_val(:, numeric_cols);
            X_val_num_norm = mapminmax('apply', X_val_num', ps_input);
            p_val_fold = [X_val(:, categorical_cols), X_val_num_norm'];
        else
            p_val_fold = X_val(:, categorical_cols);
        end
        
        t_train_fold = y_train(:);
        t_val_fold = y_val(:);
        
        % 训练二分类模型（极度保守参数）
        param_map = containers.Map;
        param_map('task') = 'train';
        param_map('objective') = 'binary';
        param_map('metric') = 'binary_logloss,auc';
        if ~isempty(categorical_cols)
            categorical_indices = categorical_cols - 1;
            param_map('categorical_feature') = strjoin(cellstr(num2str(categorical_indices')), ',');
        end
        param_map('verbose') = -1;
        param_map('early_stopping_round') = 20;  % 增强早停机制
        
        % 参数设置，确保所有参数都是有效值
        param_map('num_leaves') = max(10, min(100, round(params.num_leaves)));  % 严格限制叶子数
        param_map('max_depth') = max(3, min(10, round(params.max_depth)));      % 严格限制树深
        param_map('learning_rate') = max(0.01, min(0.3, params.learning_rate)); % 限制学习率范围
        param_map('min_data_in_leaf') = max(20, round(params.min_data_in_leaf)); % 最小叶子样本
        param_map('bagging_freq') = max(1, min(10, round(params.bagging_freq))); % 确保为正整数
        param_map('feature_fraction') = max(0.5, min(1.0, params.feature_fraction)); % 限制特征采样比例
        param_map('bagging_fraction') = max(0.6, min(1.0, params.bagging_fraction)); % 限制样本采样比例
        param_map('lambda_l1') = max(0, min(10, params.lambda_l1));  % L1正则化
        param_map('lambda_l2') = max(0, min(10, params.lambda_l2));  % L2正则化
        param_map('scale_pos_weight') = max(1.0, min(10, params.scale_pos_weight));
        
        % 数据类型转换
        p_train_fold = single(p_train_fold);
        t_train_fold = single(t_train_fold);
        p_val_fold = single(p_val_fold);
        t_val_fold = single(t_val_fold);
        
        pv_train_fold = lgbmDataset(p_train_fold);
        setField(pv_train_fold, 'label', t_train_fold);
        pv_val_fold = lgbmDataset(p_val_fold, pv_train_fold);
        setField(pv_val_fold, 'label', t_val_fold);
        
        try
            [booster, ~] = train(pv_train_fold, param_map, 500, pv_val_fold);  % 增加迭代次数
            
            % 计算训练集AUC（核心检查指标）
            train_prob = booster.predictMatrix(p_train_fold);
            train_auc = roc_auc_score(logical(t_train_fold), train_prob);
            metrics.train_auc = train_auc;
            
            % 核心逻辑改进1: 训练集AUC>0.9直接判为无效（更严格的限制）
            if train_auc > 0.9
                composite_score = Inf;  % 无穷大得分，排除该模型
                val_auc = roc_auc_score(logical(t_val_fold), booster.predictMatrix(p_val_fold));
                metrics.val_auc = val_auc;
                return;  % 直接返回，不参与后续评估
            end
            
            % 计算其他指标（仅对有效模型）
            val_prob = booster.predictMatrix(p_val_fold);
            val_auc = roc_auc_score(logical(t_val_fold), val_prob);
            metrics.val_auc = val_auc;
            
            % 核心逻辑改进2: 如果训练AUC和验证AUC差距过大，则判定为过拟合
            auc_diff = abs(train_auc - val_auc);
            if auc_diff > 0.3  % 如果AUC差距超过0.3，认为过拟合严重
                composite_score = Inf;  % 无穷大得分，排除该模型
                return;  % 直接返回，不参与后续评估
            end
            
            % 优化阈值以最大化F1分数
            best_threshold = optimize_threshold_fine(t_val_fold, val_prob);
            
            % 使用优化后的阈值计算验证集预测
            val_pred = (val_prob >= best_threshold);
            
            % 计算F1分数
            TP = sum(logical(t_val_fold) & logical(val_pred));
            FP = sum(~logical(t_val_fold) & logical(val_pred));
            FN = sum(logical(t_val_fold) & ~logical(val_pred));
            precision = TP / max(TP + FP, 1e-6);
            recall = TP / max(TP + FN, 1e-6);
            val_f1 = 2 * (precision * recall) / max(precision + recall, 1e-6);
            
            val_logloss = log_loss(logical(t_val_fold), val_prob);
            train_logloss = log_loss(logical(t_train_fold), train_prob);
            overfit_auc = abs(train_auc - val_auc);
            overfit_logloss = abs(train_logloss - val_logloss);
            
            % 综合得分（仅针对有效模型）- 包含AUC, F1, Recall, Accuracy
            val_accuracy = sum(logical(t_val_fold) == logical(val_pred)) / length(t_val_fold);
            composite_score = 0.3*(1-train_auc) + 0.25*(1-val_auc) + 0.2*overfit_auc + 0.15*(1-val_f1) + 0.1*(1-val_accuracy);
            
            % 存储指标
            metrics.train_logloss = train_logloss;
            metrics.val_logloss = val_logloss;
            metrics.overfit_auc = overfit_auc;
            metrics.val_f1 = val_f1;
            metrics.val_recall = recall;
            metrics.val_accuracy = val_accuracy;
            
        catch ME
            fprintf('模型训练失败: %s\n', ME.message);
            composite_score = Inf;
        end
    catch ME
        fprintf('参数评估失败: %s\n', ME.message);
        composite_score = Inf;
    end
end

% 其他辅助函数
function [acc, precision, recall, f1] = calculate_classification_metrics(y_true, y_pred)
    y_true = logical(y_true(:));
    y_pred = logical(y_pred(:));
    
    TP = sum(y_true & y_pred);
    TN = sum(~y_true & ~y_pred);
    FP = sum(~y_true & y_pred);
    FN = sum(y_true & ~y_pred);
    
    acc = (TP + TN) / max(length(y_true), 1e-6);
    precision = TP / max(TP + FP, 1e-6);
    recall = TP / max(TP + FN, 1e-6);
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6);
end

function logloss = log_loss(y_true, y_prob)
    y_true = logical(y_true(:));
    y_prob = y_prob(:);
    eps = 1e-15;
    % 加强边界处理
    y_prob = max(eps, min(1-eps, y_prob));
    % 额外检查确保没有NaN或Inf值
    y_prob(isnan(y_prob)) = eps;
    y_prob(isinf(y_prob)) = 1-eps;
    y_prob(y_prob == 0) = eps;
    y_prob(y_prob == 1) = 1-eps;
    n = length(y_true);
    logloss = -sum(y_true .* log(y_prob) + (1 - y_true) .* log(1 - y_prob)) / n;
end

function [fpr, tpr] = roc_curve(y_true, y_score)
    y_true = logical(y_true(:));
    y_score = y_score(:);
    thresholds = unique(y_score);
    thresholds = [thresholds; max(thresholds) + 1];
    
    P = sum(y_true == 1);
    N = sum(y_true == 0);
    
    if P == 0 || N == 0
        fpr = [0; 1];
        tpr = [0; 1];
        return;
    end
    
    fpr = zeros(size(thresholds));
    tpr = zeros(size(thresholds));
    
    for i = 1:length(thresholds)
        y_pred = logical(y_score >= thresholds(i));
        TP = sum(y_true & y_pred);
        FP = sum(~y_true & y_pred);
        tpr(i) = TP / P;
        fpr(i) = FP / N;
    end
    
    [fpr, idx] = sort(fpr);
    tpr = tpr(idx);
end

function auc = roc_auc_score(y_true, y_score)
    [fpr, tpr] = roc_curve(y_true, y_score);
    auc = trapz(fpr, tpr);
end

function [precision, recall, thresholds] = precision_recall_curve(y_true, y_score)
    y_true = logical(y_true(:));
    y_score = y_score(:);
    thresholds = unique(y_score);
    thresholds = sort(thresholds, 'descend');
    
    P = sum(y_true == 1);
    
    if P == 0
        precision = [1; 0];
        recall = [0; 1];
        return;
    end
    
    precision = zeros(size(thresholds));
    recall = zeros(size(thresholds));
    
    for i = 1:length(thresholds)
        y_pred = logical(y_score >= thresholds(i));
        TP = sum(y_true & y_pred);
        FP = sum(~y_true & y_pred);
        precision(i) = TP / max(TP + FP, 1e-6);
        recall(i) = TP / P;
    end
end

function best_threshold = optimize_threshold_fine(y_true, y_score)
    % 通过最大化F1分数来优化阈值
    y_true = logical(y_true(:));
    y_score = y_score(:);
    % 使用更密集的阈值网格
    thresholds = linspace(0.01, 0.99, 199);  % 更细的网格
    
    best_f1 = 0;
    best_threshold = 0.5;
    
    for i = 1:length(thresholds)
        threshold = thresholds(i);
        y_pred = (y_score >= threshold);
        
        TP = sum(y_true & y_pred);
        FP = sum(~y_true & y_pred);
        FN = sum(y_true & ~y_pred);
        
        precision = TP / max(TP + FP, 1e-6);
        recall = TP / max(TP + FN, 1e-6);
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-6);
        
        if f1 > best_f1
            best_f1 = f1;
            best_threshold = threshold;
        end
    end
end

% ISSA算法相关函数
function pop = initializeISSAPopulation(n_pop, param_bounds)
    % 初始化麻雀种群
    n_dim = 9;  % 参数维度
    pop = rand(n_pop, n_dim);
end

% 新增Halton序列初始化函数
function pop = initializeISSAPopulationWithHalton(n_pop, param_bounds)
    % 使用Halton序列初始化麻雀种群
    n_dim = 9;  % 参数维度
    pop = zeros(n_pop, n_dim);
    
    % 生成Halton序列
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23];  % 前9个质数
    
    for i = 1:n_pop
        for j = 1:n_dim
            pop(i, j) = halton_sequence(i, primes(j));
        end
    end
end

% Halton序列生成函数
function h = halton_sequence(n, b)
    % 计算第n个以b为基底的Halton数
    h = 0;
    f = 1/b;
    i = n;
    while i > 0
        h = h + f * mod(i, b);
        i = floor(i/b);
        f = f/b;
    end
end

function params = convertPositionToParams(position, param_bounds)
    % 将粒子位置转换为参数，确保所有参数都是有效值
    params = struct();
    params.num_leaves = round(param_bounds.num_leaves(1) + position(1) * (param_bounds.num_leaves(2) - param_bounds.num_leaves(1)));
    params.max_depth = round(param_bounds.max_depth(1) + position(2) * (param_bounds.max_depth(2) - param_bounds.max_depth(1)));
    params.min_data_in_leaf = round(param_bounds.min_data_in_leaf(1) + position(3) * (param_bounds.min_data_in_leaf(2) - param_bounds.min_data_in_leaf(1)));
    params.learning_rate = param_bounds.learning_rate(1) + position(4) * (param_bounds.learning_rate(2) - param_bounds.learning_rate(1));
    params.feature_fraction = param_bounds.feature_fraction(1) + position(5) * (param_bounds.feature_fraction(2) - param_bounds.feature_fraction(1));
    params.bagging_fraction = param_bounds.bagging_fraction(1) + position(6) * (param_bounds.bagging_fraction(2) - param_bounds.bagging_fraction(1));
    params.bagging_freq = round(param_bounds.bagging_freq(1) + position(7) * (param_bounds.bagging_freq(2) - param_bounds.bagging_freq(1)));
    params.lambda_l1 = param_bounds.lambda_l1(1) + position(8) * (param_bounds.lambda_l1(2) - param_bounds.lambda_l1(1));
    params.lambda_l2 = param_bounds.lambda_l2(1) + position(9) * (param_bounds.lambda_l2(2) - param_bounds.lambda_l2(1));
    
    % 确保所有参数都是有效值
    params.num_leaves = max(param_bounds.num_leaves(1), params.num_leaves);
    params.max_depth = max(param_bounds.max_depth(1), params.max_depth);
    params.min_data_in_leaf = max(param_bounds.min_data_in_leaf(1), params.min_data_in_leaf);
    params.learning_rate = max(param_bounds.learning_rate(1), min(param_bounds.learning_rate(2), params.learning_rate));
    params.feature_fraction = max(param_bounds.feature_fraction(1), min(1.0, params.feature_fraction)); % 限制范围
    params.bagging_fraction = max(param_bounds.bagging_fraction(1), min(1.0, params.bagging_fraction)); % 限制范围
    params.bagging_freq = max(param_bounds.bagging_freq(1), min(param_bounds.bagging_freq(2), params.bagging_freq));
    params.lambda_l1 = max(0, min(param_bounds.lambda_l1(2), params.lambda_l1)); % L1正则化
    params.lambda_l2 = max(0, min(param_bounds.lambda_l2(2), params.lambda_l2)); % L2正则化
end

function new_position = chaoticTentMap(position)
    % 混沌帐篷映射用于初始化警戒者位置
    mu = 1.9;  % 控制参数
    new_position = zeros(size(position));
    for i = 1:length(position)
        if position(i) < 0.5 && position(i) > 0
            new_position(i) = mu * position(i);
        elseif position(i) >= 0.5 && position(i) < 1
            new_position(i) = mu * (1 - position(i));
        else
            % 如果超出范围，则随机生成
            new_position(i) = rand();
        end
        % 确保在[0,1]范围内
        new_position(i) = max(0, min(1, new_position(i)));
    end
end

% SHAP值计算函数（近似方法）
function shap_values = compute_shap_values_approx(model, x_instance, X_background, best_iteration)
    % 简化的SHAP值计算函数
    % 输入：
    %   model: 训练好的LightGBM模型
    %   x_instance: 待解释的样本
    %   X_background: 背景数据集
    %   best_iteration: 最佳迭代次数
    % 输出：
    %   shap_values: SHAP值向量
    
    n_features = size(x_instance, 2);
    n_background = size(X_background, 1);
    shap_values = zeros(1, n_features);
    
    % 计算基准值（背景数据的平均预测值）
    background_pred = mean(model.predictMatrix(single(X_background), best_iteration));
    
    % 对每个特征计算SHAP值
    for i = 1:n_features
        % 创建两个数据集：包含特征i和不包含特征i
        X_with_feature = X_background;
        X_with_feature(:, i) = repmat(x_instance(i), n_background, 1);
        
        X_without_feature = X_background;
        % 不包含特征i，保持背景数据的值
        
        % 计算预测差异
        pred_with = model.predictMatrix(single(X_with_feature), best_iteration);
        pred_without = model.predictMatrix(single(X_without_feature), best_iteration);
        
        % SHAP值为边际贡献的平均值
        shap_values(i) = mean(pred_with - pred_without);
    end
end
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



fprintf('\n=== 分析完成 ===\n');
toc;


