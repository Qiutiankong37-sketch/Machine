warning off; 
close all; 
clear; 
clc;
tic

%% 配置循环参数
num_iterations = 10;  % 循环次数，可根据需要调整

% 中文列名映射（与Excel文件中的列名对应）
chinese_feature_names = {
    '年级', '性别', '父母是否在外', '家庭氛围', '是否是独生子女', '民族',...
    '父亲最高学历', '母亲最高学历', '年龄', '父亲在外地', '母亲在外地', '失眠',...
    '高敏感AES', '高敏感EOE', '高敏感LST', '网络成瘾',...
    '抑郁', '焦虑'
};

% 英文列名（用于显示和处理）
english_feature_names = {
    'Grade', 'Gender', 'BothParentsAtHome', 'FamilyEnvironment', 'OnlyChild', 'Ethnicity',...
    'FatherHighestEducation', 'MotherHighestEducation', 'Age', 'FatherOutsideWork', 'MotherOutsideWork', 'Insomnia',...
    'HighSensitivityAES', 'HighSensitivityEOE', 'HighSensitivityLST', 'Netaddiction',...
    'Depression', 'Anxiety'
};

feature_names = english_feature_names;  % 默认使用英文名称
output_name_chinese = '受欺凌';  % 目标变量中文名称
output_name_english = 'Bullying';  % 目标变量英文名称
output_name = output_name_english;  % 使用英文名称

%% 1. 数据处理（目标二分类，特征含多分类）
addpath('Lightgbm_toolbox\');
loadlibrary('lib_lightgbm.dll', 'c_api.h');

% 导入数据
data = readtable('T20816.xlsx');  
res = table2array(data);

% 验证数据规模
num_features = size(res, 2) - 1;
num_samples = size(res, 1);
fprintf('数据集信息：%d特征，%d样本\n', num_features, num_samples);

% 特征类型定义（明确多分类特征）
categorical_cols = 1:8;  % 多分类特征索引
numeric_cols = setdiff(1:num_features, categorical_cols);

% 多分类特征编码
if ~isempty(categorical_cols)
    for col = categorical_cols
        if col <= size(data, 2)
            if ~iscategorical(data.(col))
                data.(col) = categorical(data.(col));  % 转为分类变量
            end
            res(:, col) = double(data.(col));  % 整数编码（1,2,3...）
            unique_vals = unique(res(:, col));
            fprintf('特征%d为多分类，共%d个类别\n', col, length(unique_vals));
        end
    end
end

% 划分输入输出（目标为二分类）
X = res(:, 1:end-1);
y = res(:, end);

% 目标变量二分类处理（确保0/1标签）
unique_y = unique(y);
if length(unique_y) ~= 2
    error('目标变量不是二分类变量，有%d个唯一值', length(unique_y));
    error('目标变量不是二分类变量，有%d个唯一值', length(unique_y));
end
if all(ismember(unique_y, [0,1]))
    fprintf('目标变量已经是0/1编码\n');
else
    y(y == unique_y(1)) = 0;
    y(y == unique_y(2)) = 1;
    fprintf('目标变量已转换为0/1编码\n');
end
y = y(:);  % 确保是列向量

% 计算目标类别比例（二分类不平衡检查）
pos_ratio = mean(y);
fprintf('二分类目标：正例比例 = %.2f%%\n', pos_ratio*100);

%% 2. 数据划分与预处理（一次性划分，也可改为每轮重新划分）
train_ratio = 0.6;
val_ratio = 0.1;
test_ratio = 0.3;

% 分层抽样（确保测试集有足够正例）
min_pos_samples = 3;
max_attempts = 10000;
attempt = 0;

while attempt < max_attempts
    attempt = attempt + 1;
    try
        cv = cvpartition(y, 'HoldOut', 1-train_ratio, 'Stratify', true);
        train_idx = training(cv);
        remaining_idx = test(cv);
        
        remaining_pos_count = sum(y(remaining_idx) == 1);
        if remaining_pos_count < min_pos_samples
            error('剩余样本中正例不足');
        end
        
        cv_val = cvpartition(y(remaining_idx), 'HoldOut', val_ratio/(val_ratio+test_ratio), 'Stratify', true);
        val_idx = remaining_idx(training(cv_val));
        test_idx = remaining_idx(test(cv_val));
        
        test_pos_count = sum(y(test_idx) == 1);
        if test_pos_count >= min_pos_samples
            break;
        end
    catch
        if attempt == max_attempts
            fprintf('自动分层抽样失败，使用手动分层抽样...\n');
            break;
        end
    end
end

% 手动分层抽样（自动抽样失败时）
if attempt == max_attempts || test_pos_count < min_pos_samples
    fprintf('使用手动分层抽样...\n');
    pos_indices = find(y == 1);
    neg_indices = find(y == 0);
    
    n_pos = length(pos_indices);
    n_neg = length(neg_indices);
    
    n_test_pos = max(min_pos_samples, round(test_ratio * n_pos));
    n_test_pos = min(n_test_pos, n_pos - 2);
    remaining_pos = n_pos - n_test_pos;
    n_val_pos = max(1, round(val_ratio * n_pos));
    n_val_pos = min(n_val_pos, remaining_pos - 1);
    n_train_pos = remaining_pos - n_val_pos;
    
    n_test_neg = round(test_ratio * n_neg);
    n_val_neg = round(val_ratio * n_neg);
    n_train_neg = n_neg - n_test_neg - n_val_neg;
    
    pos_indices = pos_indices(randperm(length(pos_indices)));
    neg_indices = neg_indices(randperm(length(neg_indices)));
    
    train_idx_pos = pos_indices(1:n_train_pos);
    val_idx_pos = pos_indices(n_train_pos+1 : n_train_pos+n_val_pos);
    test_idx_pos = pos_indices(n_train_pos+n_val_pos+1 : end);
    
    train_idx_neg = neg_indices(1:n_train_neg);
    val_idx_neg = neg_indices(n_train_neg+1 : n_train_neg+n_val_neg);
    test_idx_neg = neg_indices(n_train_neg+n_val_neg+1 : end);
    
    train_idx = [train_idx_pos; train_idx_neg];
    val_idx = [val_idx_pos; val_idx_neg];
    test_idx = [test_idx_pos; test_idx_neg];
    
    train_idx = train_idx(randperm(length(train_idx)));
    val_idx = val_idx(randperm(length(val_idx)));
    test_idx = test_idx(randperm(length(test_idx)));
end

% 训练集预处理
X_train = X(train_idx, :);
y_train = y(train_idx);
if ~isempty(numeric_cols)
    X_train_num = X_train(:, numeric_cols);
    noise_level = 0.0001;
    X_train_num_noisy = X_train_num + noise_level * randn(size(X_train_num)) .* mean(X_train_num);
    [X_train_num_norm, ps_input] = mapminmax(X_train_num_noisy', 0, 1);
    p_train = [X_train(:, categorical_cols), X_train_num_norm'];
else
    p_train = X_train(:, categorical_cols);
end

% 验证集预处理
X_val = X(val_idx, :);
y_val = y(val_idx);
if ~isempty(numeric_cols) && exist('ps_input', 'var')
    X_val_num = X_val(:, numeric_cols);
    X_val_num_norm = mapminmax('apply', X_val_num', ps_input);
    p_val = [X_val(:, categorical_cols), X_val_num_norm'];
else
    p_val = X_val(:, categorical_cols);
end

% 测试集预处理
X_test = X(test_idx, :);
y_test = y(test_idx);
if ~isempty(numeric_cols) && exist('ps_input', 'var')
    X_test_num = X_test(:, numeric_cols);
    X_test_num_norm = mapminmax('apply', X_test_num', ps_input);
    p_test = [X_test(:, categorical_cols), X_test_num_norm'];
else
    p_test = X_test(:, categorical_cols);
end

% 输出处理
t_train = y_train;
t_val = y_val;
t_test = y_test;

% 记录类别分布
train_class_dist = [sum(y_train==0), sum(y_train==1)];
val_class_dist = [sum(y_val==0), sum(y_val==1)];
test_class_dist = [sum(y_test==0), sum(y_test==1)];
fprintf('训练集类别分布: 负例=%d, 正例=%d\n', train_class_dist(1), train_class_dist(2));
fprintf('验证集类别分布: 负例=%d, 正例=%d\n', val_class_dist(1), val_class_dist(2));
fprintf('测试集类别分布: 负例=%d, 正例=%d\n', test_class_dist(1), test_class_dist(2));

%% 3. 循环训练模型（每轮都重新优化参数）并保存SHAP重要性图
for loop_idx = 1:num_iterations
    fprintf('\n===== 开始第 %d/%d 轮循环 =====\n', loop_idx, num_iterations);
    
    %% 每轮都重新进行交叉验证参数优化
    best_params = struct();
    best_mean_score = Inf;

    % 存储迭代过程中的最优参数
    iteration_best_params = cell(0);  
    iteration_best_scores = [];       
    iteration_best_aucs = [];         

    % 参数范围定义
    param_bounds = struct();
    param_bounds.num_leaves = [0, 32];           
    param_bounds.max_depth = [2, 8];              
    param_bounds.min_data_in_leaf = [50, 200];    
    param_bounds.learning_rate = [0.02, 0.5];     
    param_bounds.feature_fraction = [0.4, 1.0];   
    param_bounds.bagging_fraction = [0.4, 1.0];   
    param_bounds.bagging_freq = [3, 7];           
    param_bounds.lambda_l1 = [0, 25];             
    param_bounds.lambda_l2 = [0, 25];             

    % 默认参数
    default_params.num_leaves = 15;
    default_params.max_depth = 4;
    default_params.min_data_in_leaf = 50;
    default_params.learning_rate = 0.1;
    default_params.feature_fraction = 0.8;
    default_params.bagging_fraction = 0.8;
    default_params.bagging_freq = 5;
    default_params.lambda_l1 = 2;
    default_params.lambda_l2 = 2;
    default_params.scale_pos_weight = max(1.0, (1-pos_ratio)/pos_ratio);

    % 贝叶斯优化设置（减少迭代次数以节省时间）
    n_initial_points = 50;  % 减少初始点数量
    n_iterations = 500;     % 减少迭代次数

    fprintf('第 %d 轮：开始贝叶斯优化参数搜索...\n', loop_idx);

    % 存储评估结果
    evaluated_params = {};
    evaluated_scores = [];
    evaluated_train_aucs = []; 
    valid_model_count = 0;  

    % 初始化：随机采样初始点
    for i = 1:n_initial_points
        % 随机生成参数
        params = struct();
        params.num_leaves = randi(param_bounds.num_leaves);
        params.max_depth = randi(param_bounds.max_depth);
        params.min_data_in_leaf = randi(param_bounds.min_data_in_leaf);
        params.learning_rate = param_bounds.learning_rate(1) + ...
            rand() * (param_bounds.learning_rate(2) - param_bounds.learning_rate(1));
        params.feature_fraction = param_bounds.feature_fraction(1) + ...
            rand() * (param_bounds.feature_fraction(2) - param_bounds.feature_fraction(1));
        params.bagging_fraction = param_bounds.bagging_fraction(1) + ...
            rand() * (param_bounds.bagging_fraction(2) - param_bounds.bagging_fraction(1));
        params.bagging_freq = randi(param_bounds.bagging_freq);
        params.lambda_l1 = param_bounds.lambda_l1(1)+...
            rand()*(param_bounds.lambda_l1(2)-param_bounds.lambda_l1(1));
        params.lambda_l2 = param_bounds.lambda_l2(1)+...
            rand()*(param_bounds.lambda_l2(2)-param_bounds.lambda_l2(1));
        params.scale_pos_weight = default_params.scale_pos_weight;

        % 评估参数
        [composite_score, metrics] = evaluateParamsWithCompositeMetric(...
            X_train, y_train, X_val, y_val, numeric_cols, categorical_cols, ps_input, params);

        if isfinite(composite_score) && metrics.train_auc > 0.5 && metrics.train_auc < 1.0
            valid_model_count = valid_model_count + 1;
            evaluated_params{end+1} = params;
            evaluated_scores(end+1) = composite_score;
            evaluated_train_aucs(end+1) = metrics.train_auc;
            
            if composite_score < best_mean_score
                best_mean_score = composite_score;
                best_params = params;

                iteration_best_params{end+1} = best_params;
                iteration_best_scores(end+1) = best_mean_score;
                iteration_best_aucs(end+1) = metrics.train_auc;
            end
        end
    end

    % 贝叶斯优化迭代
    for iter = 1:n_iterations
        next_params = suggestNextParams(evaluated_params, evaluated_scores, param_bounds);

        [composite_score, metrics] = evaluateParamsWithCompositeMetric(...
            X_train, y_train, X_val, y_val, numeric_cols, categorical_cols, ps_input, next_params);

        if isfinite(composite_score) && metrics.train_auc > 0.5 && metrics.train_auc < 1.0
            valid_model_count = valid_model_count + 1;
            evaluated_params{end+1} = next_params;
            evaluated_scores(end+1) = composite_score;
            evaluated_train_aucs(end+1) = metrics.train_auc;

            if composite_score < best_mean_score
                best_mean_score = composite_score;
                best_params = next_params;

                iteration_best_params{end+1} = best_params;
                iteration_best_scores(end+1) = best_mean_score;
                iteration_best_aucs(end+1) = metrics.train_auc;
            end
        end
    end

    % 检查是否有有效模型
    if valid_model_count == 0
        fprintf('第 %d 轮：未找到有效模型，使用默认参数...\n', loop_idx);
        best_params = default_params;
        [composite_score, metrics] = evaluateParamsWithCompositeMetric(...
            X_train, y_train, X_val, y_val, numeric_cols, categorical_cols, ps_input, best_params);
        if ~isfinite(composite_score) || metrics.train_auc <= 0.5 || metrics.train_auc >= 1.0
            fprintf('第 %d 轮：默认参数也无效，使用备用简单参数...\n', loop_idx);
            best_params = struct();
            best_params.num_leaves = 12;
            best_params.max_depth = 3;
            best_params.min_data_in_leaf = 30;
            best_params.learning_rate = 0.08;
            best_params.feature_fraction = 0.7;
            best_params.bagging_fraction = 0.7;
            best_params.bagging_freq = 4;
            best_params.lambda_l1 = 1;
            best_params.lambda_l2 = 1;
            best_params.scale_pos_weight = default_params.scale_pos_weight;
        end

        iteration_best_params{end+1} = best_params;
        iteration_best_scores(end+1) = composite_score;
        iteration_best_aucs(end+1) = metrics.train_auc;
    else
        candidate_indices = find(evaluated_train_aucs <= 0.9);
        if isempty(candidate_indices)
            fprintf('第 %d 轮：无符合AUC≤0.9的模型，使用训练集AUC最低的模型...\n', loop_idx);
            [~, min_idx] = min(evaluated_train_aucs);
            best_params = evaluated_params{min_idx};
            best_candidate_auc = evaluated_train_aucs(min_idx);
        else
            candidate_scores = evaluated_scores(candidate_indices);
            [~, best_candidate_idx] = min(candidate_scores);
            best_params = evaluated_params{candidate_indices(best_candidate_idx)};
            best_candidate_auc = evaluated_train_aucs(candidate_indices(best_candidate_idx));
        end

        fprintf('第 %d 轮：贝叶斯优化完成，共保留%d个有效模型，最佳候选模型训练AUC=%.4f\n', ...
            loop_idx, valid_model_count, best_candidate_auc);
    end
    
    %% 训练最终模型
    num_classes = 2;
    fprintf('第 %d 轮：训练最终模型...\n', loop_idx);
    
    if isempty(p_train) || isempty(t_train)
        error('训练数据为空');
    end
    if isempty(p_val) || isempty(t_val)
        error('验证数据为空');
    end
    
    try
        pv_train = lgbmDataset(single(p_train));
        setField(pv_train, 'label', single(t_train));
        pv_val = lgbmDataset(single(p_val), pv_train);
        setField(pv_val, 'label', single(t_val));
    catch ME
        fprintf('数据集创建失败: %s\n', ME.message);
        error('无法创建LightGBM数据集');
    end
    
    % 构建参数
    final_params = containers.Map;
    final_params('task') = 'train';
    final_params('objective') = 'binary';
    final_params('metric') = 'binary_logloss,auc';
    if ~isempty(categorical_cols)
        categorical_indices = categorical_cols - 1;
        final_params('categorical_feature') = strjoin(cellstr(num2str(categorical_indices')), ',');
    end
    final_params('verbose') = 1;
    final_params('early_stopping_round') = 8;
    
    % 设置最佳参数
    final_params('num_leaves') = max(6, min(best_params.num_leaves, 31));
    final_params('max_depth') = max(3, min(best_params.max_depth, 6));
    final_params('learning_rate') = max(0.02, min(best_params.learning_rate, 0.3));
    final_params('min_data_in_leaf') = max(20, best_params.min_data_in_leaf);
    final_params('bagging_freq') = best_params.bagging_freq;
    final_params('feature_fraction') = min(1.0, max(0.5, best_params.feature_fraction));
    final_params('bagging_fraction') = min(1.0, max(0.7, best_params.bagging_fraction));
    final_params('lambda_l1') = max(0, best_params.lambda_l1);
    final_params('lambda_l2') = max(0, best_params.lambda_l2);
    final_params('scale_pos_weight') = best_params.scale_pos_weight;
    
    % 训练模型
    try
        fprintf('第 %d 轮：尝试直接训练模型...\n', loop_idx);
        % 使用自定义早停训练函数
        [best_booster, best_iteration] = train_with_early_stop(pv_train, final_params, 100, pv_val, 8);
        fprintf('第 %d 轮：直接训练完成，最佳迭代次数: %d\n', loop_idx, best_iteration);
    catch ME
        fprintf('第 %d 轮：直接训练失败: %s\n', loop_idx, ME.message);
        try
            fprintf('第 %d 轮：尝试备选训练方法（调整参数）...\n', loop_idx);
            adjusted_params = final_params;
            adjusted_params('num_leaves') = 20;
            adjusted_params('max_depth') = 4;
            adjusted_params('learning_rate') = 0.08;
            adjusted_params('min_data_in_leaf') = 40;
            adjusted_params('lambda_l1') = 1;
            adjusted_params('lambda_l2') = 1;
            adjusted_params('feature_fraction') = 0.8;
            adjusted_params('bagging_fraction') = 0.8;
            
            [best_booster, best_iteration] = train_with_early_stop(pv_train, adjusted_params, 80, pv_val, 8);
            fprintf('第 %d 轮：使用调整参数训练完成，最佳迭代次数: %d\n', loop_idx, best_iteration);
        catch ME2
            fprintf('第 %d 轮：调整参数训练也出错: %s\n', loop_idx, ME2.message);
            simple_params = containers.Map;
            simple_params('task') = 'train';
            simple_params('objective') = 'binary';
            simple_params('metric') = 'binary_logloss,auc';
            if ~isempty(categorical_cols)
                categorical_indices = categorical_cols - 1;
                simple_params('categorical_feature') = strjoin(cellstr(num2str(categorical_indices')), ',');
            end
            simple_params('verbose') = 1;
            simple_params('num_leaves') = 15;
            simple_params('max_depth') = 4;
            simple_params('learning_rate') = 0.05;
            simple_params('min_data_in_leaf') = 50;
            simple_params('lambda_l1') = 0;
            simple_params('lambda_l2') = 0;
            simple_params('feature_fraction') = 0.9;
            simple_params('bagging_fraction') = 0.9;
            
            try
                [best_booster, best_iteration] = train_with_early_stop(pv_train, simple_params, 60, pv_val, 8);
                fprintf('第 %d 轮：使用简单参数训练完成，最佳迭代次数: %d\n', loop_idx, best_iteration);
            catch ME3
                fprintf('第 %d 轮：简单参数训练也失败: %s\n', loop_idx, ME3.message);
                ultra_simple_params = containers.Map;
                ultra_simple_params('task') = 'train';
                ultra_simple_params('objective') = 'binary';
                ultra_simple_params('metric') = 'binary_logloss,auc';
                if ~isempty(categorical_cols)
                    categorical_indices = categorical_cols - 1;
                    ultra_simple_params('categorical_feature') = strjoin(cellstr(num2str(categorical_indices')), ',');
                end
                ultra_simple_params('verbose') = 1;
                ultra_simple_params('num_leaves') = 12;
                ultra_simple_params('max_depth') = 3;
                ultra_simple_params('learning_rate') = 0.03;
                ultra_simple_params('min_data_in_leaf') = 60;
                ultra_simple_params('lambda_l1') = 0;
                ultra_simple_params('lambda_l2') = 0;
                ultra_simple_params('feature_fraction') = 1.0;
                ultra_simple_params('bagging_fraction') = 1.0;
                ultra_simple_params('bagging_freq') = 0;
                
                try
                    [best_booster, best_iteration] = train_with_early_stop(pv_train, ultra_simple_params, 50, pv_val, 8);
                    fprintf('第 %d 轮：使用极简参数训练完成，迭代次数: %d\n', loop_idx, best_iteration);
                catch ME4
                    fprintf('第 %d 轮：所有方法都失败，使用最简单模型: %s\n', loop_idx, ME4.message);
                    error('模型训练完全失败');
                end
            end
        end
    end
    
    %% 计算预测概率
    try
        train_prob = predictMatrix(best_booster, single(p_train), best_iteration);
        val_prob = predictMatrix(best_booster, single(p_val), best_iteration);
        test_prob = predictMatrix(best_booster, single(p_test), best_iteration);
        
        % 确保概率值有效
        train_prob = max(1e-15, min(1-1e-15, train_prob));
        val_prob = max(1e-15, min(1-1e-15, val_prob));
        test_prob = max(1e-15, min(1-1e-15, test_prob));
    catch ME
        fprintf('第 %d 轮：预测出错: %s\n', loop_idx, ME.message);
        try
            train_prob = predictMatrix(best_booster, single(p_train));
            val_prob = predictMatrix(best_booster, single(p_val));
            test_prob = predictMatrix(best_booster, single(p_test));
            
            % 确保概率值有效
            train_prob = max(1e-15, min(1-1e-15, train_prob));
            val_prob = max(1e-15, min(1-1e-15, val_prob));
            test_prob = max(1e-15, min(1-1e-15, test_prob));
            
            best_iteration = 1;
        catch ME2
            fprintf('第 %d 轮：默认预测也出错: %s\n', loop_idx, ME2.message);
            error('预测失败');
        end
    end
    
    %% 6. SHAP分析（使用shapKernel计算）
    fprintf('正在计算SHAP值...\n');
    num_background = min(2000, size(p_train,1));
    X_background = datasample(p_train, num_background);
    X_explain = p_test(1:min(2000, size(p_test,1)), :);

    % 使用shapKernel计算SHAP值
    shap_values = shapKernel(best_booster, X_background, X_explain);

    % 全局特征重要性
    feature_importance = mean(abs(shap_values), 1);
    
    % 转换为百分比
    feature_importance_percent = 100 * feature_importance / sum(feature_importance);
    
    % 筛选出重要性大于1%的特征
    valid_features = feature_importance_percent >= 1;
    valid_feature_importance = feature_importance(valid_features);
    valid_feature_importance_percent = feature_importance_percent(valid_features);
    valid_feature_names = feature_names(valid_features);
    [~, valid_sorted_idx] = sort(valid_feature_importance, 'descend');
    
    % 检查是否有有效特征
    actual_feature_count = length(valid_feature_importance);
    if actual_feature_count == 0
        fprintf('警告：没有特征的重要性超过1%%，将显示所有特征\n');
        valid_features = ones(size(feature_importance_percent)) > 0;
        valid_feature_importance = feature_importance;
        valid_feature_importance_percent = feature_importance_percent;
        valid_feature_names = feature_names;
        [~, valid_sorted_idx] = sort(valid_feature_importance, 'descend');
        actual_feature_count = length(valid_feature_importance);
    end

    % 使用英文特征名称绘制图表
    figure('Name', 'SHAP Feature Importance (Percentage) - Vertical', 'Position', [200, 200, 3000, 800]);  % 增大尺寸至 3000
    
    % 使用 bar 创建竖向柱状图
    h = bar(valid_feature_importance_percent(valid_sorted_idx), 'FaceColor', [0.1  0.5  0.9], 'EdgeColor', 'k');
    
    % 获取当前坐标轴
    ax = gca;
    
    % 设置 y 轴标签
    ylabel('Mean |SHAP Value| (%)', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 设置标题
    title(sprintf('SHAP Feature Importance (Percentage) - Vertical for %s', output_name), 'FontSize', 14, 'FontWeight', 'bold');
    
    % 添加网格
    grid on;
    grid minor;
    
    % 设置坐标轴样式
    set(ax, 'Box', 'on');
    set(ax, 'LineWidth', 1.5);
    
    % 设置坐标轴范围
    top_n = length(valid_feature_importance_percent(valid_sorted_idx));
    xlim([0, top_n + 0.5]);
    ylim([0, max(valid_feature_importance_percent(valid_sorted_idx)) * 1.1]);  % 减小上边距
    
    % 添加特征名称标签（贴近x轴）
    for i = 1:length(valid_feature_importance_percent(valid_sorted_idx))
        % 在柱子底部显示特征名称（贴近x轴）
        text(i, -max(valid_feature_importance_percent(valid_sorted_idx)) * 0.01, valid_feature_names{valid_sorted_idx(i)}, ...
            'HorizontalAlignment', 'center', 'FontSize', 5, 'Color', 'k', 'VerticalAlignment', 'top');
    end
    
    % 隐藏默认的x轴标签和刻度
    set(ax, 'XTick', []);
    set(ax, 'XTickLabel', []);
    
    % 保存图像
    filename = sprintf('SHAP_Feature_T3_Importance_Loop_%03d.png', loop_idx);
    saveas(gcf, filename);
    fprintf('SHAP特征重要性图已保存为: %s\n', filename);
    hold off;
    
    %%7. SHAP依赖图：前5个特征
    top_5_features = valid_sorted_idx(1:min(5, actual_feature_count));
    % 获取原始特征索引
    original_indices = find(valid_features);
    
    % 选择最重要的特征进行依赖图分析
    if length(top_5_features) > 0
        top_feature_idx = original_indices(top_5_features(1));  % 选择最重要的特征
        feature_name = valid_feature_names{top_5_features(1)};
        feature_data = X_explain(:, top_feature_idx);
        
                % 创建SHAP依赖图
        % 扩大画幅并增加边距，使点远离边界
        figure('Name', strcat('SHAP Dependence Plot - ', feature_name), 'Position', [100, 100, 2000, 1200]);  % 扩大画幅
        
        % 绘制散点图：x轴为特征值，y轴为SHAP值，增加点大小和透明度
        scatter(feature_data, shap_values(:, top_feature_idx), 80, ...  % 增大点大小
            shap_values(:, top_feature_idx), 'filled', 'MarkerFaceAlpha', 0.6, 'MarkerEdgeAlpha', 0.4);  % 增加透明度
        
        % 添加颜色条
        colorbar('Location', 'eastoutside', 'FontSize', 14);
        colormap(jet);
        caxis([-0.06, 0.06]);  % 设置颜色范围
        
        % 设置标题和轴标签
        title(sprintf('%s Impact on %s (SHAP Dependence Plot)', feature_name, output_name), 'FontSize', 18, 'FontWeight', 'bold');
        xlabel('Feature Value', 'FontSize', 16, 'FontWeight', 'bold');
        ylabel(sprintf('SHAP Value (Impact on %s)', output_name), 'FontSize', 16, 'FontWeight', 'bold');
        
        % 添加网格
        grid on;
        
        % 增加点与点的间距（通过调整坐标轴范围实现）
        % 调整坐标轴范围，确保点不会落在边缘，增加30%的边距
        x_range = max(feature_data) - min(feature_data);
        y_range = 0.06 - (-0.06);  % SHAP值范围
        xlim([min(feature_data) - 0.3*x_range, max(feature_data) + 0.3*x_range]);
        ylim([-0.06 - 0.15*y_range, 0.06 + 0.15*y_range]);
        
        % 设置坐标轴样式
        ax = gca;
        set(ax, 'Box', 'on');
        set(ax, 'LineWidth', 2);
        set(ax, 'FontSize', 14);
        
        % 保存图像
        filename = sprintf('SHAP_T3_Dependence_Plot_%s_Loop_%03d.png', feature_name, loop_idx);
        saveas(gcf, filename);
        fprintf('SHAP依赖图已保存为: %s\n', filename);
    end

    % SHAP交互图：前3对特征
    if length(top_5_features) >= 2
        pair_idx = nchoosek(1:length(top_5_features), 2);
        for p = 1:min(3, size(pair_idx, 1))  % 限制交互图数量
            i = pair_idx(p, 1);
            j = pair_idx(p, 2);
            original_idx_i = original_indices(top_5_features(i));
            original_idx_j = original_indices(top_5_features(j));
            feat1_name = valid_feature_names{top_5_features(i)};
            feat2_name = valid_feature_names{top_5_features(j)};

            % 扩大画幅并增加边距
            figure('Position', [400, 400, 1000, 800]);
            scatter(X_explain(:, original_idx_i), X_explain(:, original_idx_j), 30, ...
                    shap_values(:, original_idx_i) + shap_values(:, original_idx_j), ...
                    'filled', 'MarkerFaceAlpha', 0.7);
            colormap('jet');
            cbar = colorbar;
            cbar.Label.String = sprintf('Total Impact on %s', output_name);
            xlabel(feat1_name, 'FontSize', 12);
            ylabel(feat2_name, 'FontSize', 12);
            title(sprintf('SHAP Interaction between %s and %s', feat1_name, feat2_name), 'FontSize', 14);
            grid on;
            box on;
            set(gca, 'FontSize', 10);
            set(gcf, 'Color', 'w');
            
            % 增加边距使点远离边界
            x_data = X_explain(:, original_idx_i);
            y_data = X_explain(:, original_idx_j);
            x_range = max(x_data) - min(x_data);
            y_range = max(y_data) - min(y_data);
            xlim([min(x_data) - 0.15*x_range, max(x_data) + 0.15*x_range]);
            ylim([min(y_data) - 0.15*y_range, max(y_data) + 0.15*y_range]);
            
            % 保存图像
            filename = sprintf('SHAP_T3_Interaction_Plot_%s_%s_Loop_%03d.png', ...
                feat1_name, feat2_name, loop_idx);
            saveas(gcf, filename);
            fprintf('SHAP交互图已保存为: %s\n', filename);
        end
    end
end
toc 
fprintf('所有循环运行结束\n');

%% 辅助函数
function [composite_score, metrics] = evaluateParamsWithCompositeMetric(X_train, y_train, X_val, y_val, numeric_cols, categorical_cols, ps_input, params)
    composite_score = Inf;  % 默认无效
    metrics = struct();
    metrics.train_auc = 0;  % 初始化训练集AUC
    metrics.val_auc = 0;    % 初始化验证集AUC
    
    try
        % 训练集预处理
        if ~isempty(numeric_cols)
            X_train_num = X_train(:, numeric_cols);
            noise_level = 0.02;  % 减少噪声
            X_train_num_noisy = X_train_num + noise_level * randn(size(X_train_num)) .* mean(X_train_num);
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
        
        % 训练二分类模型
        param_map = containers.Map;
        param_map('task') = 'train';
        param_map('objective') = 'binary';
        param_map('metric') = 'binary_logloss,auc';
        if ~isempty(categorical_cols)
            categorical_indices = categorical_cols - 1;
            param_map('categorical_feature') = strjoin(cellstr(num2str(categorical_indices')), ',');
        end
        param_map('verbose') = -1;
        param_map('early_stopping_round') = 10;
        
        % 参数设置
        param_map('num_leaves') = max(6, min(31, round(params.num_leaves)));
        param_map('max_depth') = max(3, min(6, round(params.max_depth)));
        param_map('learning_rate') = max(0.02, min(0.3, params.learning_rate));
        param_map('min_data_in_leaf') = max(20, round(params.min_data_in_leaf));
        param_map('bagging_freq') = max(3, min(7, round(params.bagging_freq)));
        param_map('feature_fraction') = max(0.5, min(1.0, params.feature_fraction));
        param_map('bagging_fraction') = max(0.7, min(1.0, params.bagging_fraction));
        param_map('lambda_l1') = max(0, min(10, params.lambda_l1));
        param_map('lambda_l2') = max(0, min(10, params.lambda_l2));
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
            % 使用自定义早停训练函数
            [booster, best_iteration] = train_with_early_stop(pv_train_fold, param_map, 80, pv_val_fold, 10);
            
            % 计算训练集AUC
            train_prob = predictMatrix(booster, p_train_fold, best_iteration);
            train_prob = max(1e-15, min(1-1e-15, train_prob));
            train_auc = roc_auc_score(logical(t_train_fold), train_prob);
            metrics.train_auc = train_auc;
            
            % 核心逻辑：训练集AUC>0.9直接判为无效
            if train_auc > 0.9 || train_auc < 0.5
                composite_score = Inf;
                val_prob = predictMatrix(booster, p_val_fold, best_iteration);
                val_prob = max(1e-15, min(1-1e-15, val_prob));
                val_auc = roc_auc_score(logical(t_val_fold), val_prob);
                metrics.val_auc = val_auc;
                return;
            end
            
            % 计算其他指标
            val_prob = predictMatrix(booster, p_val_fold, best_iteration);
            val_prob = max(1e-15, min(1-1e-15, val_prob));
            val_auc = roc_auc_score(logical(t_val_fold), val_prob);
            metrics.val_auc = val_auc;
            
            val_logloss = log_loss(logical(t_val_fold), val_prob);
            train_logloss = log_loss(logical(t_train_fold), train_prob);
            overfit_auc = abs(train_auc - val_auc);
            overfit_logloss = abs(train_logloss - val_logloss);
            
            % 综合得分（优化目标：提高AUC，减少过拟合）
            composite_score = 0.4*(1-train_auc) + 0.4*(1-val_auc) + 0.1*overfit_auc + 0.1*val_logloss;
            
            % 存储指标
            metrics.train_logloss = train_logloss;
            metrics.val_logloss = val_logloss;
            metrics.overfit_auc = overfit_auc;
            
        catch ME
            fprintf('模型训练失败: %s\n', ME.message);
            composite_score = Inf;
        end
    catch ME
        fprintf('参数评估失败: %s\n', ME.message);
        composite_score = Inf;
    end
end

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
    booster.addValidationData(valid_data);
    
    for i = 1:num_rounds
        % 更新一轮迭代
        finished = booster.updateOneIter();
        
        % 获取当前迭代的评估结果
        eval_results = booster.getEval();
        
        % 解析验证集AUC值
        if length(eval_results) >= 4
            valid_auc = eval_results(4);  % 验证集AUC
        else
            valid_auc = eval_results(end);
        end
        
        % 计算本次迭代的性能提升
        if i == 1
            improvement = valid_auc;  % 第一轮以绝对值为提升
        else
            improvement = valid_auc - previous_auc;
        end
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
        if mod(i, 10) == 0 || i <= 5 || no_improvement_count >= early_stop_rounds || ...
                (i >= 2 && valid_auc > 0.9 && improvement > 0.1)
            fprintf('[%4d] valid auc: %.6f, best auc: %.6f\n', i, valid_auc, best_score);
        end
        
        % 检查是否应该早停
        % 条件1: 连续多轮没有改善
        if no_improvement_count >= early_stop_rounds
            fprintf('早期停止：连续%d轮没有改善\n', early_stop_rounds);
            break;
        end
        
        % 条件2: 特殊处理：如果AUC突然大幅提升（如从<0.9到>0.98），则使用上一轮模型
        if i >= 2
            if valid_auc > 0.9 && improvement > 0.1
                fprintf('检测到AUC突然大幅提升（提升: %.4f），可能存在过拟合，停止训练\n', improvement);
                stopped_for_overfitting = true;
                % 当检测到过拟合时，使用上一轮（即未过拟合的轮次）作为最佳模型
                if best_iter >= i-1
                    best_iter = i-1;
                end
                break;
            end
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
    y_prob = max(eps, min(1-eps, y_prob));
    n = length(y_true);
    logloss = -sum(y_true .* log(y_prob) + (1 - y_true) .* log(1 - y_prob)) / n;
end

function [fpr, tpr] = roc_curve(y_true, y_score)
    y_true = logical(y_true(:));
    y_score = y_score(:);
    % 确保分数有效
    y_score(isnan(y_score)) = 0.5;
    y_score(isinf(y_score)) = 0.5;
    
    thresholds = unique(y_score);
    thresholds = sort(thresholds, 'descend');
    thresholds = [thresholds; -inf];
    
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
        y_pred = y_score >= thresholds(i);
        TP = sum(y_true & y_pred);
        FP = sum(~y_true & y_pred);
        tpr(i) = TP / P;
        fpr(i) = FP / N;
    end
end

function auc = roc_auc_score(y_true, y_score)
    % 确保输入有效
    y_true = logical(y_true(:));
    y_score = y_score(:);
    
    % 处理无效值
    invalid_idx = isnan(y_score) | isinf(y_score);
    if any(invalid_idx)
        y_score(invalid_idx) = 0.5;
    end
    
    [fpr, tpr] = roc_curve(y_true, y_score);
    auc = trapz(fpr, tpr);
    
    % 确保AUC值有效
    if isnan(auc) || isinf(auc)
        auc = 0.5;
    end
end

function threshold = find_optimal_threshold(y_true, y_score)
    % 寻找最优F1分数的阈值
    thresholds = 0.1:0.01:0.9;
    best_f1 = 0;
    threshold = 0.5;
    
    for i = 1:length(thresholds)
        th = thresholds(i);
        y_pred = y_score >= th;
        [~, ~, ~, f1] = calculate_classification_metrics(y_true, y_pred);
        
        if f1 > best_f1
            best_f1 = f1;
            threshold = th;
        end
    end
end

% 贝叶斯优化辅助函数
function next_params = suggestNextParams(evaluated_params, evaluated_scores, param_bounds)
    n_evals = length(evaluated_scores);
    if n_evals == 0
        % 初始参数
        next_params = struct();
        next_params.num_leaves = randi([6, 31]);
        next_params.max_depth = randi([3, 6]);
        next_params.min_data_in_leaf = randi([20, 200]);
        next_params.learning_rate = 0.02 + rand() * 0.28;
        next_params.feature_fraction = 0.5 + rand() * 0.5;
        next_params.bagging_fraction = 0.7 + rand() * 0.3;
        next_params.bagging_freq = randi([3, 7]);
        next_params.lambda_l1 = rand() * 10;
        next_params.lambda_l2 = rand() * 10;
        next_params.scale_pos_weight = 1.0;
        return;
    end
    
    best_score = min(evaluated_scores);
    param_matrix = zeros(n_evals, 9);
    for i = 1:n_evals
        if isstruct(evaluated_params{i})
            param_matrix(i, 1) = evaluated_params{i}.num_leaves;
            param_matrix(i, 2) = evaluated_params{i}.max_depth;
            param_matrix(i, 3) = evaluated_params{i}.min_data_in_leaf;
            param_matrix(i, 4) = evaluated_params{i}.learning_rate;
            param_matrix(i, 5) = evaluated_params{i}.feature_fraction;
            param_matrix(i, 6) = evaluated_params{i}.bagging_fraction;
            param_matrix(i, 7) = evaluated_params{i}.bagging_freq;
            param_matrix(i, 8) = evaluated_params{i}.lambda_l1;
            param_matrix(i, 9) = evaluated_params{i}.lambda_l2;
        end
    end
    
    best_ei = -Inf;
    next_params = defaultNextParams();
    n_candidates = 50;
    for i = 1:n_candidates
        candidate = struct();
        candidate.num_leaves = randi([6, 31]);
        candidate.max_depth = randi([3, 6]);
        candidate.min_data_in_leaf = randi([20, 200]);
        candidate.learning_rate = 0.02 + rand() * 0.28;
        candidate.feature_fraction = 0.5 + rand() * 0.5;
        candidate.bagging_fraction = 0.7 + rand() * 0.3;
        candidate.bagging_freq = randi([3, 7]);
        candidate.lambda_l1 = rand() * 10;
        candidate.lambda_l2 = rand() * 10;
        candidate.scale_pos_weight = 1.0;
        
        ei = expectedImprovement(candidate, param_matrix, evaluated_scores, best_score, param_bounds);
        if ei > best_ei
            best_ei = ei;
            next_params = candidate;
        end
    end
end

function params = defaultNextParams()
    params = struct();
    params.num_leaves = 15;
    params.max_depth = 4;
    params.min_data_in_leaf = 50;
    params.learning_rate = 0.1;
    params.feature_fraction = 0.8;
    params.bagging_fraction = 0.8;
    params.bagging_freq = 5;
    params.lambda_l1 = 2;
    params.lambda_l2 = 2;
    params.scale_pos_weight = 1.0;
end

function ei = expectedImprovement(candidate, param_matrix, scores, best_score, param_bounds)
    % 检查输入有效性
    if isempty(param_matrix) || all(all(param_matrix == 0))
        ei = rand();
        return;
    end
    
    candidate_vec = [candidate.num_leaves, candidate.max_depth, candidate.min_data_in_leaf, ...
                     candidate.learning_rate, candidate.feature_fraction, candidate.bagging_fraction, ...
                     candidate.bagging_freq, candidate.lambda_l1, candidate.lambda_l2];
    
    n_samples = size(param_matrix, 1);
    distances = zeros(n_samples, 1);
    valid_count = 0;
    
    for i = 1:n_samples
        if any(param_matrix(i, :) ~= 0)
            valid_count = valid_count + 1;
            d1 = abs(candidate_vec(1) - param_matrix(i, 1)) / (31 - 6);
            d2 = abs(candidate_vec(2) - param_matrix(i, 2)) / (6 - 3);
            d3 = abs(candidate_vec(3) - param_matrix(i, 3)) / (200 - 20);
            d4 = abs(candidate_vec(4) - param_matrix(i, 4)) / (0.3 - 0.02);
            d5 = abs(candidate_vec(5) - param_matrix(i, 5)) / (1.0 - 0.5);
            d6 = abs(candidate_vec(6) - param_matrix(i, 6)) / (1.0 - 0.7);
            d7 = abs(candidate_vec(7) - param_matrix(i, 7)) / (7 - 3);
            d8 = abs(candidate_vec(8) - param_matrix(i, 8)) / (10 - 0);
            d9 = abs(candidate_vec(9) - param_matrix(i, 9)) / (10 - 0);
            distances(valid_count) = sqrt(d1^2 + d2^2 + d3^2 + d4^2 + d5^2 + d6^2 + d7^2 + d8^2 + d9^2);
        end
    end
    
    if valid_count == 0
        ei = rand();
        return;
    end
    
    distances = distances(1:valid_count);
    scores = scores(1:min(valid_count, length(scores)));
    
    sigma = 0.3;
    weights = exp(-distances.^2 / (2 * sigma^2));
    weights = weights / (sum(weights) + eps);
    predicted_score = sum(weights .* scores);
    improvement = best_score - predicted_score;
    ei = max(improvement, 0);
end