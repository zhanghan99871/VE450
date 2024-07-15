# HumanEyesAdaptator.py详细说明

## 1. 定义 `RawImage` 类

`RawImage` 类用于加载和处理图像。该类有两个主要方法：

- `loadRGB(self, png_file)`：加载给定路径的PNG图像文件，将其转换为RGB格式并存储在实例变量中。如果文件不存在或读取失败，会抛出相应的错误。
- `convert_rgb_to_lab_luminance(self)`：将加载的RGB图像转换为LAB色彩空间，并提取L通道（亮度），存储在实例变量中。

##23. 定义 `HumanEyesAdaptator` 类

### 初始化和数据加载

`HumanEyesAdaptator` 类用于处理图像亮度数据并进行拟合。初始化时，它会加载初始图像的亮度信息和调整后的图像文件列表，并计算每个调整后图像的平均亮度。

### 提取亮度信息和计算平均亮度

`extract_luminance_from_png(self, png_file)` 方法用于从图像中提取亮度信息，返回亮度矩阵。

`calculate_X_Ave_values(self)` 方法用于计算每个调整后图像的平均亮度值，并将其存储在列表中。

### 定义拟合函数

定义了一个伽马函数 `gamma_function(self, X, k, b, c, X_Ave, epsilon=1e-10, max_val=1e3)` 用于亮度调整。该函数使用输入的参数 `k`、`b` 和 `c` 以及亮度值 `X` 和平均亮度 `X_Ave` 来计算伽马校正后的亮度值。函数中加入了 `epsilon` 和 `max_val` 以防止数值计算中的溢出和零除错误。

### 拟合函数和结果可视化

`fit(self)` 方法对每个调整后的图像进行拟合，并记录拟合参数 `k`、`b` 和 `c` 的值，以及拟合的优度指标R²和色差ΔE。方法提供初始猜测和参数边界，通过最小二乘法进行拟合。

`visualize_fit(self, k_values, b_values, c_values, r2_scores, delta_Es, output_file, r2_avg)` 方法用于可视化拟合结果，包括R²和ΔE的分布，以及平均参数值。图表将R²和ΔE的变化趋势展示出来，并在图表下方标注平均参数值和R²。

### 保存对比图像

`save_comparison_images(self, output_dir, k_values, b_values, c_values, sample_luminance_values, r2_scores, delta_Es)` 方法用于保存原始图像和调整后图像的对比图像，并在图像上添加相关信息（如亮度值、R²和ΔE）。它首先确保输出目录存在，然后对每个调整后的图像进行处理，将其亮度调整后与原始图像并排展示，并保存到指定目录。

## 3. 对所有数据集进行拟合和可视化

### 拟合所有数据集

`fit_on_all_data_sets(data_sets, fit_func, output_base_dir)` 方法对所有提供的数据集进行拟合，并保存每个数据集的对比图像和拟合结果的可视化。方法依次处理每个数据集，计算并保存拟合参数和相关的亮度值。

### 可视化参数与亮度的关系

`visualize_params(all_params, luminance_values, output_file)` 方法用于可视化拟合参数 `k`、`b` 和 `c` 与亮度的关系。方法计算每个唯一亮度值对应的平均参数值，并生成散点图展示参数值与亮度的关系。

### 拟合参数与亮度的线性关系

`fit_relationships(all_params, luminance_values)` 方法拟合参数 `k`、`b` 和 `c` 与亮度的线性关系。通过线性回归，计算每个参数与亮度的线性关系模型（即斜率和截距）。

### 应用广义模型并可视化预测结果

`apply_generalized_model(data_sets, k_params, b_params, c_params, output_base_dir)` 方法应用广义模型来调整图像亮度，并保存对比结果。该方法使用之前计算出的线性模型参数，调整每个图像的亮度，并计算预测结果的R²和ΔE。

### 可视化最佳拟合结果

`visualize_best_fit_results(all_r2_scores, all_delta_Es, output_file, k_params, b_params, c_params, r2_avg)` 方法可视化最佳拟合结果，包括R²和ΔE的变化趋势。图表展示了使用广义模型调整后的图像的R²和ΔE值，并在图表下方标注拟合的线性模型参数和平均R²。
