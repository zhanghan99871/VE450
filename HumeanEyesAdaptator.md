# HumanEyesAdaptator.py 详细说明

## 1. 定义 `RawImage` 类

`RawImage` 类用于加载和处理图像。该类有两个主要方法：

- `loadRGB(self, png_file)`：加载给定路径的PNG图像文件，将其转换为RGB格式并存储在实例变量中。如果文件不存在或读取失败，会抛出相应的错误。
- `convert_rgb_to_lab_luminance(self)`：将加载的RGB图像转换为LAB色彩空间，并提取L通道（亮度），存储在实例变量中。

## 2. 定义 `HumanEyesAdaptator` 类

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

## 数据拟合及存储

### 数据集

数据集包含初始图像和多个调整后的图像，分为高亮度和低亮度两组：

#### 高亮度数据集
1. VW216 数据集：初始亮度 6809.47 cd/m²
2. VW310 数据集：初始亮度 2654.74 cd/m²
3. VW310-PL 数据集：初始亮度 1744.43 cd/m²
4. VW316 数据集：初始亮度 2124.45 cd/m²
5. VW323 数据集：初始亮度 2381.67 cd/m²
6. VW326 数据集：初始亮度 9001.23 cd/m²
7. VW331 数据集：初始亮度 15241.2 cd/m²

#### 低亮度数据集
1. VW316-TLB 数据集：初始亮度 25.1441 cd/m²
2. VW323-TL 数据集：初始亮度 49.3145 cd/m²

### 拟合过程

#### 拟合公式

使用伽马函数进行拟合：

$$
\text{gamma\_function}(X, k, b, c, X_\text{Ave}) = 100 \cdot \log_{10}(1 + 9 \cdot \left( \frac{X}{X_{\text{Max}}} \right)^{k \cdot \log_{10}(1 + c \cdot X_\text{Ave}) + b})
$$

其中：
- \(X\) 是输入亮度值
- \(X_{\text{Max}}\) 是初始图像亮度的最大值
- \(X_Ave\) 是调整后图像的平均亮度
- \(k\)、\(b\) 和 \(c\) 是拟合参数

### 拟合方法

1. 对每个数据集中的初始图像提取亮度值 \(X\)。
2. 对每个调整后的图像，提取亮度值 \(Y\) 并计算平均亮度 \(X_Ave\)。
3. 使用 `curve_fit` 方法拟合伽马函数，获取最佳拟合参数 \(k\)、\(b\) 和 \(c\)。
4. 计算拟合的 \(R^2\) 值和色差 \(\Delta E\)。

### 参数初始猜测和边界

初始猜测值为 \([-1, 1, 1]\)，参数边界为 \([-20, -20, 0.1]\) 到 \([20, 20, 10]\)。

### 结果存储

#### 结果目录

所有结果存储在以下目录：
- 高亮度数据集结果存储在 `comparison_images_high_luminance` 目录。
- 低亮度数据集结果存储在 `comparison_images_low_luminance` 目录。
- 广义模型结果存储在 `generalized_model` 目录。

#### 存储内容

1. **对比图像**：存储在 `comparison_images_{dataset_name}` 目录下，文件名为 `comparison_{i+1}.png`。
2. **拟合结果可视化**：每个数据集生成的拟合结果可视化图表存储在 `r2_and_delta_e_curve.png` 文件中。
3. **参数与亮度关系**：参数 \(k\)、\(b\) 和 \(c\) 与亮度关系的可视化图表存储在 `param_vs_luminance.png` 文件中。
4. **广义模型应用结果**：存储在 `adjusted_with_generalized_model` 目录下的各个子目录中。
5. **最佳拟合结果可视化**：存储在 `best_fit_results.png` 文件中。

## 数据集和算法的对应关系

- 数据集是通过 SPEOS 软件获得的。
- 数据集包含 9 个模型，每个模型都有不同的初始亮度。
- 使用 SPEOS 的本地自适应功能，调整平均亮度（输入到 SPEOS 中）。
- 从 SPEOS 的输出中获得了 10 到 20 组不同平均亮度下的 PNG 图片。
- 使用这些图片进行伽马拟合，为每张图片得到不同的参数。
- 该算法通过 SPEOS 输出的不同平均亮度下的 PNG 图片进行伽马拟合，提取并拟合亮度信息，计算并存储拟合参数，并进行结果的可视化和保存。
