% 读取corn.mat文件
mat_data = load('corn.mat');
% http://www.eigenvector.com/data/Corn/index.html The wavelength range is 1100-2498nm at 2 nm intervals (700 channels).
labels = ["m5spec", "mp5spec", "mp6spec"];
y = ["moisture", "oil", "protein", "starch"];

for i = 1:length(labels)
    for j = 1:length(y)
        data = mat_data.(labels{i}).data;

        ys = mat_data.propvals.data;

        t = ys(:,j);
        % 按列合并数据和属性值
        a = [data, t];
        % 保存为txt文件
        filename = sprintf('%s_%s.txt', labels{i}, y{j});
        dlmwrite(filename, a, 'delimiter', ',', 'precision', '%.6f');
    end
end