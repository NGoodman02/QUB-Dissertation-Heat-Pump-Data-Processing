

%% Smoothed COP vs Time
% Ensure the time column is correctly formatted (assuming it's in string or numeric format)
time = datetime(FourHourDataIntervalsV2.FormattedTime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss'); % Adjust format as necessary

% Skip the timestamp column and focus on numerical data
parameters = FourHourDataIntervalsV2(:, 2:end-2); % All columns except 'FormattedTime', 'COP', and 'COP_Smoothed'
COP = FourHourDataIntervalsV2.COP; % Use COP as the dependent variable

% Remove rows with missing data
validRows = ~any(ismissing(parameters), 2) & ~ismissing(COP) & ~ismissing(time);
parameters = parameters(validRows, :);
COP = COP(validRows);
time = time(validRows);

% Apply smoothing to COP using a moving average
windowSize = 48;  % Adjust window size for smoothing (e.g., 48 for smoothing over two days if data is hourly)
smoothedCOP = movmean(COP, windowSize);

% Plot smoothed COP against time
figure; % Create a new figure
plot(time, smoothedCOP, '-o', 'DisplayName', 'Smoothed COP'); % Plot smoothed COP with time on the x-axis
title('Smoothed COP vs Time');
xlabel('Time');
ylabel('Smoothed COP');
grid on;

% Optionally format the time axis for better readability
datetick('x', 'yyyy-mm-dd', 'keepticks'); % Adjust this format as necessary for your time data

% Optional: Adjust axis limits if needed for better clarity
xlim([time(1) time(end)]); % Adjust limits to fit the data range





%% Clustering Analysis

% Load data
% Ensure 'parameters' contains numerical data and 'Power_Absorbed' is the dependent variable
parameters = FourHourDataIntervalsV3(:, 2:end-2); % Exclude non-numerical columns
Power_Absorbed = FourHourDataIntervalsV3.Power_Absorbed;

% Remove rows with missing data

validRows = ~any(ismissing(parameters), 2) & ~ismissing(Power_Absorbed);
parameters = parameters(validRows, :);
Power_Absorbed = Power_Absorbed(validRows);

% Normalize the data for clustering

parametersArray = table2array(parameters);
parametersArray = normalize(parametersArray, 'zscore'); % Standardize data (zero mean, unit variance)

% Perform clustering (k-means in this example)

k = 4;
[idx, clusterCenters] = kmeans(parametersArray, k, 'Distance', 'sqeuclidean', 'Replicates', 10);


parameters.Cluster = idx;
parameterNames = parameters.Properties.VariableNames(1:end-1);  

for c = 1:k
    fprintf('Cluster %d Parameters:\n', c);
    clusterData = parameters(idx == c, 1:end-1);  
   
    clusterMeans = mean(clusterData{:,:}, 1);  
    for i = 1:length(parameterNames)
        fprintf('%s: %.4f\n', parameterNames{i}, clusterMeans(i));
    end
    fprintf('\n');
end

[coeff, score] = pca(parametersArray); 
figure;
gscatter(score(:,1), score(:,2), idx);
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('Clusters of Parameters');
legend(arrayfun(@(x) sprintf('Cluster %d', x), 1:k, 'UniformOutput', false));
grid on;


clusters = arrayfun(@(c) mean(Power_Absorbed(idx == c)), 1:k);
disp('Average Power_Absorbed by Cluster:');
for c = 1:k
    fprintf('Cluster %d: %.4f\n', c, clusters(c));
end


%% Power Abs Vs Time 
% 
time = FourHourDataIntervalsV3.FormattedTime;  % Time column
Power_Absorbed = FourHourDataIntervalsV3.Power_Absorbed;  % Power Absorbed column


% Remove rows with missing Power_Absorbed values (ignore days with no values)
validRows = ~ismissing(Power_Absorbed);  % Find non-missing data
time = time(validRows);  % Filter time based on valid rows
Power_Absorbed = Power_Absorbed(validRows);  % Filter Power_Absorbed based on valid rows

% Apply smoothing to Power_Absorbed using a moving average
windowSize = 48;  % Adjust window size for smoothing (e.g., 48 for smoothing over two days if data is hourly)
smoothedPowerAbsorbed = movmean(Power_Absorbed, windowSize);

% Add the smoothed data back to the table (optional)
FourHourDataIntervalsV3.Smoothed_Power_Absorbed = NaN(size(FourHourDataIntervalsV3, 1), 1);  % Initialize the column
FourHourDataIntervalsV3.Smoothed_Power_Absorbed(validRows) = smoothedPowerAbsorbed;  % Insert smoothed values into the original table

% Plot original Power_Absorbed and smoothed Power_Absorbed against time
figure;
plot(time, Power_Absorbed, '-o', 'DisplayName', 'Original Power Absorbed');
hold on;
plot(time, smoothedPowerAbsorbed, '-r', 'LineWidth', 2, 'DisplayName', 'Smoothed Power Absorbed');
xlabel('Time');
ylabel('Power Absorbed (kWh)');
title('Power Absorbed vs Time');
legend('show');
grid on;
;

%%
%CORRELATION ANALYSIS OF COP - 4 HOUR INTERVALS - Final
%Parameters

% Skip the timestamp column and focus on numerical data
parameters = FourHourDataIntervalsV4(:, 2:end-2); % 
COP = FourHourDataIntervalsV4.("COP"); % 
% Remove rows with missing data
validRows = ~any(ismissing(parameters), 2) & ~ismissing(COP);
parameters = parameters(validRows, :);
COP = COP(validRows);

% Perform correlation analysis
correlationMatrix = corr(table2array(parameters), COP, 'Rows', 'complete');

% Display correlation results
parameterNames = parameters.Properties.VariableNames;
[sortedCorrelations, sortedIndices] = sort(abs(correlationMatrix), 'descend');

disp('Correlation of each parameter with COP (sorted):');
for i = 1:length(sortedCorrelations)
    fprintf('%s: %.4f\n', parameterNames{sortedIndices(i)}, sortedCorrelations(i));
end

figure;
bar(sortedCorrelations);
xticks(1:length(sortedCorrelations));
xticklabels(parameterNames(sortedIndices));
xtickangle(45);
title('Correlation of Parameters with COP');
ylabel('Correlation Coefficient');
xlabel('Parameters');
grid on;

%%
%CORRELATION ANALYSIS OF POWER ABSORBED - 4 HOUR INTERVALS - Final
%Parameters

% Skip the timestamp column and focus on numerical data
parameters = FourHourDataIntervalsV4(:, 2:end-2); % 
Power_Absorbed = FourHourDataIntervalsV4.("PowerAbsorbed(kWh)"); % 
% Remove rows with missing data
validRows = ~any(ismissing(parameters), 2) & ~ismissing(Power_Absorbed);
parameters = parameters(validRows, :);
Power_Absorbed = Power_Absorbed(validRows);

% Perform correlation analysis
correlationMatrix = corr(table2array(parameters), Power_Absorbed, 'Rows', 'complete');

% Display correlation results
parameterNames = parameters.Properties.VariableNames;
[sortedCorrelations, sortedIndices] = sort(abs(correlationMatrix), 'descend');

disp('Correlation of each parameter with Power_Absorbed (sorted):');
for i = 1:length(sortedCorrelations)
    fprintf('%s: %.4f\n', parameterNames{sortedIndices(i)}, sortedCorrelations(i));
end

figure;
bar(sortedCorrelations);
xticks(1:length(sortedCorrelations));
xticklabels(parameterNames(sortedIndices));
xtickangle(45);
title('Correlation of Parameters with Power Absorbed');
ylabel('Correlation Coefficient');
xlabel('Parameters');
grid on;


%%
%Scatter Plots of all Important Parameters

% Extract relevant columns (excluding 'FormattedTime' and 'Power_Absorbed')
parameters = FourHourDataIntervalsV4(:, 2:end-1); % Exclude 'FormattedTime' and 'Power_Absorbed'
Power_Absorbed = FourHourDataIntervalsV4.("PowerAbsorbed(kWh)"); % Assuming this is the correct column name

% Remove rows with missing data
validRows = ~any(ismissing(parameters), 2) & ~ismissing(Power_Absorbed);
parameters = parameters(validRows, :);
Power_Absorbed = Power_Absorbed(validRows);

% Loop through all parameters and create a separate scatter plot for each
for i = 1:width(parameters)
    % Create a new figure for each parameter
    figure;
    
    % Get the parameter name and values
    parameterName = parameters.Properties.VariableNames{i};
    parameterValues = table2array(parameters(:, i));
    
    % Scatter plot of the parameter vs Power_Absorbed
    scatter(parameterValues, Power_Absorbed, 'filled');
    
    % Add titles and labels
    title(['Scatter Plot: ' parameterName ' vs Power Absorbed']);
    xlabel(parameterName);
    ylabel('Power Absorbed (kWh)');
    grid on;
end

%%

% Scatter Plots of all Important Parameters w/Line of Best Fit

% Extract relevant columns (excluding 'FormattedTime' and 'Power_Absorbed')
parameters = FourHourDataIntervalsV4(:, 2:end-1); % Exclude 'FormattedTime' and 'Power_Absorbed'
Power_Absorbed = FourHourDataIntervalsV4.("PowerAbsorbed(kWh)"); % Assuming this is the correct column name

% Remove rows with missing data
validRows = ~any(ismissing(parameters), 2) & ~ismissing(Power_Absorbed);
parameters = parameters(validRows, :);
Power_Absorbed = Power_Absorbed(validRows);


for i = 1:width(parameters)

    figure;
    
    parameterName = parameters.Properties.VariableNames{i};
    parameterValues = table2array(parameters(:, i));
    
    scatter(parameterValues, Power_Absorbed, 'filled');
    hold on;
    
    p = polyfit(parameterValues, Power_Absorbed, 1);
    Y_pred = polyval(p, parameterValues); 
    
    plot(parameterValues, Y_pred, 'r-', 'LineWidth', 1.5);
    
    title(['Scatter Plot: ' parameterName ' vs Power Absorbed']);
    xlabel(parameterName);
    ylabel('Power Absorbed (kWh)');
    grid on;
    hold off; % Release the figure for the next plot
end


%% Scatter Plots of all Important Parameters Daily 

% Extract relevant columns (excluding 'FormattedTime' and 'Power_Absorbed')
parameters = DailyDataFixed(:, 2:end); % Exclude 'FormattedTime' and 'Power_Absorbed'
Power_Absorbed = DailyDataFixed.("PowerAbsorbed(kWh)"); % Assuming this is the correct column name

% Remove rows with missing data
validRows = ~any(ismissing(parameters), 2) & ~ismissing(Power_Absorbed);
parameters = parameters(validRows, :);
Power_Absorbed = Power_Absorbed(validRows);


for i = 1:width(parameters)

    figure;
    
    parameterName = parameters.Properties.VariableNames{i};
    parameterValues = table2array(parameters(:, i));

    if strcmp(parameterName, "FirstStage Temp Delta°C")
        parameterValues = parameterValues / 10;
    end
    
    scatter(parameterValues, Power_Absorbed, 'filled');
    hold on;
    
    
    title(['Scatter Plot: ' parameterName ' vs Power Absorbed']);
    xlabel(parameterName);
    ylabel('Power Absorbed (kWh)');
    grid on;
    hold off; % Release the figure for the next plot
end



%%

%Time Series Plots
figure;
plot(FourHourDataIntervalsV4.FormattedTime, Power_Absorbed, 'LineWidth', 1.5);
title('Power Absorbed over Time');
xlabel('Time');
ylabel('Power Absorbed (kWh)');
grid on;




%% Weather Compensation Curve
% Define temperature range
outdoor_temp = 0:0.1:20; % From 0°C to 20°C

% Initialize setpoint temperature array
setpoint_temp = zeros(size(outdoor_temp));

% Apply conditions for different temperature ranges
for i = 1:length(outdoor_temp)
    if outdoor_temp(i) <= 5
        setpoint_temp(i) = 45; % Constant at 45°C for <= 5°C
    elseif outdoor_temp(i) >= 15
        setpoint_temp(i) = 40; % Constant at 40°C for >= 15°C
    elseif outdoor_temp(i) >= 11
        % Linear decrease for outdoor temperatures >= 5°C and <= 11°C
        setpoint_temp(i) = 45 - 0.5 * (outdoor_temp(i) - 5);
    else
        % From 5°C to 11°C, adjust for 42°C setpoint at 11°C
        slope = (42 - 45) / (11 - 5); % Calculate the slope to hit 42°C at 11°C
        setpoint_temp(i) = 45 + slope * (outdoor_temp(i) - 5);
    end
end

% Plot the compensation curve
figure;
plot(outdoor_temp, setpoint_temp, 'b-', 'LineWidth', 2);
hold on;
scatter([5, 11, 15], [45, 42, 40], 'ro', 'filled'); % Mark key transition points

% Labels and title
xlabel('Outdoor Temperature (°C)');
ylabel('Setpoint Temperature (°C)');
title('Weather Compensation Curve');


ylim([38 48]); 
xlim([0, 20]);

grid on;
legend('Compensation Curve', 'Key Points (5°C, 11°C, 15°C)');
hold off;



%% Weather Compensation Curve Scatter Plot

% Extract numerical parameters (excluding timestamp) for the daily interval
parameters_daily = DailyDataFixed(:, 2:end);
PowerAbsorbed_daily = DailyDataFixed.("PowerAbsorbed(kWh)");

% Remove missing data for the daily dataset
validRows_daily = ~any(ismissing(parameters_daily), 2) & ~ismissing(PowerAbsorbed_daily);
parameters_daily = parameters_daily(validRows_daily, :);
PowerAbsorbed_daily = PowerAbsorbed_daily(validRows_daily, :);

% Extract set point and temperature data
Setpoint_daily = DailyDataFixed.("SetPoint °C"); 
Temperature_daily = DailyDataFixed.("External Temperature°C");


min_temp = floor(min(Temperature_daily) * 2) / 2; 
max_temp = ceil(max(Temperature_daily) * 2) / 2;
temp_bins = min_temp:0.5:max_temp; 

avg_power_45 = NaN(length(temp_bins)-1, 1);
avg_power_42 = NaN(length(temp_bins)-1, 1);

for i = 1:length(temp_bins)-1
    temp_range_min = temp_bins(i);
    temp_range_max = temp_bins(i+1);
    
    data_45 = PowerAbsorbed_daily(Setpoint_daily == 45 & Temperature_daily >= ...
        temp_range_min & Temperature_daily < temp_range_max);
    if ~isempty(data_45)
        avg_power_45(i) = mean(data_45); 
    end
    
    data_42 = PowerAbsorbed_daily(Setpoint_daily == 42 & Temperature_daily >= ...
        temp_range_min & Temperature_daily < temp_range_max);
    if ~isempty(data_42)
        avg_power_42(i) = mean(data_42); 
    end
end

temp_bins_interp = temp_bins(1:end-1); % Temperature ranges
valid_45 = ~isnan(avg_power_45);
valid_42 = ~isnan(avg_power_42);

% Plotting both setpoints 45°C and 42°C in the same figure
figure;
hold on;


scatter(temp_bins_interp(valid_45), avg_power_45(valid_45), 'b', 'filled'); 
p_45 = polyfit(temp_bins_interp(valid_45), avg_power_45(valid_45), 1); 
y_45_fit = polyval(p_45, temp_bins_interp); 
plot(temp_bins_interp, y_45_fit, 'b-', 'LineWidth', 2); 

scatter(temp_bins_interp(valid_42), avg_power_42(valid_42), 'r', 'filled'); 
p_42 = polyfit(temp_bins_interp(valid_42), avg_power_42(valid_42), 1);
y_42_fit = polyval(p_42, temp_bins_interp); 
plot(temp_bins_interp, y_42_fit, 'r-', 'LineWidth', 2); 

% Labels and title
xlabel('Temperature Range (°C)');
ylabel('Average Power Consumption (kWh)');
title('Average Power Consumption for Setpoints 45°C and 42°C');
legend('Setpoint 45°C (Scatter)', 'Setpoint 45°C (Fit)', 'Setpoint 42°C (Scatter)', 'Setpoint 42°C (Fit)', 'Location', 'best');
grid on;
hold off;


%% Hypothesis Testing

T_ext = FourHourDataIntervalsV4.("External Temperature°C");
P_consumption = FourHourDataIntervalsV4.("PowerAbsorbed(kWh)");

T_ext = T_ext(validIdx);
P_consumption = P_consumption(validIdx);

[R, P] = corrcoef(T_ext, P_consumption);

fprintf('Pearson Correlation Coefficient: %.4f\n', R(1,2));


p_value_manual = 2 * (1 - tcdf(abs(t_score), df));

alpha = 0.05; 
if P(1,2) < alpha
    fprintf(['Reject the null hypothesis: Ext1ernal air temperature significantly' ...
        ' affects power consumption.\n']);
else
    fprintf('Fail to reject the null hypothesis: No significant relationship found.\n');
end

n = length(T_ext);
t_score = (R(1,2) * sqrt(n - 2)) / sqrt(1 - R(1,2)^2);
df = n - 2;


fprintf('Manually calculated t-score: %.4f\n', t_score);
fprintf('Manually calculated p-value: %.6f\n', p_value_manual);


%%

% Loop through each parameter and perform regression analysis
for i = 1:length(parameterNames)
    % Extract X (independent variable) and Y (Power Absorbed) for both datasets
    x_daily = parameters_daily.(parameterNames{i});
    y_daily = PowerAbsorbed_daily;
    x_4hr = parameters_4hr.(parameterNames{i});
    y_4hr = PowerAbsorbed_4hr;

    % Fit a linear regression model for both datasets using polyfit
    p_daily = polyfit(x_daily, y_daily, 1);
    p_4hr = polyfit(x_4hr, y_4hr, 1);
    
    % Compute fitted values
    y_fit_daily = polyval(p_daily, x_daily);
    y_fit_4hr = polyval(p_4hr, x_4hr);
    
    % Compute R² values
    residuals_daily = y_daily - y_fit_daily;
    residuals_4hr = y_4hr - y_fit_4hr;
    SS_res_daily = sum(residuals_daily.^2);
    SS_tot_daily = sum((y_daily - mean(y_daily)).^2);
    R2_daily = 1 - (SS_res_daily / SS_tot_daily);
    
    SS_res_4hr = sum(residuals_4hr.^2);
    SS_tot_4hr = sum((y_4hr - mean(y_4hr)).^2);
    R2_4hr = 1 - (SS_res_4hr / SS_tot_4hr);
    
    % Regression equations
    equationStr_daily = sprintf('y = %.2fx + %.2f', p_daily(1), p_daily(2));
    equationStr_4hr = sprintf('y = %.2fx + %.2f', p_4hr(1), p_4hr(2));

    % Create a new figure for the plot
    figure; 
    
    % Plot scatter for daily data with transparency
    scatter(x_daily, y_daily, 20, 'b', 'filled', 'MarkerFaceAlpha', 0.3, 'MarkerEdgeAlpha', 0.3);
    hold on;
    
    % Plot scatter for 4-hour data with transparency
    scatter(x_4hr, y_4hr, 20, 'r', 'filled', 'MarkerFaceAlpha', 0.3, 'MarkerEdgeAlpha', 0.3);
    
    % Plot the regression line for daily data
    plot(x_daily, y_fit_daily, 'b-', 'LineWidth', 1.5);
    
    % Plot the regression line for 4-hour data
    plot(x_4hr, y_fit_4hr, 'r-', 'LineWidth', 1.5);
    
    % Add labels, title, and grid
    xlabel(parameterNames{i});
    ylabel('Power Absorbed (kWh)');
    title(['Power Absorbed (kWh) vs ' parameterNames{i}]);
    grid on;

    % Add R² and equation to the plot outside the scatter area
    % Position the equations and R² values above the plot area
    text(min(x_daily), max(y_daily) + (max(y_daily) * 0.05), sprintf('R² (Daily) = %.2f\n%s', R2_daily, equationStr_daily), ...
        'FontSize', 12, 'Color', 'b', 'FontWeight', 'bold');
    text(min(x_4hr), max(y_4hr) + (max(y_4hr) * 0.05), sprintf('R² (4-Hour) = %.2f\n%s', R2_4hr, equationStr_4hr), ...
        'FontSize', 12, 'Color', 'r', 'FontWeight', 'bold');
    
    % Add legend to the plot
    legend({'Daily Data', '4-Hour Data', ['Fit (Daily): ', equationStr_daily], ['Fit (4-Hour): ', equationStr_4hr]}, 'Location', 'Best');
    
    hold off; % Release the plot hold
end




