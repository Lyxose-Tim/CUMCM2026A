% plot_all.m —— 读取 ../exports/*.csv 出图（MATLAB 版，对应 plots.py）。
% 用法：在 matlab/ 目录下运行 plot_all，或 run('matlab/plot_all.m')。
% 数据由 Python 端 run_all 生成到 exports/。

here = fileparts(mfilename('fullpath'));
exdir = fullfile(here, '..', 'exports');
figdir = fullfile(here, '..', 'figs');
if ~exist(figdir, 'dir'); mkdir(figdir); end

%% 图2 环境驱动
f = fullfile(exdir, 'env_curve.csv');
if isfile(f)
    D = readmatrix(f, 'NumHeaderLines', 1);
    t = D(:,1); Tair = D(:,2); Cenv = D(:,3);
    fig = figure('Visible','off','Position',[100 100 1000 380]);
    subplot(1,2,1); plot(t, Tair, '-','LineWidth',1.2); hold on;
    xline(14400,'r--'); xlabel('t (s)'); ylabel('T_{air} (\circC)');
    title('环境温度：线性插值 + 常值外推'); grid on;
    subplot(1,2,2); plot(t, Cenv, '-','LineWidth',1.2); hold on;
    xline(14400,'r--'); xlabel('t (s)'); ylabel('C_{env} (kg/kg)');
    title('环境含水率'); grid on;
    saveas(fig, fullfile(figdir,'fig2_env_matlab.png')); close(fig);
end

%% 图3 半径
f = fullfile(exdir, 'radius_curve.csv');
if isfile(f)
    D = readmatrix(f, 'NumHeaderLines', 1);
    fig = figure('Visible','off');
    plot(D(:,1)/3600, D(:,2), '.-'); hold on; yline(1.2,'g:');
    xlabel('t (h)'); ylabel('R (cm)'); title('药材半径 R(t)'); grid on;
    saveas(fig, fullfile(figdir,'fig3_radius_matlab.png')); close(fig);
end

%% 图8 收敛（t* vs N，界面对照）
f = fullfile(exdir, 'convergence.csv');
if isfile(f)
    T = readtable(f);
    fig = figure('Visible','off');
    hold on; g = findgroups(T.interface); u = unique(T.interface);
    for k = 1:numel(u)
        m = strcmp(T.interface, u{k});
        plot(T.N(m), T.t_star_h(m), 'o-', 'DisplayName', u{k});
    end
    xlabel('N'); ylabel('t^* (h)'); legend; grid on;
    title('t^* 网格收敛（界面系数对照）');
    saveas(fig, fullfile(figdir,'fig8_convergence_matlab.png')); close(fig);
end

disp('MATLAB 出图完成，见 figs/。');
