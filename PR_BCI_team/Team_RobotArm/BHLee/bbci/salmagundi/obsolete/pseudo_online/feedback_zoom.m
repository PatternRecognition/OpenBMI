%% run feedback_overview first !!
%% xZoom must be defined

clf;
he= plot(time_line', yLim(2)*err_out);
set(he(1), 'lineStyle','none', 'marker','o');
set(he(1), 'color',err_col(1,:), 'markerSize',6, 'lineWidth',2);
set(he(2:3), 'lineStyle','none', 'marker','.');
set(he(2), 'color',err_col(2,:), 'markerSize',12);
set(he(3), 'color',err_col(3,:), 'markerSize',12);
hold on;
hp= plot(time_line', [1.1*comb_out goal dscr_out_ma dtct_out_ma]);
set(hp(1), 'color',[0 0.8 0], 'linewidth',2);
set(hp(2), 'color','k', 'linewidth',2);
set(hp(3), 'color','m', 'linewidth',1);
set(hp(4), 'color',[1 0.7 0], 'linewidth',1);
hold off;
h= line(repmat([test_begin/cnt.fs; test_end/cnt.fs], 1, 6), ...
        repmat([.5 -.5 0.25 -0.25 1 -1], 2, 1));
set(h, 'lineStyle',':', 'color','k');
axis tight
set(gca, 'yLim',yLim, 'xLim', xZoom);
