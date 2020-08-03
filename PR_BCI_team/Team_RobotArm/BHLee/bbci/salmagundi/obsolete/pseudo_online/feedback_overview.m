ma= feedback_opt.integrate;

if exist('dtct','var'),
  recalc_combination;
  dtct_out_ma= movingAverageCausal(dtct_out, ma);
end

dscr_out_ma= movingAverageCausal(dscr_out, ma);

inter= floor(linspace(0, test_len, nFigs*nPlots+1))';
xLim= time_line([inter(1:end-1)+1 inter(2:end)]);
yLim= [-1.4 1.4];

if ~exist('dtct','var'),

for ig= 1:nFigs,
  figure(ig);
  clf;
  for ip= 1:nPlots,
    ii= (ig-1)*nPlots+ip;
    ha(ip)= subplotxl(nPlots, 1, ip, [0.05 0.06 0.02], [0.04 0 0.02]);
    hp= plot(time_line', [goal dscr_out_ma]);
    set(hp(1), 'color','k', 'linewidth',2);
    set(hp(2), 'color','m', 'linewidth',1);
    h= line(repmat([test_begin/cnt.fs; test_end/cnt.fs], 1, 6), ...
            repmat([.5 -.5 0.25 -0.25 1 -1], 2, 1));
    set(h, 'lineStyle',':', 'color','k');
%    axis tight
    set(gca, 'yLim',yLim, 'xLim', xLim(ii,:));
  end
end

else
  
err_out= mark_errors('init', goal);
err_out= mark_errors(comb_out, goal, err_out, feedback_opt.fs);
err_col= [1 0 0; 1 0 0; 0 0 1];
fprintf('[FN, false cfy, FP] %d / %d / %d out of %d trials\n', ...
        sum(~isnan(err_out)) - sum(~isnan(diff(err_out))), sum(abs(goal)));

for ig= 1:nFigs,
  figure(ig);
  clf;
  for ip= 1:nPlots,
    ii= (ig-1)*nPlots+ip;
    ha(ip)= subplotxl(nPlots, 1, ip, [0.05 0.06 0.02], [0.04 0 0.02]);
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
%    axis tight
    set(gca, 'yLim',yLim, 'xLim', xLim(ii,:));
  end
end

end  %% if-else isempty(dtct)

figure(1)
