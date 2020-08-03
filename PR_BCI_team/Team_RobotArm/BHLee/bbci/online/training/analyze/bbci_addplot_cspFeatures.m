%% Typically this function is used which using
%% bbci_bet_analyze
%% with one of the CSP setups (e.g. bbci.setup= 'cspauto')

if ~exist('class_combination', 'var'),
  class_combination= [1 2];
end
if ~exist('opt_grid', 'var') | ~isfield(opt_grid, 'colorOrder'),
  opt_grid.colorOrder= [1 0 0; 0 0 1];
end


figure
clf;
ci1= find(features.y(1,:));
ci2= find(features.y(2,:));
X= squeeze(features.x);
nPat= size(X, 1);
Xfs= getfield(proc_rfisherScore(features), 'x');
for ii= 1:nPat,
  suplot(nPat, ii);
  plot(ci1, movingAverage(X(ii,ci1)', 10, 'method','centered'), ...
       'Color',max(0.6, opt_grid.colorOrder(class_combination(1),:)), ...
       'LineWidth',2);
  hold on;
  plot(ci2, movingAverage(X(ii,ci2)', 10, 'method','centered'), ...
       'Color',max(0.6, opt_grid.colorOrder(class_combination(2),:)), ...
       'LineWidth',2);
  plot(ci1, X(ii,ci1), '.', ...
       'Color',opt_grid.colorOrder(class_combination(1),:));
  plot(ci2, X(ii,ci2), '.', ...
       'Color',opt_grid.colorOrder(class_combination(2),:));
  hold off;
  set(gca, 'XLim', [-1 size(X,2)+2]);
  title(sprintf('{\\bf %s}: %.2f', features.clab{ii}, Xfs(ii)));
end
