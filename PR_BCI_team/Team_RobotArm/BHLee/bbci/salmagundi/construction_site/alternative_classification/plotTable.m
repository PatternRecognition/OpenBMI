function plotTable(table, names, labels, varargin)
% plotTable(table, names, labels, varargin)
%
% Inputs
%  <opt>
%   .mfun   : ('nanmean')
%   .msign  : (1)
%
% Examples:
% plotTable(table, names, subdir_list);
% plotTable(btable, names, subdir_list, 'msign', -1);

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'fontsize', 16,...
                        'mfun', 'nanmean',...
                        'msign', 1);

nData = length(labels);

[M, I]=sort(feval(opt.mfun, opt.msign*table));
M = M*opt.msign;

[s, J]=sort(opt.msign*table(:,I(1)));

hold on;
col= hsv2rgb([(0.5:nData)'/nData ones(nData,1)*[1 0.85]]);
set(gca,'colororder',col, 'fontsize', opt.fontsize);
plot(table(J,I), 1:length(names), '-o', 'linewidth', 2);
hold on;
plot(M, 1:length(names));
set(gca,'ydir','reverse','ytick',1:length(names),'yticklabel',names(I));
h=get(gca,'children');
ms = {'o', 'x', '^'};
for i=2:length(h)
  set(h(i),'marker', ms{mod(i,length(ms))+1});
end
set(h(1),'color',[.7,.7,.7],'marker', 'none', 'linewidth', 4, 'linestyle', '--');
hold off;
set(gcf,'position', [2, 7, 1022, 683]);
set(gcf,'paperpositionmode','auto');
set(gcf,'paperorientation','landscape');
legend(untex([labels(J);{opt.mfun}]),-1);
set(gca,'position',[0.2, 0.11, 0.8, 0.815]);









