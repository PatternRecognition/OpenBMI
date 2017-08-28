function visualization_SSVEP(fig, varargin)
in = opt_cellToStruct(varargin{:});

results = in.results;
nClasses = size(in.results,2);

bar(fig,1:nClasses,results);
set(fig, 'xticklabel', in.marker(:,2));
labels = arrayfun(@(value) num2str(value,'%2.2f'),results,'UniformOutput',false);
text(1:nClasses,results,labels,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom', 'Parent', fig);
ylim(fig, in.lim);drawnow;
end