function vis_grpYlabel(plots, title)
pos = get([plots{:}], 'Position');
if iscell(pos), pos = cell2mat(pos); end
axes('Position',[min(pos(:,1)), min(pos(:,2)), min(pos(:,1))*0.2,...
    abs(max(pos(:,2))-min(pos(:,2)))+max(pos(:,4))], 'Visible', 'off');
set(get(gca,'Ylabel'), 'Visible','on', 'String', {title, ''}, ...
    'Interpreter', 'none', 'FontWeight', 'bold', 'FontSize', 12);
end