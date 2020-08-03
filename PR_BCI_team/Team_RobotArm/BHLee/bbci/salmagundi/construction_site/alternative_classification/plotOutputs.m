function plotOutputs(pred_y, true_y, labels, ylm)
% plotOutputs(pred_y, true_y, labels, ylm)
%
% INPUTS:
%  pred_y : cell array of size (1 * #datasets)
%           elements are either memo structure with 'out' field or
%           the outputs as vectors.
%  true_y : cell array of size (1* #datasets).
%           load 'true_y_classification.mat' or
%                'true_y_transfer.mat'
%  labels : cell array of size (1* #datasets)
%           
% EXAMPLE:
%  load true_y_classification
%  load([DATA_DIR 'results/alternative_classification/specCSP.mat']);
%  figure, plotOutputs(memo, true_y, subdir_list);
%

if isnumeric(pred_y)
  pred_y = {pred_y};
end

if isnumeric(true_y)
  true_y = {true_y};
end

if ischar(labels)
  labels = {labels};
end
  

if isstruct(pred_y{1})
  memo = pred_y;
  for i=1:length(pred_y)
    try, pred_y{i}=memo{i}.out; catch, pred_y{i}=[]; end
  end
end


n=[0,cumsum(cell2mat(foreach(@length, pred_y)))];
for i=1:length(true_y),
  if ~isempty(pred_y{i})
    I1=find(true_y{i}<0);
    I2=find(true_y{i}>0);
    hold on;
    plot(n(i)+I1, pred_y{i}(I1), 'x', n(i)+I2, pred_y{i}(I2), 'o');
  end
end

if exist('ylm','var')
  ylim(ylm);
end


I=find(~cell2mat(foreach(@isempty, true_y)));
set(gca,'xtick', n(I)+1);
grid on;

yl=ylim;
for i=1:length(I),
  text(n(I(i)), yl(2), untex(labels{I(i)}), 'rotation', 90,...
       'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
end
