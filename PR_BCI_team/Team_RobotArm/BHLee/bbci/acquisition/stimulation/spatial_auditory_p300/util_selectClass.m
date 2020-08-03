function varargout = util_selectClass(clsOut, varargin)
%
% Synopsis: This function selects the most probable target from a set of
% classifier outputs. Classifier outputs should be sorted grouped according
% to their labels.
%
% use:
%    [class, prob, sumVec] = util_selectClass(clsOut, opt);
%
% INPUT
%    clsOut            Array with the classifier outpus with size
%                      [iterations x classes]. Each column can contain only
%                      classifier outputs from a single class.
%    OPT
%     .tarSgin         String ('Neg'/'Pos') that sets on which side of the
%                      separating hyperplane the targets lie. ['Neg' =
%                      default]
%     .mapping         Array mapping the columns of clsOut to the
%                      respective classnumber. If empty [=default], the
%                      column index is assumed as classlabel. The length of
%                      mapping must be equal to the number of columns in
%                      clsOut.
%     .degrees         Array with the degree angle for every class. Far
%                      right is 0 degrees, front is 90 degrees. This is
%                      used to calculate the sum of the vectors (sumVec).
%     .sumVec          Boolean. If true, the sum of vectors from the
%                      different classes is computed and returned.
%
% OUTPUT
%    class             Scalar. Returns the classlabel of the most likely target
%                      (columnumber in clsOut, or converted with 'mapping')
%    prob              Scalar. Returns the average of the classifier outputs. To
%                      get true probabilities, some scaling should be done
%                      before.
%    sumVec            Array. Returns the X and Y components of the sum of
%                      the vectors. Intended to be used for navigation
%                      purposes.
%
%
% Martijn Schreuder, 11/08/2009

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'tarSign', 'Neg', ... % 'Pos'/'Neg'. Default: 'Neg'
                  'mapping', [], ...    % specify the class belonging to a column.
                  'degrees', [90:-45:-225], ...
                  'sumVec', 1, ...      % calculate the sum of the vectors
                  'inputProb',0);       % are the values in clsOut probabilisitic
          
if opt.sumVec && isempty(opt.degrees),
    error('Need the angle of speakers for calculation of the vector sum');
end
if ~isempty(opt.mapping) && (length(opt.mapping) ~= size(clsOut, 2)),
    error('If mapping is set, it''s length should be equal to the number of columns in clsOut');
end

[rounds nrClasses] = size(clsOut);

if ~opt.inputProb,
  clsOut = nanmedian(clsOut, 1);
%   clsOut = nanmean(clsOut, 1)
  switch opt.tarSign,
      case 'Neg'
          [dum class] = min(clsOut);
          [tmpCls order] = sort(clsOut, 2, 'ascend');
          prob = tmpCls(2)-tmpCls(1);
      case 'Pos'
          [prob class] = max(clsOut);
  %         tmpCls = sort(clsOut, 2, 'descend');
  %         prob = abs(tmpCls(1)-tmpCls(2));        
  end
else
  %% MICHAEL ENTER CODE HERE THAT RETURNS A CLASS LABEL AND PROB VALUE
end

if opt.sumVec,
    if strcmp(opt.tarSign, 'Neg'),
        clsOut = clsOut*-1; %Adjust for negative sign of cls output
    end
    sumVec = zeros([size(clsOut,2),2]);
    for i= 1:size(clsOut, 2),
        sumVec(i,1) = clsOut(i)*sind(opt.degrees(i)); %front-back
        sumVec(i,2) = clsOut(i)*cosd(opt.degrees(i)); %left-right
    end
    sumVec = nansum(sumVec,1);
end

if ~isempty(opt.mapping),
    class = opt.mapping(class);
end

varargout{1} = class;
varargout{2} = prob;
varargout{3} = sumVec;
if nargout == 4,
  varargout{4} = order;
end
end
