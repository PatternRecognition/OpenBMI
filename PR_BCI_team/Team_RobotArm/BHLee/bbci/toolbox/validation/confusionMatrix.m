function [cm, out2]= confusionMatrix(label, out, varargin)
%[cm, cms]= calcConfusionMatrix(label, out, <opt>)
%
% IN   label  - true class labels, can also be a data structure (like epo)
%               including label field '.y'
%      out    - classifier output (as given, e.g., by the third output 
%               argument of xvalidation)
%      opt 
%       .mode        - {'count', 'normalized', 'mean'}
%       .loss_matrix - loss matrix for penelizing misclassifications
%
% OUT  cm   - confusion matrix, cm(t,e) refers to samples of
%             true class #t that were classified as class #e,
%             depending on opt.mode, this can be counts, percentage ('mean'),
%             or classwise percentage ('normalized')
%      cms  - confusion matrix std, ms(t,e) is the std of the % of trials
%             of true class #t that were classified as class #e,
%             TODO: refers only to mode 'mean'!
%
% SEE  xvalidation, out2label

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'mode', 'count', ...
                  'loss_matrix', []);

%% convert continuous output to estimated classes
est= out2label(out);

if isstruct(label), label= label.y; end

nClasses= size(label, 1);
mc= zeros(nClasses, nClasses);
cms= zeros(nClasses, nClasses);

for tt= 1:nClasses,
  iC= find(label(tt,:));
  for ee= 1:nClasses,
    mc(tt,ee)= sum(sum(est(:,iC)==ee));
    mme= mean(est(:,iC)==ee, 2);
    me(tt,ee)= mean(100*mme);
    cms(tt,ee)= std(100*mme);
  end
end

switch(lower(opt.mode)),
 case 'count',
  cm= mc;
 case 'normalized',
  scm= sum(mc,2);
  scm(find(scm==0))= 1;
  cm= diag(1./scm)*mc;
 case 'mean',
  cm= cmp;
 otherwise,
  error('mode not known');
end

if ~isempty(opt.loss_matrix)
  cm= (cm.*loss) ./ repmat(sum(cm,2), 1, nClasses);
end

if nargout>2,
  out2= cms;
end
