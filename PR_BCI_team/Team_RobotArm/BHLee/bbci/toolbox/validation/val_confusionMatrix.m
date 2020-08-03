function [cm, out2, out]= val_confusionMatrix(label, out, varargin)
%[cm, cms]= val_calcConfusionMatrix(label, out, <opt>)
%
% IN   label  - true class labels, can also be a data structure (like epo)
%               including label field '.y'
%      out    - classifier output (as given, e.g., by the third output 
%               argument of xvalidation)
%      opt 
%       .mode        - {'count', 'normalized', 'mean'}
%       .loss_matrix - loss matrix for penelizing misclassifications
%       .bias_correction - calculate a new bias in order to equalize
%                      the true detections for each class, 
%                      {'on', 'off'}, default is 'off'
%                      the choice of the new bias depends on opt.mode
%                      NOTE: bias may be suboptimal (making an unneccessary
%                      error in order to get more equalized true
%                      detections).
%       .show        - 0 or 1, if non-zero then pretty-print the confusion
%                      matrix with row and column labels. Default is 0,
%                      unless no output argument is given.
%       .fid         - File handle for pretty print. Default: 1 (terminal)
%       .format      - specifies the output format, used by fprintf. The
%                      default depends on opt.mode.
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

% bb ida.first.fhg.de

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'mode', 'count', ...
                  'loss_matrix', [], ...
                  'bias_correction', 'off', ...
                  'show', nargout==0, ...
                  'format', [], ...
                  'fid', 1);

if isstruct(label), label= label.y; end
nClasses= size(label, 1);

switch(opt.bias_correction),
 case {1, 'on'},
  if nClasses~=2,
    error('so far this bias correction only implemented for 2 classes');
  end
  NN= sum(label, 2)*size(out,3);
  oo= permute(out, [3 2 1]);
  [so,si]= sort([1 -1]*label);
  oo= oo(:,si);
  ll= ones(size(oo,1),1) * ([1 2]*label(:,si));
  [so,si]= sort(oo(:));
  lo= ll(si);
  tp1= cumsum(lo==1);
  tp2= NN(2)-cumsum(lo==2);
  if ~isempty(strmatch(opt.mode, {'normalized','mean'})),
    tp1= tp1/NN(1);
    tp2= tp2/NN(2);
  end
  [mm,mi]= min(abs(tp1-tp2));
  %% plot([tp1 tp2]); hold on; plot(mi, tp1(mi), 'ro'); hold off
  new_bias= mean(so([mi mi+1]));
  out= out - new_bias;
 case {0, 'off'},
 otherwise,
  error(sprintf('bias correction policy <%s> unknown', opt.bias_correction));
end

%% convert continuous output to estimated classes
est= out2label(out);

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
  fmt= '%6d';
 case 'normalized',
  scm= sum(mc,2);
  scm(find(scm==0))= 1;
  cm= diag(1./scm)*mc;
  fmt= '%6.3f';
 case 'mean',
  cm= me;
  fmt= '%6.2f';
 otherwise,
  error('mode not known');
end
if isempty(opt.format),
  opt.format= fmt;
end

if ~isempty(opt.loss_matrix)
  cm= (cm.*loss) ./ repmat(sum(cm,2), 1, nClasses);
end

if nargout>2,
  out2= cms;
end

if opt.show,
  %% do layout correctly also for nClasses>=10
  ll= 1+floor(log10(nClasses));
  fprintf(opt.fid, repmat(' ', [1 7+ll]));
  fprintf(opt.fid, ['  est #%0' int2str(ll) 'd'], 1:nClasses);
  fprintf(opt.fid, '\n');
  fmt= [repmat(' ',[1 ll+1]) opt.format];
  for tt= 1:nClasses,
    fprintf(opt.fid, ['true #%0' int2str(ll) 'd:'], tt);
    fprintf(opt.fid, fmt, cm(tt,:));
    fprintf(opt.fid, '\n');
  end
end
