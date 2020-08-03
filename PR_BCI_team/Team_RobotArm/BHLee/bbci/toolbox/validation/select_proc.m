function [best_proc, classy, min_loss, min_loss_std, ms_memo]= ...
    select_proc(fv, model, proc, varargin)
%[best_proc, classy, min_loss, min_loss_std]= ...
%    select_proc(fv, model, proc, <opt>)
%
% Select the best values for free variables in a fv transform (proc).
% When proc is a cell array of fv transforms the best of those
% transforms is selected (after the best variable assignment for
% each transform was selected).
% 'best' is measured by xvalidation.
%
% IN  fv    - struct of feature vectors
%     model - classification model, see select_model
%     proc  - a fv transform with (possibly) free variable, for which
%             a list of values is defined which are tried out and
%             the best value for each free variable is selected.
%             proc is either a string that can be evaluated
%             by matlab's eval function (no free variables),
%             or a struct with fields
%         .eval  - a string that can be eval'ed (fv -> fv)
%         .param - a struct array with one struct for each free
%                  variable in proc.eval with fields
%               .var   - variable name (string)
%               .value - vector or cell array of values,
%                        when this is of length>1, the proc-transform
%                        has 'free parameters', otherwise all parameters
%                        are assigned
%         .memo - a cell array of variable names that are assigned in the
%                 evaluation of proc.eval and should be saved
%                 (for only one variable .memo may just be a string)
%             finally proc can be a cell array of fv transforms are
%             described above.
%     opt   - property/value list or struct of options
%       .verbosity - output intermediate results
%       opt is passed to select_model / xvalidation.
%
% OUT best_proc - selected proc with selected assignment of free
%                 variables. the list of variables is extended by
%                 the variable proc.memo
%     classy    - selected classifier
%     min_loss  - loss as obtained by xvalidation of the selected proc
%     min_loss_std - the respective std
% 
% SEE select_model, xvalidation, proc_applyProc

%% bb ida.first.fhg.de 07/2004

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'verbosity',0, ...
                  'out_prefix', '', ...
                  'debug', 0);

if opt.verbosity==0,
  opt.progress_bar= 0;
end

if iscell(proc),
  min_loss= inf;
  for cc= 1:length(proc),
    if opt.verbosity>0 | opt.debug,
      fprintf('validating: ');
      disp_proc(proc{cc}, 'param',0);
    end
    [pp, cl, ml, mls]= select_proc(fv, model, proc{cc}, opt);
    if ml(1)<min_loss(1),
      classy= cl;
      min_loss= ml;
      min_loss_std= mls;
      best_proc= pp;
    end
  end
  return;
elseif ~prochasfreevar(proc),  %% nothing to select
  best_proc= proc;
  if nargout>1,
    if nargout>2,
      [classy, min_loss, min_loss_std, P, out_test,ms_memo]= select_model(fv, model, opt);
    else
      %% this does not perform xvalidation when the model has no
      %% free parameters
      classy= select_model(fv, model, opt);
    end
  end
  return;
end

nParams= length(proc.param);
pvi= ones(1, nParams);
endpvi= cell2mat(apply_cellwise({proc.param(:).value}, 'length'));

prefix_memo= opt.out_prefix;
ok= 1;
min_loss= inf;
while ok,
  if opt.verbosity>0,
    opt.out_prefix= [disp_proc(proc, 'proc',0, 'param',1, 'lf',0,'pvi',pvi) ...
                     ' -> ' prefix_memo];
  end
%  ff= proc_applyProc(fv, proc, pvi);
%  valid= find(all(~isnan(ff.y)));
%  ff= xval_selectSamples(ff, valid);
%  [classy, loss, loss_std]= select_model(ff, model, opt, 'proc','');
  [classy, loss, loss_std]= select_model(fv, model, opt, ...
                                         'proc',setfield(proc, 'pvi',pvi));
  if loss(1)<min_loss(1),
    best_pvi= pvi;
    min_loss= loss;
    min_loss_std= loss_std;
  end
  ok= 0;
  vv= 1;
  while ~ok & vv<=nParams,
    pvi(vv)= pvi(vv)+1;
    if pvi(vv)<=endpvi(vv),
      ok= 1;
    else
      pvi(vv)= 1;
      vv= vv+1;
    end
  end
end

best_proc= proc;
for pp= 1:nParams,
  best_proc.param(pp).value= {proc.param(pp).value{best_pvi(pp)}};
end
