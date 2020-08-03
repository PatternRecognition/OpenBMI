function [fv,proc] = proc_combineFeatures(epo,varargin);
%PROC_COMBINEFEATURES combine features for the use with train_probCombiner and son on
%
% usage: 
%   [fv,proc_train,proc_apply] = proc_combineFeatures(epo1,epo2,...);
%
% input:
%   epo      usual epo structure, use .proc_train, .proc_apply , .memo to announce label dependent class processing, 
%
% output:
%   fv          combined feature, with field classifier_param as cell array of dimensions
%   proc  should be moved to xvaliation
%
% DOES NOT TAKE CARE IF LABELS MATCH
%
% Guido Dornhege, 24/11/2004

fv = copyStruct(epo,'x','proc','proc');

siz = size(epo.x);

fv.classifier_param = {{siz(1:end-1)}};

if isfield(epo,'proc')
  if ~isstruct(epo.proc)
    epo.proc = struct('train', epo.proc);
  end
  if ~isfield(epo.proc,'apply')
    epo.proc.apply = epo.proc.train;
  end
      
  if isfield(epo.proc,'memo')
    proc.train = ['[fv,hlp_combine1] = process_extractfeatures(fv,1,fv.proc_train1,fv.comb_memo1);'];
    fv.comb_memo1 = epo.proc.memo;
    proc.memo = {'hlp_combine1'};
  else
    proc.train = ['fv = process_extractfeatures(fv,1,fv.proc_train1);'];
    memo = {};
  end
  fv.proc_train1 = epo.proc.train;
  if isfield(epo.proc,'memo')
    proc.apply = ['fv = process_extractfeatures(fv,1,fv.proc_apply1,fv.comb_memo1,hlp_combine1);'];
  else
    proc.apply = ['fv = process_extractfeatures(fv,1,fv.proc_apply1);'];
  end
  fv.proc_apply1 = epo.proc.apply;
else
  proc.train = '';
  proc.apply = '';
  proc.memo = {};
end


ff = proc_flaten(epo);
fv.x = ff.x;


for i = 1:length(varargin)
  epo= varargin{i};
  
  siz = size(epo.x);

  fv.classifier_param{1}{i+1} = siz(1:end-1);

  if isfield(epo,'proc')
    if ~isstruct(epo.proc)
      epo.proc = struct('train', epo.proc);
    end
    if ~isfield(epo.proc,'apply')
      epo.proc.apply = epo.proc.train;
    end
    
    if isfield(epo.proc,'memo')
      proc.train = [proc.train, '[fv,hlp_combine' int2str(i+1) '] = process_extractfeatures(fv,' int2str(i+1) ',fv.proc_train' int2str(i+1) ',fv.comb_memo' int2str(i+1) ');'];
      eval(sprintf('fv.comb_memo%i = epo.proc.memo;',i+1));
      proc.memo = {proc.memo{:},['hlp_combine' int2str(i+1)]};
    else
      proc.train = [proc.train,'fv = process_extractfeatures(fv,' int2str(i+1) ',fv.proc_train' int2str(i+1) ');'];
    memo = {};
    end
    eval(sprintf('fv.proc_train%i = epo.proc.train;',i+1));
    if isfield(epo.proc,'memo')
      proc.apply = [proc.apply,'fv = process_extractfeatures(fv,' int2str(i+1) ',fv.proc_apply' int2str(i+1) ',fv.comb_memo' int2str(i+1) ',hlp_combine' int2str(i+1) ');'];
    else
      proc.apply = [proc.apply,'fv = process_extractfeatures(fv,' int2str(i+1) ',fv.proc_apply' int2str(i+1) ');'];
    end
    eval(sprintf('fv.proc_apply%i = epo.proc.apply;',i+1));
  end

  
  ff = proc_flaten(epo);
  fv.x = cat(1,fv.x,ff.x);

end

