function [epo_tr, epo_te] = Divide_tr_te_bbci(epo)
%
% input
%       smt: smt
%
% output
%       smt_tr: SMT for training
%       smt_te: SMT for test
%

nSamples_tr = floor(size(epo.y,2)*0.8);

epo_tr = struct('fs',epo.fs,'clab',{epo.clab},'x',epo.x(:,:,1:nSamples_tr),...
    't',epo.t,'y',epo.y(:,1:nSamples_tr),'className',{epo.className},'refIval',epo.refIval);

epo_te = struct('fs',epo.fs,'clab',{epo.clab},'x',epo.x(:,:,1+nSamples_tr:end),...
    't',epo.t,'y',epo.y(:,1+nSamples_tr:end),'className',{epo.className},'refIval',epo.refIval);






