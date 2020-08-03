function [cnt_tdp]= proc_tdp(cnt, p_list);
%PROC_TDP - Calculate Time-Domain Parameters
%
%Synopsis:
% CNT_TDP= proc_tdp(CNT, P)
%
%Arguments:
% CNT: structure of continuous data
% P:   model order
%
%Returns:
% CNT_TDP: structure of time-domain parameters

p= max(p_list);
[T,ch]= size(cnt.x);
cnt_tdp= cnt;
cnt_tdp.x= zeros(T, (p+1)*ch);
cnt_tdp.x(:,1:ch)= cnt.x;
zz= zeros([1 ch]);
for ip= 1:p,
  cnt_tdp.x(:,ip*ch+1:(ip+1)*ch)= ...
      diff([zz; cnt_tdp.x(:,(ip-1)*ch+1:ip*ch)], 1, 1);
	cnt_tdp.clab= cat(2, cnt_tdp.clab, strcat(cnt.clab, ['_tdp' int2str(ip)]));
end
sel= ismember(floor([0:(p+1)*ch-1]/ch), p_list);
cnt_tdp.x(:,find(~sel))= [];
cnt_tdp.clab= cnt_tdp.clab(find(sel));

return



%% This is the old version.

%function [cnt_tdp]= proc_tdp(cnt, p);
%PROC_TDP - Calculate Time-Domain Parameters
%
%Synopsis:
% CNT_TDP= proc_tdp(CNT, P)
%
%Arguments:
% CNT: structure of continuous data
% P:   model order
%
%Returns:
% CNT_TDP: structure of time-domain parameters


[T,ch]= size(cnt.x);
cnt_tdp= cnt;
cnt_tdp.x= zeros(T, (p+1)*ch);
cnt_tdp.x(:,1:ch)= cnt.x;
zz= zeros([1 ch]);
for ip= 1:p,
  cnt_tdp.x(:,ip*ch+1:(ip+1)*ch)= ...
      diff([zz; cnt_tdp.x(:,(ip-1)*ch+1:ip*ch)], 1, 1);
	cnt_tdp.clab= cat(2, cnt_tdp.clab, strcat(cnt.clab, ['_tdp' int2str(ip)]));
end

