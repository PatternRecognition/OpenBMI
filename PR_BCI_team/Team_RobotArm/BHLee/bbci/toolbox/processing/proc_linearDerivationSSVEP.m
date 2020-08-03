function out= proc_linearDerivationSSVEP(dat, f_inx, A,  varargin)
%out= proc_linearDerivation(dat, A, <OPT>)
%out= proc_linearDerivation(dat, A, <appendix>)
%
% calculate out.x= dat.x * A but the function works also for epoched data.
% if every column of A contains exactly one entry of value 1,
% channel labels are matched accordingly.
%
% IN   dat      - data structure of continuous or epoched data
%      A        - spatial filter matrix [nOldChans x nNewChans]
%      appendix - in case of input-output channel matching
%                 this string is appended to channel labels, default ''
%      OPT      - struct or property/value list of properties:
%       .clab - cell array of channel labels to be used for new channels,
%               or 'generic' which uses {'ch1', 'ch2', ...}
%               or 'copy' which copies the channel labels from input struct.
%       .appendix - (as above) tries to find (naive) channel matching
%               and uses as channel labels: old label plus opt.appendix.
%
% OUT  dat      - updated data structure

% bb, ida.first.fhg.de




if length(varargin)==1,
  opt= struct('appendix', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'clab', [], ...
                 'prependix', '', ...
                 'appendix', '');

%out= copy_struct(dat, 'not', 'x','clab');
clear out
if iscell(f_inx)
  f_inx=f_inx{1};
end

%if iscell(A)
%  A=A{1};
%end


N.ti=size(dat.x,1);
[N.cl N.freq]=size(f_inx);
N.ch=size(A{1},1);
out.x=[];

dat.x=reshape(dat.x,N.ti,N.ch/(sum(f_inx(1,:),2)),N.freq);

for i=1:N.cl
  x=dat.x(:,:,find(f_inx(i,:)));
  x=reshape(x,size(x,1),size(x,2)*size(x,3));
  out.x(:,:,i)=x*A{i};
end
out.clab= {'1','2','3','4','5','6','7','8','9','10','11','12'};
out.x=reshape(out.x,size(out.x,1),size(out.x,2)*size(out.x,3));