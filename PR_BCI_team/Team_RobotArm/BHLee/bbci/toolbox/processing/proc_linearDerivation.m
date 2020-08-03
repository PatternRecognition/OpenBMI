function out= proc_linearDerivation(dat, A, varargin)
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
out= dat;

nNewChans= size(A,2);
if ndims(dat.x)==2,
  out.x= dat.x*A;
else
  sz= size(dat.x);
  out.x= reshape(permute(dat.x, [1 3 2]), sz(1)*sz(3), sz(2));  
  out.x= out.x*A;
  out.x= permute(reshape(out.x, [sz(1) sz(3) nNewChans]), [1 3 2]);
end

if ~isdefault.clab,
  if isequal(opt.clab, 'generic'),
    out.clab= cellstr([repmat('ch',nNewChans,1) int2str((1:nNewChans)')])';
  elseif isequal(opt.clab, 'copy'),
    out.clab= dat.clab;
  else
    out.clab= opt.clab;
  end
elseif ~isdefault.prependix,
%  the following results, e.g., in 'csp 1', but the space is impractical
%  out.clab= cellstr([repmat(opt.prependix,nNewChans,1) ...
%		     int2str((1:nNewChans)')])';
  out.clab= cell(1,nNewChans);
  for ic= 1:nNewChans,
    out.clab{ic}= [opt.prependix int2str(ic)];
  end  
else
  no= NaN*ones(1, nNewChans);
  for ic= 1:nNewChans,
    io= find(A(:,ic)==1);
    if length(io)==1,
      no(ic)= io;
    end
  end
  
  out.clab= cell(1,nNewChans);
  if ~any(isnan(no)),
    for ic= 1:nNewChans,
      out.clab{ic}= [dat.clab{no(ic)} opt.appendix];
    end
  else
    for ic= 1:nNewChans,
      out.clab{ic}= [opt.prependix int2str(ic)];
    end
  end
end

out.origClab= dat.clab;
