function [epo,w]  = proc_spatialprojection(epo, nProj, varargin);
%PROC_SENSORPROJECTON projects all channels to one 
%
% usage:
%  [epo,w]= proc_spatialprojection(epo, <nProj, [signal, <criteria, param>], ...>);
%
% input: 
%       epo      usual two classes epo structure
%       nProj    the number of spatial projections (default 1, can
%                omitted)
%       signal   A signal as matrix or struct.x where further
%                optimisation takes place, if signal is not given, no further
%                optimisation takes place. If no further
%                optimisation is given, only one projection by
%                setting norm(w)=1 is given.
%       criteria 'id' to use the identity signal (default)
%                'smooth' to minimize second derivation
%                criteria can be a cell array, where the first
%                argument has the same meaning as above during the
%                second is a parameter for comparing different
%                optimisation. With only one signal a parameter
%                makes no sense! The parameter is 1 per default
%       param    is the parameter described in param. 
% NOTE: criteria and/or param can be left out by giving an empty
% field or by going directly to the next entry.
%
% output:
%       epo - with one channel
%       w - the projection transformation
%
% by an idea of Michael Zibulevsky
% GUido Dornhege, 13/02/2003

signal = {};
criteria = {};
param = [];

state = 1;

args = {varargin{:}};

if ~exist('nProj','var') | isempty(nProj)
  nProj = 1;
end

if ~isnumeric(nProj) | length(nProj(:))>1
  args = {nProj, args{:}};
  nProj = 1;
end

field = 0; 

while ~isempty(args)
  arg = args{1};
  if isstruct(arg)
    arg = arg.x;
    fi = 1;
  elseif iscell(arg)
    args{1} = arg{1};
    args{3:end+1} = {args{2:end}};
    args{2} = arg{2};
    fi = 0;
  elseif ischar(arg)
    fi = 2;
  elseif isnumeric(arg) & length(arg)==1
    fi = 3;
  elseif isnumeric(arg)
    fi = 1;
  elseif isempty(arg)
    fi = 4
  end

  
  switch fi
   case 0
    % nothing
   case 4
    % go to the next
    state = mod(state,3)+1;
    args = {args{2:end}};
   case 2
    if state ==1 | state == 3
      error([mfilename ': Wrong Input: String at wrong place']);
    end
    args = {args{2:end}};
    if field>0    
      criteria{field} = arg;
      state = 3;
    end
   case 3
    if state == 1 
      error([mfilename ': Wrong input: Parameter at wrong place']);
    end
    args = {args{2:end}};
    if field>0    
      param(field) = arg;
      state = 1;
    end
   case 1
    field = length(signal)+1;
    signal{field} = arg;
    param(field) = 1;
    criteria{field} = 'id';
    state = 2;
    args = {args{2:end}};
  end
end


if size(epo.y,1)==1
  ind1 = find(epo.y<0);
  ind2 = find(epo.y>0);
else  
  ind1 = find(epo.y(1,:));
  ind2 = find(epo.y(2,:));
end

cla1x = mean(epo.x(:,:,ind1),3);
cla2x = mean(epo.x(:,:,ind2),3);

A = (cla1x-cla2x)'*(cla1x-cla2x);

if length(signal)>0
  B = zeros(size(epo.x,2));
  for i = 1:length(signal)
    sig = signal{i};
    crit = criteria{i};
    para = param(i);
    switch crit
     case 'id'
      sig = permute(sig,[1 3:length(size(sig)) 2]);
      CC = size(sig);
      sig = reshape(sig,[prod(CC(1:end-1)),CC(end)]);
      D = sig'*sig;
     case 'smooth'
      CC = size(sig);
      sig = reshape(sig(3:end,:)-2*sig(2:end-1,:)+sig(1:end-2,:), ...
		    [CC(1)-2,CC(2:end)]);
      sig = permute(sig,[1 3:length(size(sig)) 2]);
      CC = size(sig);
      sig = reshape(sig,[prod(CC(1:end-1)),CC(end)]);
      D = sig'*sig;
    end
    B = B+para*D;
  end
  if sum(abs(B(:)))==0
    piB = eye(size(epo.x,2));
  else
    B = chol(B);
    piB = pinv(B);
  end
else
  piB = eye(size(epo.x,2));
end

M = piB'*A*piB;
      
[v,d] = eigs(M,nProj,'LM',struct('disp',0));


w = piB*v;

nn = size(epo.x);
dat = permute(epo.x,[2 1 3:length(nn)]);
dat = w'*dat(:,:);
dat = reshape(dat,[nProj,nn(1),nn(3:end)]);
epo.x = permute(dat,[2 1 3:length(nn)]);


epo.clab = cellstr([repmat('Channel ',[nProj,1]),num2str((1:nProj)')])';
