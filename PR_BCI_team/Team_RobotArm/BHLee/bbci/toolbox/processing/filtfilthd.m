function x=filtfilthd(varargin)
% FILTFILTHD Zero-phase digital filtering with dfilt objects.
%
% FILTFILLTHD provides zero phase filtering and accepts dfilt objects on
% input. A number of end-effect minimization methods are supported.
%
% Examples:
% x=FILTFILTHD(Hd, x)
% x=FILTFILTHD(Hd, x, method)
% where Hd is a dfilt object and x is the input data. If x is a matrix,
% each column will be filtered.
%
% ------------------------------------------------------------------------
% The filter states of Hd on entry to FILTFILTHD will be used at the
% beginning of each forward and each backward pass through the data. The 
% user should normally ensure that the initial states are zeroed before
% calling filtfilthd [e.g. using reset(Hd);]
% ------------------------------------------------------------------------
%
% x=FILTFILTHD(b, a, x) 
% x=FILTFILTHD(b, a, x, method)
%           format is also supported.
%
% x=FILTFILTHD(...., IMPULSELENGTH)
%   allows the impulse response length to be specified on input. 
%
%
% method is a string describing the end-effect correction technique:
%   reflect:      data at the ends are reflected and mirrored as in
%                   the MATLAB filtfilt function (default)
%   predict:      data are extraploated using linear prediction
%   spline/pchip: data are extrapolated using MATLAB's interp1 function
%   none:         no internal end-effect correction is applied
%                   (x may be pre-pended and appended with data externally)
%
% Each method has different merits/limitations. The most robust
% method is reflect.
% 
% The length of the padded data section at each end will be impzlength(Hd)
% points, or with 'reflect', the minimum of impzlength(Hd) and the
% data length (this is different to filtfilt where the padding is only
% 3 * the filter width). Using the longer padding reduces the need for any
% DC correction (see the filtfilt documentation).
%
%
% See also: dfilt, filtfilt, impzlength
%
% -------------------------------------------------------------------------
% Author: Malcolm Lidierth 10/07
% Copyright © The Author & King's College London 2007
% -------------------------------------------------------------------------
%
% Revisions:
%   07.11.07    nfact=len-1 (not len) when impulse response longer than
%                           data
%   11.11.07    Use x=f(x) for improved memory performance 
%                           [instead of y=f(x)]
%               Handles row vectors properly
%   31.01.08    Allow impzlength to be specified on input
%                           

% ARGUMENT CHECKS
if isnumeric(varargin{1}) && isnumeric(varargin{2})
    % [b, a] coefficients on input, convert to a singleton filter.
    Hd = dfilt.df2(varargin{1}, varargin{2});
    x=varargin{3};
elseif ~isempty(strfind(class(varargin{1}),'dfilt'))
    % dfilt object on input
    Hd=varargin{1};
    x=varargin{2};
else
    error('Input not recognized');
end

if ischar(varargin{end})
    method=varargin{end};
else
    method='reflect';
end

if isscalar(varargin{end})
    nfact=varargin{end};
else
    nfact=impzlength(Hd);
end

% DEAL WITH MATRIX INPUTS-----------------------------------------------
% Filter each column in turn through recursive calls
[m,n]=size(x);
if (m>1) && (n>1) 
    for i=1:n
        x(:,i)=filtfilthd(Hd, x(:,i), method, nfact);
    end
    return
end

% MAIN FUNCTION-------------------------------------------------------
% Make sure x is a column. Return to row vector later if needed
if m==1
    x = x(:);
    trflag=true;
else
    trflag=false;
end

len=length(x);
switch method
    case 'reflect'
        % This is similar to the MATLAB filtfilt reflect and mirror method
        nfact=min(len-1, nfact);%change to len-1 not len 07.11.07
        pre=2*x(1)-x(nfact+1:-1:2);
        post=2*x(len)-x(len-1:-1:len-nfact);
    case 'predict'
        % Use linear prediction. DC correction with mean(x).
        % Fit over 2*nfact points, with nfact coefficients
        np=2*nfact;
        m=mean(x(1:np));
        pre=lpredict(x(1:np)-m, nfact, nfact, 'pre')+m;
        m=mean(x(end-np+1:end));
        post=lpredict(x(end-np+1:end)-m, nfact, nfact, 'post')+m;
    case {'spline', 'pchip'}
        % Spline/pchip extrapolation.
        % Fit over 2*nfact points,
        np=2*nfact;
        pre=interp1(1:np, x(np:-1:1), np+1:np+nfact, method, 'extrap');
        pre=pre(end:-1:1)';
        post=interp1(1:np, x(end-np+1:end), np+1:np+nfact, method, 'extrap')';
    case 'none'
        % No end-effect correction
        pre=[];
        post=[];
end

% % UNCOMMENT TO VIEW THE PADDED DATA 
% if length(pre);plot(pre,'color', 'r');end;
% line(length(pre)+1:length(pre)+length(x), x);
% if length(post);line(length(pre)+length(x)+1:length(pre)+length(x)+length(post), post, 'color', 'm');end;


% Remember Hd is passed by reference - save entry state and restore later
memflag=get(Hd, 'persistentmemory');
states=get(Hd, 'States');
set(Hd,'persistentmemory', true);

%--------------------------------
% ----- FORWARD FILTER PASS -----
% User-supplied filter states at entry will be applied
% Pre-pended data
pre=filter(Hd, pre); %#ok<NASGU>
% Input data
x=filter(Hd ,x);
% Post-pended data
post=filter(Hd, post);

% ------ REVERSE FILTER PASS -----
% Restore user-supplied filter states for backward pass
set(Hd, 'States', states);
% Post-pended reversed data
post=filter(Hd, post(end:-1:1)); %#ok<NASGU>
% Reversed data
x=filter(Hd, x(end:-1:1));

% Restore data sequence
x=x(end:-1:1);
%---------------------------------
%---------------------------------

% Restore Hd
set(Hd, 'States', states);
set(Hd,'persistentmemory', memflag)

% Revert to row if necessary
if trflag
    x=x.';   
end

return
end


%--------------------------------------------------------------------------
function y=lpredict(x, np, npred, pos)
% LPREDICT estimates the values of a data set before/after the observed
% set.
% LPREDICT Local version. For a stand-alone version see:
% http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=16798&objectType=FILE
% -------------------------------------------------------------------------
% Author: Malcolm Lidierth 10/07
% Copyright © The Author & King's College London 2007
% -------------------------------------------------------------------------
% Order input sequence
if nargin==4 && strcmpi(pos,'pre')
    x=x(end:-1:1);
end
% Get the forward linear predictor coefficients via the LPC
% function
a=lpc(x,np);
% Negate coefficients, and get rid of a(1)
cc=-a(2:end);
% Pre-allocate output
y=zeros(npred,1);
% Seed y with the first value
y(1)=cc*x(end:-1:end-np+1);
% Next np-1 values
for k=2:min(np,npred)
    y(k)=cc*[y(k-1:-1:1); x(end:-1:end-np+k)];
end
% Now do the rest
for k=np+1:npred
    y(k)=cc*y(k-1:-1:k-np);
end
% Order the output sequence if required
if nargin==4 && strcmpi(pos,'pre')
    y=y(end:-1:1);
end
return
end
% -------------------------------------------------------------------------