function dat = nirs_LB(dat, varargin)
% Synopsis:
%   DAT = nirs_LB(DAT, 'Property1',Value1, ...)
%
% Arguments:
%   DAT: data struct with NIRS cnt data split in the x field
%
%   OPT - struct or property/value list of optional properties:
%   'citation'   - if set, the epsilon (extinction values) is taken from
%                  the specified citation number (see GetExtinctions.m for
%                  details). (default 1 = Gratzer et al)
%   'epsilon'    - sets extinction coefficients manually (wl1 wl2 vs deoxy
%                  and oxy). In this case, citation is ignored. State in
%                  millimol/liter(?).
%   'opdist'     - optode (source-detector) distance in cm (default 2.5)
%   'ival'       - either 'all' (default) or a vector [start end] in samples
%                  specifying the baseline for the LB transformation
%   'DPF'        - differential pathlength factor: probabilistic average
%                  distance travelled by photons, default [5.98 7.15]
%  
% Returns:
%   DAT - updated data struct with oxy/deoxy fields specifying absorption
%         values in mmol/l.
%
%
% See also: nirs_* nirsfile_* GetExtinctions
% 
% Note: Based on the nirX Nilab toolbox functions u_LBG and u_popLBG.
%
% matthias.treder@tu-berlin.de 2011

opt = propertylist2struct(varargin{:});
[opt,isdefault] = ...
    set_defaults(opt, ...
                 'citation',1, ...
                 'opdist', 2.5, ...
                 'ival', 'all', ...
                 'DPF',  [5.98 7.15], ...
                 'epsilon', [], ...
                 'verbose', 0);


if strcmp(opt.ival,'all')
  opt.ival = [1 size(dat.x,1)];
end

s1=size(dat.x,1);
s2=size(dat.x,2)/2;

wl1 = dat.x(:,1:end/2) + eps;
wl2 = dat.x(:,end/2+1:end) + eps;

%% Get epsilon
if isempty(opt.epsilon)
  if ~isfield(dat,'wavelengths')
    error('Wavelengths should be given in the .wavelengths field.')
  end
  [ext,nfo] = GetExtinctions(dat.wavelengths,opt.citation);
  if opt.verbose, fprintf('Citation: %s\n',nfo), end
  epsilon = ext(:,1:2);
  
  % Divide by 1000 to obtain the required unit
  epsilon = epsilon/1000;
  
else
  epsilon = opt.epsilon;
end

%% Arrange epsilon so that higher wavelength is on top
[mw,idx] = max(dat.wavelengths);
if idx==2 % higher wavelength is on bottom
  epsilon = flipud(epsilon);
  if opt.verbose, fprintf('Epsilon matrix was rearranged so that higher WL is on top\n'), end
end

%% Apply Lambert-Beer law
Att_highWL= real(-log10( wl2 ./ ...
    ( repmat(mean(wl2(opt.ival(1):opt.ival(2),:),1), [s1,1]))   ));

Att_lowWL= real(-log10( wl1./ ...
    ( repmat(mean(wl1(opt.ival(1):opt.ival(2),:),1), [s1,1]))   ));

A=[];
A(:,1)=reshape(Att_highWL,s1*s2,1);
A(:,2)=reshape(Att_lowWL,s1*s2,1);

%----------------------------------
%       3.cc
%----------------------------------
% e=...looks like this
%               oxy-Hb         deoxy-Hb
% higherWL: 830 | e: 0.974       0.693
% lowerWL : 690 | e: 0.35         2.1

e= epsilon/10;

e2=   e.* [opt.DPF' opt.DPF']  .*  opt.opdist;
c= ( inv(e2)*A'  )';

dat.x    =reshape(c(:,1),s1,s2); %in mmol/l
dat.x    = [dat.x reshape(c(:,2),s1,s2)]; %in mmol/l

dat.yUnit = 'mmol/l';