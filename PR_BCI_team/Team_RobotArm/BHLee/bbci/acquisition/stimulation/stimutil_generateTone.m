function tone= stimutil_generateTone(freq, varargin)

% blanker@cs.tu-berlin.de, Nov-2007

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'duration', 500, ...
                 'harmonics', 7, ...
                 'pan', [1 1], ...
                 'fs', 22050, ...
                 'rampon', 25, ...
                 'rampoff', 50);

freqs= freq * [1:(opt.harmonics)]';
wFreqs= freqs.*(2*pi/opt.fs);
db= linspace(60, 0, opt.harmonics+1);
db(end)= [];
%if length(db)>3,
%  db(1)= db(3);
%end
amp= 20*10.^(db/20);
phases= 2*pi * rand(opt.harmonics, 1);
N= ceil(opt.duration/1000*opt.fs);

tone= amp * sin(wFreqs*(1:N) + repmat(phases, [1 N]));

Non= round(opt.rampon/1000*opt.fs);
ramp= sin((1:Non)*pi/Non/2).^2;
tone(1:Non)= tone(1:Non) .* ramp;
Noff= round(opt.rampoff/1000*opt.fs);
ramp= cos((1:Noff)*pi/Noff/2).^2;
tone(end-Noff+1:end)= tone(end-Noff+1:end) .* ramp;

tone= [opt.pan(:) * tone/20000]';
