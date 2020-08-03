N= 60*5;

opt= struct('perc_dev', 0.33);
opt.intensity= [0.2 0.04];
opt.intensity_jitter_perc= 0.5;
opt.require_response= 0;
opt.bv_host= 'localhost';
opt.isi= 1000;
opt.breaks= [50 10];
opt.break_msg= 'Kurze Pause  (%d s)';

opt.cue_std= stimutil_generateTone(250, 'harmonics',1, 'duration',250, 'rampon',25,'rampoff',25);
T= length(opt.cue_std);
%hull= 1 - 0.4*exp(-(([1:T]-T/2)/T*5).^2)';
hull= 1 - 0.6*exp(-(([1:T]-T/2)/T*5).^2)';
opt.cue_dev= opt.cue_std .* repmat(hull, [1 2]);

desc= stimutil_readDescription('season10_oddballTactile');
stimutil_showDescription(desc, 'clf',1, 'waitfor',0);
