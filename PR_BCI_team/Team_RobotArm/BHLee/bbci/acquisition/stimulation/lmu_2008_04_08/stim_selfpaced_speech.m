function stim_selfpaced_speech(varargin)
%stim_selfpaced_speech(<OPT>)
%
% 'duration': in minutes

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'filename', 'selfpaced_speech', ...
                  'test', 0, ...
                  'fs', 22050, ...
                  'duration', 5);

wav= stimutil_generate_tone(500, 'duration',150, 'fs',opt.fs);

isi_list= [1000 1000 1000 750 500 250];

if ~opt.test & ~isempty(opt.filename),
  bvr_startrecording([opt.filename VP_CODE]);
  pause(1);
end

ppTrigger(251);
pause(1);

waitForSync;
ppTrigger(101);
wavplay(wav, opt.fs);
for ii= 1:length(isi_list),
  waitForSync(isi_list(ii));
  ppTrigger(101);
  wavplay(wav, opt.fs);
end

pause(opt.duration*60);

waitForSync;
ppTrigger(101);
wavplay(wav, opt.fs);
for ii= 1:length(isi_list),
  waitForSync(isi_list(ii));
  ppTrigger(101);
  wavplay(wav, opt.fs);
end

pause(1);
ppTrigger(254);
pause(1);

if ~opt.test & ~isempty(opt.filename),
  bvr_sendcommand('stoprecording');
end
