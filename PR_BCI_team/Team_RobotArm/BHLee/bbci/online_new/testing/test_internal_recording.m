% --- Generate random signals and markers
C= 2;
T= 10000;
cnt= struct('fs', 100);
cnt.x= 50*randn(T, C);
cnt.clab= cprintf('Ch%d', 1:C)';

M= 100;
mrk= struct('fs', cnt.fs);
mrk.pos= round(linspace(0, T, M+2));
mrk.pos([1 end])= [];
mrk.desc= cprintf('S%3d', ceil(rand(1,M)*10))';


% --- Setup a very simple system for (simulated) online processing
bbci= struct;
bbci.source.acquire_fcn= @bbci_acquire_offline;
bbci.source.acquire_param= {cnt, mrk};
bbci.source.marker_mapping_fcn= @marker_mapping_SposRneg;

bbci.feature.proc= {@proc_variance};
bbci.feature.ival= [-500 0];

bbci.classifier.C= struct('w',randn(C,1), 'b',0);


% --- Enable recording with internal function
bbci.source.record_signals= 1;
bbci.source.record_basename= '/tmp/internal_recording_test';
%bbci.source.record_param= {'internal',1};  %% INT16
bbci.source.record_param= {'internal',1, 'precision','double'};


data= bbci_apply(bbci);


% --- Test consistency of results
[cnt_re, mrk_re]= eegfile_readBV(data.source.record.filename);

isequal(cnt.clab, cnt_re.clab)
max(abs(cnt.x(:)-cnt_re.x(:)))

isequal(mrk.pos, mrk_re.pos(2:end))
isequal(mrk.desc, mrk_re.desc(2:end))
