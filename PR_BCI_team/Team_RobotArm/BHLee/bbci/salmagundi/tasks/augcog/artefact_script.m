list = {'ab_base','ab_kh','ab_kkh','ab_kkh_low_high','ab_ls_base', ...
	'ab_ls_high','ab_ls_low'};

li = 2;

mrk = readMarkerComments(['/home/schlauch/blanker/Daten/eegImport/augcog/EEG_KH_ARTEFACTM/',list{li}]);


cnt = readGenericEEG(['/home/schlauch/blanker/Daten/eegImport/' ...
		    'augcog/EEG_KH_ARTEFACTM/',list{li}]);

if li==7
  mrk.pos = [mrk.pos,size(cnt.x,1)];
  mrk.str = {mrk.str{1},'Off'};
end

mrk.y = eye(length(mrk.str)/2);
mrk.ival = reshape(mrk.pos,[2,length(mrk.str)/2]);
mrk.className = mrk.str(1:2:end);


  
  