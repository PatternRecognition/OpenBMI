reference= 'VPcm_06_06_13/arteVPcm';
damaged= 'VPcm_06_06_06/arteVPcm';
%damaged= 'Friederike_06_06_06/arteFriederike';

mkref= eegfile_loadMatlab(reference, 'vars','mrk');
mrk= eegfile_loadMatlab(damaged, 'vars','mrk');

startseq= [1:length(mrk.className)]*mrk.y(:,1:3);
startref= [1:length(mkref.className)]*mkref.y(:,1:3);

if ~isequal(mrk.className(startseq),mkref.className(startref)),
  error;
end

%for k= 1:length(mkref.className),
%  if isempty(strmatch(mkref.className{k},mrk.className)),
%    mrk.className{end+1}= mkref.className{k};
%    mrk.y= [mrk.y; zeros(1,size(mrk.y,2))];
%    ii= find(mkref.y(k,:));
%    pos= mkref.pos(ii) - mkref.pos(1) + mrk.pos(1);
%    mrk.pos= [mrk.pos, pos];
%    lab= zeros(size(mrk.y,1), 1);
%    lab(end)= 1;
%    mrk.y= [mrk.y, repmat(lab, [1 length(pos)])];
%  end
%end
%mrk= mrk_sortChronologically(mrk);

mrk_fix= mkref;
mrk_fix.pos= mkref.pos - mkref.pos(1) + mrk.pos(1);
mrk= mrk_fix;

save([EEG_MAT_DIR damaged], '-APPEND','mrk');
