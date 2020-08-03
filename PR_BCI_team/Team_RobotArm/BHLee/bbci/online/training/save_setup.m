% $Id: save_setup.m,v 1.1 2006/04/27 14:24:38 neuro_cvs Exp $

% the variables to save

force= 0;
vars = {'bbci','cont_proc','feature','cls','post_proc','marker_output'};

% if find(ismember(vars, bbci))
%   vars= {'bbci'};
% end

% Check whether such a setup file exists. Several setup files may be
% generated and get individual numbers
typ = 1;
save_file = bbci.save_name;
if exist(strcat(save_file, '.mat'),'file');
  if force
    warning('%s already existed and has been overwritten.', save_file);
  else
    num = 1;
    while exist(strcat(save_file, '.mat'), 'file')
      num = num + 1;
      save_file= sprintf('%s%03d', save_file, num);
    end
    warning('%s already existed, save classifier to ', bbci.save_name, save_file);
  end
  bbci.save_name= save_file;
end

logfi = false;

if ~isfield(bbci,'logfilebase')
  bbci.logfilebase = [save_file, '_log'];
  logfi = true;
end

if ~exist(bbci.logfilebase, 'dir'),
  mkdir_rec(bbci.logfilebase);
end

save(save_file,vars{:});

if isunix
  eval(['! chmod g=u ' bbci.logfilebase]);
end

if logfi
  bbci = rmfield(bbci,'logfilebase');
end

message_box(sprintf('Setup written to %s\n',save_file),1);

if isfield(bbci, 'store_compatibility') & bbci.store_comatibility,
    settings = load(save_file);
    bbci = bbci_apply_convertSettings(settings);
    save([save_file '_new'], bbci);
end


