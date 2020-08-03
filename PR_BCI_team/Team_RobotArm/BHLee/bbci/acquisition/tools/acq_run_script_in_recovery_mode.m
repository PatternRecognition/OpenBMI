global TMP_DIR 

if ~isfield(ACQLOCAL,'file'),
  error('ACQLOCAL.file must be defined');
end
if ~isfield('ACQLOCAL','opt'),
  ACQLOCAL.opt= [];
end
ACQLOCAL.opt= set_defaults(ACQLOCAL.opt, ...
                       'blockmarker', '%-newblock', ...
                       'tmp_dir', TMP_DIR);

ACQLOCAL.today_vec= clock;
ACQLOCAL.today_str= sprintf('%02d_%02d_%02d', ACQLOCAL.today_vec(1)-2000, ...
                            ACQLOCAL.today_vec(2:3));

ACQLOCAL.tmpfile= [ACQLOCAL.opt.tmp_dir 'acquisition_script_state_' ...
                   ACQLOCAL.today_str];

ACQLOCAL.S= textread(ACQLOCAL.file,'%s', 'delimiter','\n');
ACQLOCAL.S= strtrim(ACQLOCAL.S);
ACQLOCAL.S(cellfun(@isempty, ACQLOCAL.S))= [];
ACQLOCAL.iblock= [0; strmatch(ACQLOCAL.opt.blockmarker, ACQLOCAL.S)];

ACQLOCAL.block_completed= 0;
if exist([ACQLOCAL.tmpfile '.mat'], 'file'),
  ACQLOCAL.tmp= load(ACQLOCAL.tmpfile, 'block_completed');
  ACQLOCAL.block_completed= ACQLOCAL.tmp.block_completed;
  fprintf('Restarting script %s at block %d.\n', ACQLOCAL.file, ...
          ACQLOCAL.block_completed+1);
else
  fprintf('Starting freshly script %s.\n', ACQLOCAL.file);
end

% From now on, we use only variables with prefix 'ACQLOCAL.' to make them
% distinct from the variables that are used in the run script:
ACQLOCAL.ptr= ACQLOCAL.iblock(ACQLOCAL.block_completed+1);
while ACQLOCAL.ptr<length(ACQLOCAL.S),
  ACQLOCAL.cmdstr= '...';
  while length(ACQLOCAL.cmdstr)>=3 && strcmp(ACQLOCAL.cmdstr(end-2:end), '...'),
    ACQLOCAL.ptr= ACQLOCAL.ptr+1;
    ACQLOCAL.cmdstr= strcat(ACQLOCAL.cmdstr(1:end-3), deblank(ACQLOCAL.S{ACQLOCAL.ptr}));
  end
  eval(ACQLOCAL.cmdstr);
  if ismember(ACQLOCAL.ptr, ACQLOCAL.iblock),
    ACQLOCAL.block_completed= ACQLOCAL.block_completed+1;
    save(ACQLOCAL.tmpfile, '-STRUCT', 'ACQLOCAL', 'block_completed');
    fprintf('\n- <Block no. %d completed.> -\n\n', ACQLOCAL.block_completed);
  end
end

delete([ACQLOCAL.tmpfile '.mat']);
