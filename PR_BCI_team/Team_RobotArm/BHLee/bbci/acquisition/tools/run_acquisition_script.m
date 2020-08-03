function run_acquisition_script(file, varargin)

global TMP_DIR 

%% The following variable may be used with in the run scripts
global VP_CODE CLSTAG TODAY_DIR general_port_fields

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'blockmarker', '%-newblock', ...
                  'tmp_dir', TMP_DIR);

today_vec= clock;
today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));

tmpfile= [opt.tmp_dir 'acquisition_script_state_' today_str];
                   
S= textread(file,'%s', 'delimiter','\n');
S= strtrim(S);
isblank= apply_cellwise2(S, 'isempty');
S(isblank)= [];
iblock= [0; strmatch(opt.blockmarker, S)];

block_completed= 0;
if exist([tmpfile '.mat'], 'file'),
  T= load(tmpfile);
  block_completed= T.LOCALblock_completed;
  fprintf('Restarting script %s at block %d.\n', file, block_completed+1);
else
  fprintf('Starting freshly script %s.\n', file);
end

% From now on, we use only variables with prefix 'LOCAL' to make them
% distinct from the variables that are used in the run script:
LOCAL_S= S;
LOCALiblock= iblock;
LOCALblock_completed= block_completed;
LOCALtmpfile= tmpfile;
LOCALptr= LOCALiblock(LOCALblock_completed+1);
while LOCALptr<length(LOCAL_S),
  LOCALcmdstr= '...';
  while strcmp(LOCALcmdstr(end-2:end), '...'),
    LOCALptr= LOCALptr+1;
    LOCALcmdstr= strcat(LOCALcmdstr(1:end-3), deblank(LOCAL_S{LOCALptr}));
  end
  eval(LOCALcmdstr);
  if ismember(LOCALptr, LOCALiblock),
    LOCALblock_completed= LOCALblock_completed+1;
    save(LOCALtmpfile, 'LOCALblock_completed');
    fprintf('\n- <Block no. %d completed.> -\n\n', LOCALblock_completed);
  end
end

delete([LOCALtmpfile '.mat']);
