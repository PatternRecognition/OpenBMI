% LOOKS FOR A NEW SETUP AND MODIFY. FURTHERMORE SOME INFORMATIONS ARE SENT BACK

if ~exist('BBCI_BET_APPLY','var')
  BBCI_BET_APPLY = false;
end
if ~exist('run','var')
  run = true;
end
if ~exist('loop','var')
  loop = true;
end

allowedtimedelay = 10/60/60/24; % 10 seconds as portion of one day

get_actual_setup_list = {};
get_directory_informations = {};
timecheck = [];

new_setup = get_data_udp(control_communication,0);
new_setup2 = '';
while ~isempty(new_setup)
% check for a new setup
  
  setupchanged = 1;  
  
  tw = 0.1;
  
  new_setup = char(new_setup);
  new_setup2 = [new_setup2,new_setup];
  new_setup
  run_old = run; loop_old = loop;

  try
    if isempty(strfind(new_setup,'!')) & isempty(strfind(new_setup,'system')) & isempty(strfind(new_setup,'dos')) & isempty(strfind(new_setup,'unix'))
      eval(new_setup);
      if exist('feedback_opt','var') & ~isempty(strfind(new_setup,'feedback_opt')) 
        if ~loop
          clear feedback_opt;
          eval(new_setup);
          if BBCI_BET_APPLY
            run_fb = run; loop_fb = loop;
            run = run_old; loop = loop_old;
          end
        end
        feedback_opt.changed = 1;

      end
    end
  end
  
  if ~isempty(timecheck) & abs(now-timecheck)>allowedtimedelay
    error('The system times are not identical, please check!');
  end
    
  % inform about setup_list
  if ~isempty(get_actual_setup_list)
    if iscell(setup_list)
      str = sprintf('%s,',setup_list{:});
    else
      str = '. ';
    end
    str = double(str(1:end-1));
    if isempty(str)
      str = [];
    end
    send_data_udp(get_actual_setup_list{1},get_actual_setup_list{2},str);
    setupchanged = 0;
  end
  
  % inform about directory informations
  if ~isempty(get_directory_informations)
    if ~exist(get_directory_informations{1})
      str = '-1';
    else
      d = dir(get_directory_informations{1});
      ind = find([d.isdir]);
      str = sprintf('%s\n',d(ind).name);
      str = sprintf('%s\n',str);
      d = dir([get_directory_informations{1} '/*.mat']);
      isvalid= ones(1, length(d));
      for i = 1:length(d),
        matfile= [get_directory_informations{1} '/' d(i).name];
        if isempty(whos('-file',matfile,'bbci')),
          isvalid(i)= 0;
        end
      end
      d= d(find(isvalid));
      for i = 1:length(d);
        S = load([get_directory_informations{1} '/' d(i).name],'bbci');
        if isfield(S.bbci,'player'),  %% check for valid bbci struct
          if isfield(S.bbci,'classes');
            if iscell(S.bbci.classes)
              classstr= sprintf('%s/',S.bbci.classes{:});
              classstr = classstr(1:end-1);
            else
              classstr = '';
            end
          else
            classstr = '';
          end
          str = sprintf('%s%s - %d - %s\n',str,d(i).name,S.bbci.player,classstr);
        end
        
      end
    end                   

    str = double(str);
    
    send_data_udp(get_directory_informations{2},get_directory_informations{3},str);
    
    setupchanged = 0;
  end
  new_setup = get_data_udp(control_communication,tw);

end

new_setup = new_setup2;