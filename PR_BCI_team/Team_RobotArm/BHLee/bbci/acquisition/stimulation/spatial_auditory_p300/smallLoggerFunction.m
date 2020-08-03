function smallLoggerFunction(todo, data, varargin),

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'test', 1);

persistent fid poi data_dimension

global TODAY_DIR

switch todo,
  case 'init',
    data_dimension = [];
    ve = datevec(now);
    v = cat(1, {num2str(ve(1))}, ...
            cellstr(char(max('0',double(num2str(round(ve(2:6))'))))));    
    logDir = [TODAY_DIR 'AudioLog/'];
    if ~exist(logDir, 'dir'),
      mkdir(logDir);
    end
    name = data;
    poi=1;
    while exist(sprintf('%sfeedback_%s_%03i.log', ...
                        logDir,name,poi),'file'),
      poi = poi+1;
    end
    str = sprintf('%sfeedback_%s_%03i.log',logDir,name,poi);
    if ~isempty(fid) & fid>0 ; try;fclose(fid);end;end
    fid = fopen(str,'w');
    stri = sprintf('Routine started at ');
    stri = [stri,sprintf('%s_',v{1:end-1})];
    stri = sprintf('%s%s with values:\nfeedback_%s_fb_opt',stri,v{end},name);
    stri = [stri,'_', sprintf('%03i',poi),char(10)];
    fprintf(fid,'%s', stri);
    fb_opt = varargin;
    save([logDir 'feedback_' name '_fb_opt' ...
          sprintf('_%03i', poi)],'fb_opt');
    fprintf('Write log-file: %i\n',poi);
    stri = '';
  case 'add',
    if isempty(data_dimension),
      data_dimension = numel(data);
    end
    if numel(data) ~= data_dimension,
      warning('Wrong number of elements received for logging');
    end
    ve = datevec(now);
    v = cat(1, {num2str(ve(1))}, ...
            cellstr(char(max('0',double(num2str(round(ve(2:6))')))))); 
    stri = sprintf('%s_',v{1:end});
    stri = [stri(1:end-1) '  ;  '];
    stri = [stri createString(data)];
    fprintf(fid, '%s\n', stri);
    stri = '';
  case 'exit',
    try;fclose(fid);end
    stri = '';
    status = 0;
    fid = [];
end

function string = createString(data),
  string = '';
  for i = 1:length(data),
    if ischar(data{i}), 
      string = [string data{i} '  ;  '];
    elseif isnumeric(data{i}),
      if length(data{i}) == 1,
        string = [string num2str(data{i}) '  ;  '];
      else
        string = [string mat2str(data{i}) '  ;  '];
      end
    elseif islogical(data{i}),
      string = [string num2str(data{i}) '  ;  '];
    end
  end  
