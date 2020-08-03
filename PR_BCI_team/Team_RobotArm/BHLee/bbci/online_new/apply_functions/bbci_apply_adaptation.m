function [bbci, data]= bbci_apply_adaptation(bbci, data, cmd)
%BBCI_APPLY_ADAPTATION - Wrapper for specific adaptation procedures
%
%Synopsis:
%  [BBCI, DATA]= bbci_apply_adaptation(BBCI, DATA, 'init')
%  [BBCI, DATA]= bbci_apply_adaptation(BBCI, DATA)

% 03-2011 Benjamin Blankertz


BA= bbci.adaptation;
if length(BA)==1 && ...
      strcmp(BA.mode,'everything_at_once'),
  if ~BA.active,
    return;
  end
  if nargin>2 && strcmp(cmd, 'close'),
    % Closing case
    fields= {'signal','feature','classifier','control'};
    save(fullfile(BA.folder, BA.file), '-STRUCT', 'bbci', fields{:});
    str= sprintf('# %s final adapted classifier saved as <%s>.', ...
                 data.adaptation{k}.opt.tag, BA.file);
    bbci_log_write(data.adaptation{1}.log.fid, str);
  elseif nargin>2 && strcmp(cmd, 'init'),
    % Init case
    tag= upper(func2str(BA.fcn));
    if BA.load_classifier,
      filename= fullfile(BA.folder, BA.file);
      if exist([filename '.mat'], 'file'),
        S= load(filename);
        fields= fieldnames(S);
        for j= 1:length(fields),
          bbci.(fields{j})= S.(fields{j});
        end
        str= sprintf('# %s classifier loaded from file <%s>', ...
                     tag, BA.file);
      else
        str= sprintf('# %s requested classifier file <%s> not found !!!', ...
                     tag, filename);
        % Or should we better issue an error here?
        warning(str);
      end
      bbci_log_write(data.adaptation{1}.log.fid, str);
    end
    [bbci, data]= BA.fcn(bbci, data, 'init', BA.param{:}, 'tag',tag);
  else
    [bbci_new, data]= BA.fcn(bbci, data);
    if BA.save_everytime && ~isequal(bbci_new, bbci),
      bbci= bbci_new;
      fields= {'signal','feature','classifier','control'};
      save(fullfile(BA.folder, BA.file), '-STRUCT', 'bbci', fields{:});
      str= sprintf('# %s adapted classifier saved as <%s>.', ...
                   data.adaptation.opt.tag, BA.file);
      bbci_log_write(data.adaptation{1}.log.fid, str);
    else
      bbci= bbci_new;
    end
  end
  return;
end

for k= 1:length(BA),
  if ~BA(k).active,
    continue;
  end
  icls= BA(k).classifier;
  if nargin>2 && strcmp(cmd, 'close'),
    % Closing case
    classifier= bbci.classifier(icls);
    save(fullfile(BA(k).folder, BA(k).file), '-STRUCT', 'classifier');
    str= sprintf('# %s final adapted classifier saved as <%s>.', ...
                 data.adaptation{k}.opt.tag, BA(k).file);
    bbci_log_write(data.adaptation{k}.log.fid, str);
  elseif nargin>2 && strcmp(cmd, 'init'),
    % Init case
    tag= upper(func2str(BA(k).fcn));
    if length(BA)>1.
      tag= [tag sprintf('-%02d', k)];
    end
    if BA(k).load_classifier,
      filename= fullfile(BA(k).folder, BA(k).file);
      if exist([filename '.mat'], 'file'),
        bbci.classifier(icls)= load(filename);
        str= sprintf('# %s classifier loaded from file <%s>', ...
                     tag, BA(k).file);
      else
        str= sprintf('# %s requested file <%s> not found', ...
                     tag, filename);
        % Or should we better issue an error here?
        warning(str);
      end
      bbci_log_write(data.adaptation{k}.log.fid, escape_printf(str));
    end
    [bbci.classifier(icls), data.adaptation{k}]= ...
        BA(k).fcn(bbci.classifier(icls), ...
                  data.adaptation{k}, 'init', BA(k).param{:}, 'tag',tag);
  else
    ifeat= bbci.classifier(icls).feature;
    [classifier, data.adaptation{k}]= ...
        BA(k).fcn(bbci.classifier(icls), ...
                  data.adaptation{k}, data.marker, data.feature(ifeat));
    if BA(k).save_everytime && ~isequal(classifier, bbci.classifier(icls)),
      save(fullfile(BA(k).folder, BA(k).file), '-STRUCT', 'classifier');
      str= sprintf('# %s adapted classifier saved as <%s>.', ...
                   data.adaptation{k}.opt.tag, BA(k).file);
      bbci_log_write(data.adaptation{k}.log.fid, str);
    end
    bbci.classifier(icls)= classifier;
  end
end
