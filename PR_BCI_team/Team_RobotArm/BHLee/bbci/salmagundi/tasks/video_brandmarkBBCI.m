function video_brandmarkBBCI(file, varargin)

global DATA_DIR TEX_DIR

[filepath, filename, fileext]= fileparts(file);
if isempty(fileext),
  fileext= '.avi';
end

video= propertylist2struct(varargin{:});
[video, isdefault]= ...
    set_defaults(video, ...
                 'verbose', 1, ...
                 'input_dir', [DATA_DIR 'eegVideo/'], ...
                 'output_dir', [DATA_DIR 'eegVideo/bbci_public/'], ...
                 'output_file', filename, ...
                 'override', 0, ...
                 'logo_file', [TEX_DIR 'pics/logos/bbci_black_trans.png'], ...
                 'brand', 1, ...
                 'audio', 0, ...
                 'audio_fs', 48000, ...
                 'size', [], ...
                 'type', [], ...
                 'intro', 1, ...
                 'intro_fadein', 1, ...
                 'intro_stay', 3, ...
                 'intro_fadeout', 1, ...
                 'extro', 1, ...
                 'extro_fadein', 0.5, ...
                 'extro_stay', 1, ...
                 'extro_fadeout', 0.5, ...
                 'extro_blank', 0.5, ...
                 'fs', 25, ...
                 'desc_text', {'The Berlin', 'Brain-Computer Interface', ...
                    'Project presents ...'}, ...
                 'logo_size', 0.3, ...
                 'logo_pos', [0.01 0.01], ...
                 'title_text', 'BBCI Feedback', ...
                 'title_pos', [0.5 0.35], ...
                 'title_fontsize', 0.06, ...
                 'desc_maxsize', [0.9 0.3], ...
                 'desc_textspec', {'FontSize',0.05}, ...
                 'desc_pos', [0.5 0.65], ...
                 'desc_boxgap', 0.02, ...
                 'copyright_text', '(c) BBCI:  Fraunhofer FIRST (IDA),  TU Berlin,  CBF Charité Berlin', ...
                 'copyright_fontsize', 0.035, ...
                 'copyright_pos', [0.5 0.99], ...
                 'extro_logo_size', 0.5, ...
                 'extro_logo_pos', [0.5 0.5], ...
                 'web_text', 'http://www.bbci.de', ...
                 'web_textspec', {'FontSize',0.04, 'FontName','Courier', ...
                    'FontWeight','bold'}, ...
                 'web_pos', [0.5 0.92]);

if length(filepath)>1 & ...
      ((isunix & filepath(1)=='/') | (ispc & filepath(2)==':')),
  video.input_dir= '';
end
filepath= [video.input_dir filepath];

%% checking source file
if ~exist([filepath filename fileext], 'file'),
  error(sprintf('file %s cannot be read', [filepath filename fileext]));
end

%% checking destination file
if ~video.override & exist([video.output_dir video.output_file '.avi'],'file'),
  error('output files exists. use property ''override''.');
end

%% automatic setting of some properties
if isdefault.type,
  if strpatterncmp('replay_*', filename),
    video.type= 'replay';
  elseif strpatterncmp('video_*', filename),
    video.type= 'video';
  end
end
if isdefault.title_text & ~isempty(video.type),
  video.title_text= ['BBCI Feedback (' video.type ')'];
end

%% extracting video size and frame rate from source file, if not defined
if isempty(video.size) | isdefault.fs,
  cmd= sprintf('cd %s; tcprobe -i %s%s;', filepath, filename, fileext);
  [stat,out]= unix(cmd);
  if stat,
    error(sprintf('error using tcprobe (%s -> %s)', cmd, out));
  end
  if isdefault.fs,
    ii= strfind(out, 'frame rate:');
    if length(ii)~=1,
      warning('error retrieving frame rate - unsing default');
    else
      video.fs= sscanf(out(ii+[15:20]), '%f');
    end
  end
  if isempty(video.size),
    iw= strfind(out, 'width=');
    ih= strfind(out, 'height=');
    if length([iw; ih])~=2,
      error('error in retrieving video size');
    end
    video.size= sscanf(out(iw+[6:9]),'%d');
    video.size(2)= sscanf(out(ih+[7:10]),'%d');
  end
  if isdefault.audio,
    ii= strfind(out, 'audio track:');
    if length(ii)~=1,
      error('error in retrieving audio information');
    end
    video.audio= ~strcmp(out(ii+[-3:-2]), 'no');
  end
end


%% brandmark original video with BBCI logo
if video.brand,
  if video.verbose,
    fprintf('branding video %s\n', filename);
  end
  cmd= sprintf('cd %s; brand_bbci %s%s;', filepath, filename, fileext);
  [stat,out]= unix(cmd);
  if stat,
    error(sprintf('could not brand video (%s -> %s)', cmd, out));
  end
  cmd= sprintf('cd %s; mv %s%s %s%s_tmp.avi;', filepath, ...
                 filename, fileext, video.output_dir, video.output_file);
  [stat,out]= unix(cmd);
  if stat,
    error(sprintf('could not rename (%s -> %s)', cmd, out));
  end
  cmd= sprintf('cd %s; mv %s%s_nologo %s%s;', filepath, ...
                 filename, fileext, filename, fileext);
  [stat,out]= unix(cmd);
  if stat,
    error(sprintf('could not rename (%s -> %s)', cmd, out));
  end
  mergefiles= {[video.output_dir video.output_file '_tmp.avi']};
else
  mergefiles= {[filepath filename fileext]};
end


%% prepare figure and bitmap array for intro and/or extro screen
if video.intro | video.extro,
  if ~exist('image_scale','file'),
    addpath('/home/neuro/blanker/matlab/image_stuff');
  end
  
  im= uint8(zeros([video.size([2 1]) 3]));
  im_logo= imread(video.logo_file);

  losz= [size(im_logo,2) size(im_logo,1)];
  clf;
  set(gcf, 'Units','Pixel', 'MenuBar','none', ...
           'Pointer','custom', 'PointerShapeCData',ones(16)*NaN);
%  pos= get(gcf, 'Position');
%  newpos= pos;
  pos_sc= get(0, 'ScreenSize');
  newpos([1 2])= [5 pos_sc(4)-24-video.size(2)];
  newpos([3 4])= video.size;
%  newpos(2)= pos(2)-(newpos(4)-pos(4));
  set(gcf, 'Position',newpos);
  drawnow;
  set(gcf, 'Position',newpos);  %% hack needed 
  drawnow;
  
  xx= linspace(0, 1, video.size(1));
  yy= linspace(0, 1, video.size(2));
end


%% create intro video
if video.intro,
  %% create bitmap for intro screen
  if video.verbose,
    fprintf('creating intro for video %s\n', filename);
  end

  %% render logo
  lsz= video.size*video.logo_size;
  logo_factor= min(lsz./losz);
  im_logo_sc= image_scale(im_logo, logo_factor);
  logo_pos= video.logo_pos(1)*(size(im,2)-size(im_logo_sc,2)+1);
  logo_pos(2)= video.logo_pos(2)*(size(im,1)-size(im_logo_sc,1)+1);
  im= image_paste(im, im_logo_sc, round(logo_pos));
  imagesc(xx, yy, im);
  set(gca, 'Position', [0 0 1 1], 'XTick',[], 'YTick',[]);

  %% render copyright note
  ht= text(video.copyright_pos(1), video.copyright_pos(2), ...
           video.copyright_text);
  set(ht, 'Color',[1 1 1], 'HorizontalAli','center', ...
          'VerticalAli','bottom', ...
          'FontUnits','normalized', ...
          'FontSize',video.copyright_fontsize);
  
  %% render title
  ht= text(video.title_pos(1), video.title_pos(2), ...
           video.title_text);
  set(ht, 'Color',[1 1 1], 'HorizontalAli','center', ...
          'VerticalAli','middle', ...
          'FontUnits','normalized', ...
          'FontSize',video.title_fontsize);
  
  %% choose size of the description box by trial and error
  factor= 1;
  too_small= 1;
  nLines= length(video.desc_text);
  nChars= max(apply_cellwise2(video.desc_text, 'length'));
  while too_small,
    desc_fontsize= factor * min( video.desc_maxsize./[nChars nLines] );
    ht= text(video.desc_pos(1), video.desc_pos(2), video.desc_text);
    set(ht, 'FontUnits','normalized', 'FontSize',desc_fontsize, ...
            'Color',[1 1 1], 'HorizontalAli','center');
    drawnow;
    rect= get(ht, 'Extent');
    too_small= rect(3)<video.desc_maxsize(1) & rect(4)<video.desc_maxsize(2);
    if too_small,
      factor= factor*1.1;
    end
    delete(ht);
  end
  factor= factor/1.1;
  
  %% render description text
  desc_fontsize= factor * min( video.desc_maxsize./[nChars nLines] );
  ht= text(video.desc_pos(1), video.desc_pos(2), video.desc_text);
  set(ht, 'FontUnits','normalized', 'FontSize',desc_fontsize, ...
          'Color',[1 1 1], 'HorizontalAli','center');
  drawnow;
  rect= get(ht, 'Extent');

  %% render description frame
  hl= line([rect(1)+rect(3) rect(1)+rect(3) rect(1) rect(1);
            rect(1)+rect(3) rect(1) rect(1) rect(1)+rect(3)] + ...
           video.desc_boxgap*[1 1 -1 -1; 1 -1 -1 1], ...
           [rect(2)-rect(4) rect(2) rect(2) rect(2)-rect(4);
            rect(2) rect(2) rect(2)-rect(4) rect(2)-rect(4)] + ...
           video.desc_boxgap*[-1 1 1 -1; 1 1 -1 -1]);
  set(hl, 'LineWidth',2, 'Color',[1 1 1]);
  
  
  %% save intro in an AVI video file
  movie= avifile([video.output_dir 'intro_tmp.avi'], ...
                 'Compression','none', 'Quality',100, ...
                 'fps',video.fs);
  figure(gcf); drawnow;
  F= getframe(gcf);
  
  %% intro fade in
  nFrames1= video.fs * video.intro_fadein;
  for i= 1:nFrames1,
    fadefactor= (i-1)/nFrames1;
    Ff= F;
    Ff.cdata= uint8( double(F.cdata) * fadefactor );
    movie= addframe(movie, Ff);
  end
  
  %% intro screen
  nFrames2= video.fs * video.intro_stay;
  for i= 1:nFrames2,
    movie= addframe(movie, F);
  end
  
  %% intro fade out
  nFrames3= video.fs * video.intro_fadeout;
  for i= 1:nFrames3,
    fadefactor= (nFrames3-i)/nFrames3;
    Ff= F;
    Ff.cdata= uint8( double(F.cdata) * fadefactor );
    movie= addframe(movie, Ff);
  end
  
  movie= close(movie);
  
  %% generate audio track for intro if neccessary
  if video.audio,
    audio_file= [video.output_dir 'intro.wav'];
    full_length= ceil((nFrames1+nFrames2+nFrames3)/video.fs*video.audio_fs);
    wave= zeros(full_length, 2);
    wavwrite(wave, video.audio_fs, 16, audio_file);
    if video.verbose,
      fprintf('audio track for intro written (%d samples at %d Hz).\n', ...
              full_length, video.audio_fs);
    end
    optaudio= '-p intro.wav';
  else
    optaudio= '';
  end
  
  %% compress video file (XVID codec)
  cmd= sprintf('cd %s; transcode -i intro_tmp.avi %s -use_rgb -z -y xvid -o intro.avi',...
               video.output_dir, optaudio);
  [stat, out]= unix(cmd);
  if stat~=0,
    error(sprintf('error using transcode (%s -> %s)', cmd, out));
  end
  mergefiles= cat(2, {[video.output_dir 'intro.avi']}, mergefiles);
end


%% create extro video
if video.extro,
  %% create bitmap for extro screen
  if video.verbose,
    fprintf('creating extro for video %s\n', filename);
  end
  
  clf;
  im(:)= 0;

  %% render logo
  lsz= video.size*video.extro_logo_size;
  logo_factor= min([1 lsz./losz]);
  im_logo_sc= image_scale(im_logo, logo_factor);
  logo_pos= video.extro_logo_pos(1)*(size(im,2)-size(im_logo_sc,2)+1);
  logo_pos(2)= video.extro_logo_pos(2)*(size(im,1)-size(im_logo_sc,1)+1);
  im= image_paste(im, im_logo_sc, round(logo_pos));
  imagesc(xx, yy, im);
  set(gca, 'Position', [0 0 1 1], 'XTick',[], 'YTick',[]);

  %% render web address
  ht= text(video.web_pos(1), video.web_pos(2), video.web_text);
  set(ht, 'Color',[1 1 1], 'HorizontalAli','center', ...
          'VerticalAli','bottom', ...
          'FontUnits','normalized', ...
          video.web_textspec{:});
  
  %% save extro in an AVI video file
  movie= avifile([video.output_dir 'extro_tmp.avi'], ...
                 'Compression','none', 'Quality',100, ...
                 'fps',video.fs);
  figure(gcf); drawnow;  
  F= getframe(gcf);
  
  %% extro fade in
  nFrames1= video.fs * video.extro_fadein;
  for i= 1:nFrames1,
    fadefactor= (i-1)/nFrames1;
    Ff= F;
    Ff.cdata= uint8( double(F.cdata) * fadefactor );
    movie= addframe(movie, Ff);
  end
  
  %% extro screen
  nFrames2= video.fs * video.extro_stay;
  for i= 1:nFrames2,
    movie= addframe(movie, F);
  end
  
  %% extro fade out
  nFrames3= video.fs * video.extro_fadeout;
  for i= 1:nFrames3,
    fadefactor= (nFrames3-i)/nFrames3;
    Ff= F;
    Ff.cdata= uint8( double(F.cdata) * fadefactor );
    movie= addframe(movie, Ff);
  end
  
  %% extro blank screen
  nFrames4= video.fs * video.extro_blank;
  F.cdata= uint8( zeros(size(F.cdata)) );
  for i= 1:nFrames4,
    movie= addframe(movie, F);
  end
  movie= close(movie);
  
  %% generate audio track for extro if neccessary
  if video.audio,
    audio_file= [video.output_dir 'extro.wav'];
    full_length= ceil((nFrames1+nFrames2+nFrames3+nFrames4) ...
                      /video.fs*video.audio_fs);
    wave= zeros(full_length, 2);
    wavwrite(wave, video.audio_fs, 16, audio_file);
    if video.verbose,
      fprintf('audio track for extro written (%d samples at %d Hz).\n', ...
              full_length, video.audio_fs);
    end
    optaudio= '-p extro.wav';
  else
    optaudio= '';    
  end
  
  %% compress video file (XVID codec)
  cmd= sprintf('cd %s; transcode -i extro_tmp.avi %s -use_rgb -z -y xvid -o extro.avi',...
               video.output_dir, optaudio);
  [stat, out]= unix(cmd);
  if stat~=0,
    error(sprintf('error using transcode (%s -> %s)', cmd, out));
  end

  mergefiles= cat(2, mergefiles, {[video.output_dir 'extro.avi']});
end


%% merging files together
if video.verbose,
  fprintf('merging files for video %s\n', filename);
end
cmd= sprintf('avimerge -o %s%s.avi -i %s', ...
             video.output_dir, video.output_file, ...
             vec2str(mergefiles,'%s',' '));
[stat,out]= unix(cmd);
if stat~=0,
  error(sprintf('error using avimerge (%s -> %s)', cmd, out));
end


%% deleting temporary files
cmd= sprintf('cd %s; rm -f intro*.* extro*.* *_tmp.avi', video.output_dir);
[stat,out]= unix(cmd);
if stat~=0,
  error(sprintf('error deleting tmp files (%s -> %s)', cmd, out));
end

set(gcf, 'MenuBar','figure', 'Pointer','arrow');
