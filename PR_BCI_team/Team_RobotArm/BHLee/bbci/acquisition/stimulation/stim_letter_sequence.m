function stim_letter_sequence(nBlocks, varargin);
%STIM_LETTER_SEQUENCE - Show sequence of letters
%
%Description:
% This function displays a blockwise a sequence of letters. Each block
% contains the whole sequence of letters in randomized order. Letters
% are displayed interleaved with a fixation cross.
%
%Synopsis:
% stim_letter_sequence(nBlocks, OPT)
%
%Arguments:
% nBlocks: number of blocks
% OPT: struct of property/value list of optional properties:
%  'seq': sequence of letter stimuli, default 'ENISRATULGMOFKP'.
%  'duration_cue': duration for which the cue (letters) is displayed [ms],
%     default: 1500.
%  'isi': inter-stimulus interval [ms], default 3000.
%  'fixation': switch to show (1) or not to show (0) a fixation cross.
%  'break_every': number, after this many blocks a break is given.
%  'duration_break': duration of the break after OPT.break_every many blocks,
%     default: 5.
%
%Markers sent to parallel port:
% 001 - 026: code of letter (1='A', ..., 26='Z')
% 060: letter disappears, fixation cross appears
% 240: start of countdown
% 241: end of countdown
% 249: start of break
% 250: end of break
% 251: beginning of initial relax period
% 252: end of initial relax period
% 253: beginning of final relax period
% 254: end of final relax period

% blanker@cs.tu-berlin.de

global DATA_DIR

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'seq', 'ENISRATULGMOFKP', ...
                  'breakFactor', 1,...
                  'duration_cue', 1500,...
                  'isi', 3000,...
									'color_background',0.3*[1 1 1], ...
                  'countdown', 3, ...
                  'duration_break', 15*1000, ...
                  'break_every', 5, ...
									'msg_relax', 'entspannen', ...
                  'msg_break', 'kurze Pause', ...
                  'sound_file', [DATA_DIR 'sound/A5.wav'], ...
                  'fixation', 1, ...
                  'fixation_vpos', 0.57, ...
                  'fixation_size', 0.04, ...
                  'fixation_spec', {'LineWidth',12, 'Color',[0 0 0]}, ...
                  'lett_vpos', 0.55, ...
                  'lett_spec', {'FontSize',0.2, 'FontWeight','bold', ...
                                'Color',[0 0 0]}, ...
									'msg_vpos', 0.57, ...
									'msg_spec', {'FontSize',0.1, 'FontWeight','bold', ...
                                'Color',[0 0 0]});

dur= nBlocks*(length(opt.seq)*opt.isi/1000 + opt.countdown) + ...
     floor((nBlocks-1)/opt.break_every)*opt.duration_break/1000 + ...
     3 + 30*opt.breakFactor;     
fprintf('approximate duration: %.1fs\n', dur);

if ~isempty(opt.sound_file),
  [snd_wav, snd_fs]= wavread(opt.sound_file);
end

clf;
ppTrigger(251);
figureMaximize;
set(gcf,'Menubar','none', 'Toolbar','none', ...
        'Color',opt.color_background, 'DoubleBuffer','on', ...
        'Colormap',gray(256));
text_spec= {'HorizontalAli','center', 'VerticalAli','middle', ...
           'FontUnits','normalized'};
fp= get(gcf, 'Position');
h_ax_msg= axes('Position',[0 0 1 1]);
set([h_ax_msg], 'Visible','off');
h_msg= text(0.5, opt.msg_vpos, opt.msg_relax, text_spec{:}, opt.msg_spec{:});
h_lett= text(0.5, opt.lett_vpos, ' ', text_spec{:}, opt.lett_spec{:});
fix_w= opt.fixation_size;
fix_h= fix_w/fp(4)*fp(3);
set(h_ax_msg, 'XLim',[0 1], 'YLim',[0 1]);
h_fix= line(0.5 + [-fix_w fix_w; 0 0]', ...
            opt.fixation_vpos + [0 0; -fix_h fix_h]', ...
            opt.fixation_spec{:});
set(h_fix, 'Visible','off');
pause(1+19*opt.breakFactor);
set(h_msg, 'Visible','off');
ppTrigger(252);
pause(1);
do_countdown= 1;

for blockno= 1:nBlocks,
  
waitForSync;
rseq= randperm(length(opt.seq));
if do_countdown,
  if ~isempty(opt.sound_file),
    wavplay(snd_wav, snd_fs, 'async');
  end
  stimutil_countdown(h_lett, opt.countdown);
  set(h_fix, 'Visible','on');
  drawnow;
  waitForSync(opt.isi - opt.duration_cue);
  do_countdown= 0;
end

waitForSync;
for ii= 1:length(opt.seq),
  ch= opt.seq(rseq(ii));
  set(h_fix, 'Visible','off');
  set(h_lett, 'String',ch);
  drawnow;
  ppTrigger(ch-'A'+1);
  waitForSync(opt.duration_cue);
  set(h_lett, 'String',' ');
  set(h_fix, 'Visible','on');
  drawnow;
  ppTrigger(60);
  waitForSync(opt.isi - opt.duration_cue);
end

set(h_fix, 'Visible','off');
if blockno<nBlocks & mod(blockno,opt.break_every)==0,
  if ~isempty(opt.sound_file),
    wavplay(snd_wav, snd_fs, 'async');
  end
  set(h_msg, 'String', opt.msg_break, 'Visible','on');
  drawnow;
  ppTrigger(249);
  waitForSync(opt.duration_break);
  set(h_msg, 'Visible','off');
  drawnow;
  ppTrigger(250);
  waitForSync(1000);
  do_countdown= 1;
end

end

set(h_msg, 'String', opt.msg_relax, 'Visible','on');
ppTrigger(253);
pause(1+9*opt.breakFactor);
set(h_msg, 'String', 'fin');

ppTrigger(254);
