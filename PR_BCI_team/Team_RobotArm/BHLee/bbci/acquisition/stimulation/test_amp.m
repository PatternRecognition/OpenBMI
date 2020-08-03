function []= test_amp(filename,varargin)

% OPT: struct or property/value list of optional arguments:
% 'breaks': 
% 'msg_vpos': Scalar. Vertical position of message text object. Default: 0.57.
% 'msg_spec': Cell array. Text object specifications for message text object.
%   Default: {'FontSize',0.1, 'FontWeight','bold', 'Color',[.9 0 0]})

% blanker@cs.tu-berlin.de, Jul-2007


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'test', 0, ...
                  'bv_host', 'localhost', ...
                  'breaks',inf, ...
                  'break_minevents',7, ...
                  'break_markers', [249 250], ...
                  'break_msg', 'Short break for %d s', ...
                  'break_countdown', 7, ...
                  'msg_fin','fin',...
                  'repetitions',1,...
                  'pause',600);




if ~isempty(opt.bv_host),
  bvr_checkparport;
end

for k = 1:opt.repetitions
    pause(10)
    if ~opt.test,
        if ~isempty(filename),
        bvr_startrecording([filename]);
        end
    end

    
    ppTrigger(251);
    
    pause(opt.pause);
    %hier zweischen muss gezaehlt werden

    ppTrigger(254);


    pause(1);
    if ~opt.test & ~isempty(filename),
        bvr_sendcommand('stoprecording');
    end
    if k<opt.repetitions
      pause(3000);
    end
end

return;