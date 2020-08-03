mrk.fs = 1000;

% more than one keypress
mrk.desc = {'TS', '0', 'KP', '9', '18', '27', 'TE', ...
            'TS', '0', 'KP', '9', '18', 'KP', '27', 'TE'};
mrk.pos = [0 5 9 10 15 20 25 ...
          100 105 109 110 115 119 120 125] * 100;
mrk1 = mrk_addnokeypress(mrk)

% no keypress
mrk.desc = {'TS', '0', '9', '18', '27', 'TE', ...
            'TS', '0', 'KP', '9', '18', '27', 'TE'};
mrk.pos = [0 5 10 15 20 25  ...
           100 105 109 110 115 120 125] * 100;
mrk2 = mrk_addnokeypress(mrk)

% keypress after target
mrk.desc = {'TS', '0', '9', '18', '27', 'KP', 'TE'};
mrk.pos = [0 5 10 15 20 21 25] * 100;
mrk3 = mrk_addnokeypress(mrk)

% keypress before target
mrk.desc = {'TS', '0', '9', '18', 'KP', '27', 'TE'};
mrk.pos = [0 5 10 15 19 20 25] * 100;
mrk4 = mrk_addnokeypress(mrk)

% keypress at the first target
mrk.desc = {'TS', '0', 'KP', '9', '18', '27', 'TE'};
mrk.pos = [0 5 6 10 15 20 25] * 100;
mrk5 = mrk_addnokeypress(mrk)

% keypress between trials
mrk.desc = {'KP' 'TS', '0', '9', '18', '27', 'KP', 'TE', 'KP', 'KP'...
            'TS', '0', 'KP', '9', '18', '27', 'TE', 'KP'};
mrk.pos = [0 2 5 10 15 20 25 26 50 70 ...
           100 105 109 110 115 120 125 130] * 100;
mrk6 = mrk_addnokeypress(mrk)

% keypress too unprecise
mrk.desc = {'TS', '0', 'KP', '9', '18', '27', 'TE'};
mrk.pos = [0 5 7.5 10 15 20 25] * 100;
mrk7 = mrk_addnokeypress(mrk)


