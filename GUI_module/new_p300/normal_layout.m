function normal_layout(w, train_character, spell_char, loc_layout, text_size)

%%Normal layout
for i = 1:length(spell_char)
    Screen('TextFont',w, 'Arial');
    Screen('TextSize',w, text_size);
    Screen('TextStyle', w, 0);
    %     Screen('TextStyle', w, 1+2);
    textbox = Screen('TextBounds', w, spell_char(i));
    Screen('DrawText', w, spell_char(i), loc_layout(i,1)-(textbox(3)/2), ...
        loc_layout(i,2)-(textbox(4)/2), [100, 100, 100]);
end

%Display characters to be trained
Screen('TextFont',w, 'Arial');
Screen('TextSize',w, ceil(text_size/2.5));
Screen('TextStyle', w, 0);
Screen('DrawText', w, train_character, 10, 10, [255, 255, 255]); % location 10='x_axis' , 10='y_axis', 'color' 

Screen('DrawLine', w, [50 50 50], 0, 140, 1920, 140, 5);