function q = yesno_input()

while true
    user_entry = input('Flicker seen? 1 (yes) or 2 (no): ','s');
    if isempty(user_entry)
        user_entry = 'x';
    end
    user_entry = user_entry(end);
    if any(user_entry == '12')
        break
    end
end
    
q = str2double(user_entry);

