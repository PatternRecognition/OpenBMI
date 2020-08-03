function res = waitForMouseNonBlocking()

[x,y,buttons] = GetMouse;

res = any(buttons);