
## Plays Video and Closes Window automatically

import wx
import wx.media

def close(event):
    #frame.Close()
    frame.Destroy()

app = wx.PySimpleApp()
frame = wx.Frame(None, -1)#size = (1000, 500))
#frame.CenterOnScreen()
#frame.Maximize()

mc = wx.media.MediaCtrl(frame)
path = '/home/lena/Desktop/Perspektive_Movie/phy.avi'
mc.Load(path)
mc.Play()
frame.ShowFullScreen(True)

wx.media.EVT_MEDIA_STOP(frame,-1,close)

app.MainLoop()

















