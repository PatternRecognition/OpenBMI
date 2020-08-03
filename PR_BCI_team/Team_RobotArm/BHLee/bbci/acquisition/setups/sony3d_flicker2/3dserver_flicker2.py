
#!/usr/bin/env python

# Rafael Schultze-Kraft, Markus Wenzel, mar 2012
# code adapted from marton danoczy, jan 2011
# code adapted from http://docs.python.org/library/socketserver.html

import socket
import threading
import SocketServer
import win32com.client
import pythoncom
import shutil
import os
from hashlib import md5
from base64 import urlsafe_b64encode
from time import sleep

HOST     = "sony3d.ml.tu-berlin.de"
PORT     = 12345
BUFSIZ   = 2048
wav      = "D:\\sony3d_vertical_shift_study\\Messages\\blank.wav"
temp_dir = "D:\\sony3d_vertical_shift_study\\temp"

class ThreadedTCPRequestHandler(SocketServer.BaseRequestHandler):
    def setup(self):
        print "starting thread %s" % threading.currentThread()
        self.ok = False
        pythoncom.CoInitialize()
        self.player = win32com.client.Dispatch("StereoPlayer.Automation")
        self.wait()
        f = "D:\\sony3d_flicker2\\Messages\\black.png" 
        self.player.OpenLeftRightFiles(f, f, wav, 1)
        self.wait()
        self.player.EnterFullscreenMode(True)
        self.wait()
        self.ok = True
    
    def wait(self):
        ready = False
        while not ready:
            ready = self.player.GetReady()

    def process(self, s):
        if len(s)==0:
            return
        cmd, param = s.split(' ', 1)
        if cmd == 'pic':
            self.show_pic(param)
        elif cmd == 'jpg':
            self.show_jpg(param)
        elif cmd == 'msg':
            self.show_msg(param)
        elif cmd == 'vid':
            self.show_vid(param)
        elif cmd == 'freq':
            self.change_freq(param)

    # In the vertical shift study left and right image are in one png file:
    def show_pic(self, fname):
        self.ok = False;
        f = "D:\\sony3d_vertical_shift_study\\Images\\%s.png" % fname
        print "Opening: %s" % fname
        # If file names are too long, COM fails. Workaround: copy files
        hash = urlsafe_b64encode(md5(fname).digest())
        t = "D:\\sony3d_vertical_shift_study\\temp\\%s.png" % hash;
        shutil.copyfile(f, t)
      #  self.player.OpenLeftRightFiles(t, t, wav, 1)
        self.player.OpenFile(t)
        self.wait()
        self.ok = True;

    def show_msg(self, fname):
        self.ok = False;
        f = "D:\\sony3d_flicker2\\Messages\\%s.png" % fname
        print "Opening: %s" % fname
        self.player.OpenLeftRightFiles(f, f, wav, 1)
        self.wait()
        self.ok = True;

    def change_freq(self, params):
        params = params.rsplit(' ')
        self.ok = False
        os.system('d:\sony3d_flicker2\qres.exe /x:%(x)s /y:%(y)s /r:%(rate)s /d' %{'x':params[0], 'y':params[1], 'rate':params[2]})
        print 'frequency changed to %s Hz...' %params[2]
        self.wait()
        self.ok = True
        
    def show_jpg(self, fname):
        self.ok = False;
        f = "D:\\sony3d_flicker2\\Images\\%s.jpg" % fname
        print "Opening: %s" % fname
        self.player.OpenLeftRightFiles(f, f, wav, 1)
        self.wait()
        self.ok = True;

    def show_vid(self, s):
        self.ok = False;
        try:
            secs, fname = s.split(' ', 1)
            #fname = s
            fvid = "D:\\sony3d_flicker2\\Videos\\%s" % fname
            print "Opening: %s" % fname
            print self.player
            #self.player.OpenLeftRightFiles(fvid, fvid, "", 2)
            self.player.OpenFile(fvid)
            self.player.SetPosition(float(secs))
            self.player.SetPlaybackState(0)
        except ValueError:
            self.player.SetPlaybackState(1)
        self.wait()
        self.ok = True;
    
    def handle(self):
        while self.ok:
            data = self.request.recv(BUFSIZ)
            if not data:
                break
            else:
                self.process(data.strip())

    def finish(self):
        print "stopping thread %s" % threading.currentThread()

# SONY3D HORIZONTAL SHIFT METHODS (MARTON)        
  #  def show_pic(self, fname):
  #      self.ok = False;
  #      f1 = "D:\\101215_Cube\\%s_l.png" % fname
  #      f2 = "D:\\101215_Cube\\%s_r.png" % fname
  #      print "Opening: %s" % fname
  #      # If file names are too long, COM fails. Workaround: copy files
  #      hash = urlsafe_b64encode(md5(fname).digest())
  #      t1 = "D:\\project_sony3d\\temp\\%s_l.png" % hash;
  #      t2 = "D:\\project_sony3d\\temp\\%s_r.png" % hash;
  #      shutil.copyfile(f1, t1)
  #      shutil.copyfile(f2, t2)
  #      self.player.OpenLeftRightFiles(t1, t2, wav, 1)
  #      self.wait()
  #      self.ok = True;

  #  def show_msg(self, fname):
  #      self.ok = False;
  #      f = "D:\\project_sony3d\\%s.png" % fname
  #      print "Opening: %s" % fname
  #      self.player.OpenLeftRightFiles(f, f, wav, 1)
  #      self.wait()
  #      self.ok = True;
          
class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    pass

if __name__ == "__main__":

    server_set_up = False
    while not server_set_up:
        try:
            server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
            server_set_up = True
        except socket.gaierror:
            print "Couldn't start listening: maybe network cable isn't plugged in?"
            sleep(2)
        
    ip, port = server.server_address

    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target=server.serve_forever)

    # Exit the server thread when the main thread terminates
    server_thread.setDaemon(True)
    server_thread.start()

    try:
        os.mkdir(temp_dir)
    except:
        pass
    
    print "Server is up and running on %s:%i" % (HOST, port) 

    try:
        print "Press CTRL+C to shutdown."
        while True:
            pass
    except:
        print

    print "Shutting down..."
    server.shutdown()
 
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    
    print "done."