Set objArgs = WScript.Arguments
If objArgs.Count < 1 Then
  waitingtime= 2000
Else
  waitingtime= objArgs(0)
End If

Set Rec = CreateObject("VisionRecorder.Application")

If Rec.State <> 1 Then  ' 1 is vrStateMonitoring
  Rec.Acquisition.ViewData()
  WScript.Sleep waitingtime 
End If
  
